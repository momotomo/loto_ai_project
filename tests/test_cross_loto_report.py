"""Tests for cross_loto_report.py.

Covers:
- Markdown report generation (required sections, content correctness)
- CSV artifact required columns
- Recommendation decision logic format
- Run metadata tracking
- CLI --report_only / --skip_training smoke
"""

from __future__ import annotations

import csv
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

from cross_loto_report import (
    CROSS_LOTO_REPORT_SCHEMA_VERSION,
    build_markdown_report,
    build_pairwise_comparisons_csv,
    build_recommendation_summary_csv,
    build_run_metadata,
    build_variant_metrics_csv,
    save_report_artifacts,
)
from cross_loto_summary import (
    build_cross_loto_summary,
    build_recommendation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_comparison_summary(
    loto_type: str,
    variant_names: list[str],
    *,
    logloss_base: float = 0.300,
    promote: bool = False,
    seeds: list[int] | None = None,
    run_count: int = 1,
) -> dict:
    """Build a minimal comparison_summary dict for testing."""
    seeds = seeds or [42]
    variants = {}
    for v in variant_names:
        variants[v] = {
            "run_count": run_count,
            "logloss": {"mean": logloss_base, "std": 0.001, "values": [logloss_base] * run_count},
            "brier": {"mean": 0.040, "std": 0.001, "values": [0.040] * run_count},
            "ece": {"mean": 0.010, "std": 0.001, "values": [0.010] * run_count},
            "calibration_recommendations": {"none": run_count},
            "promote_count": (1 if promote and v != "legacy" else 0),
            "hold_count": (0 if promote and v != "legacy" else 1),
        }
    pairwise_comparisons = {
        "settransformer_vs_deepsets": {
            "run_count": run_count,
            "ci_wins": run_count if promote else 0,
            "permutation_wins": run_count if promote else 0,
            "both_pass_count": run_count if promote else 0,
        },
        "deepsets_vs_multihot": {
            "run_count": run_count,
            "ci_wins": 0,
            "permutation_wins": 0,
            "both_pass_count": 0,
        },
    }
    return {
        "schema_version": 1,
        "loto_type": loto_type,
        "preset": "archcomp",
        "seeds": seeds,
        "run_count": run_count,
        "alpha": 0.05,
        "variants": variants,
        "pairwise_comparisons": pairwise_comparisons,
    }


def _make_cross_loto_and_rec(
    loto_types: list[str] | None = None,
    variant_names: list[str] | None = None,
    promote: bool = False,
    run_count: int = 1,
) -> tuple[dict, dict]:
    """Build a minimal cross_loto_summary and recommendation for testing."""
    if loto_types is None:
        loto_types = ["loto6", "miniloto"]
    if variant_names is None:
        variant_names = ["legacy", "deepsets", "settransformer"]
    per_loto = {
        lt: _make_comparison_summary(
            lt, variant_names, promote=promote, run_count=run_count
        )
        for lt in loto_types
    }
    cross_summary = build_cross_loto_summary(
        per_loto_summaries=per_loto,
        loto_types=loto_types,
        preset="archcomp",
        seeds=[42],
        alpha=0.05,
    )
    rec = build_recommendation(cross_summary)
    return cross_summary, rec


# ---------------------------------------------------------------------------
# 1. Markdown report — required sections
# ---------------------------------------------------------------------------


class TestMarkdownReportSections:
    def test_required_h2_sections_present(self):
        cross_summary, rec = _make_cross_loto_and_rec()
        md = build_markdown_report(cross_summary, rec)
        required = [
            "## Execution Conditions",
            "## Variant Metrics Summary",
            "## Promote / Hold Summary",
            "## Pairwise Comparison Summary",
            "## Recommendation",
            "## Decision Rules",
            "## Production Change Rationale",
            "## Calibration Recommendations",
        ]
        for section in required:
            assert section in md, f"Missing section: {section}"

    def test_header_present(self):
        cross_summary, rec = _make_cross_loto_and_rec()
        md = build_markdown_report(cross_summary, rec)
        assert "# Cross-Loto Architecture Comparison Report" in md

    def test_loto_types_in_execution_conditions(self):
        cross_summary, rec = _make_cross_loto_and_rec(loto_types=["loto6", "miniloto"])
        md = build_markdown_report(cross_summary, rec)
        assert "loto6" in md
        assert "miniloto" in md

    def test_variants_appear_in_report(self):
        cross_summary, rec = _make_cross_loto_and_rec(variant_names=["legacy", "deepsets"])
        md = build_markdown_report(cross_summary, rec)
        assert "legacy" in md
        assert "deepsets" in md

    def test_recommended_next_action_in_report(self):
        cross_summary, rec = _make_cross_loto_and_rec()
        md = build_markdown_report(cross_summary, rec)
        action = rec.get("recommended_next_action") or ""
        assert action in md

    def test_decision_rules_thresholds_documented(self):
        cross_summary, rec = _make_cross_loto_and_rec()
        md = build_markdown_report(cross_summary, rec)
        # Decision rule thresholds must be explicitly stated
        assert "0.5" in md
        assert "CONSISTENT_PROMOTE_THRESHOLD" in md
        assert "PAIRWISE_SIGNAL_THRESHOLD" in md

    def test_production_rationale_present(self):
        cross_summary, rec = _make_cross_loto_and_rec(promote=False)
        md = build_markdown_report(cross_summary, rec)
        assert "Production" in md or "production" in md

    def test_run_metadata_included_when_provided(self):
        cross_summary, rec = _make_cross_loto_and_rec()
        meta = build_run_metadata(
            loto_types=["loto6"],
            preset="archcomp",
            seeds=[42],
            evaluation_model_variants="legacy,deepsets",
            source_summary_paths={"loto6": "data/comparison_summary_loto6.json"},
            run_id="test_run_001",
        )
        md = build_markdown_report(cross_summary, rec, run_metadata=meta)
        assert "test_run_001" in md
        assert "comparison_summary_loto6.json" in md

    def test_run_metadata_omitted_when_none(self):
        cross_summary, rec = _make_cross_loto_and_rec()
        md = build_markdown_report(cross_summary, rec, run_metadata=None)
        # Should not crash and still produce sections
        assert "## Execution Conditions" in md

    def test_pma_isab_section_when_signal(self):
        """Report notes PMA/ISAB when settransformer beats deepsets."""
        cross_summary, rec = _make_cross_loto_and_rec(
            loto_types=["loto6", "miniloto"], promote=True, run_count=3
        )
        md = build_markdown_report(cross_summary, rec)
        if rec.get("whether_to_try_pma_or_isab_next"):
            assert "PMA" in md or "ISAB" in md

    def test_pairwise_table_contains_both_pass(self):
        cross_summary, rec = _make_cross_loto_and_rec()
        md = build_markdown_report(cross_summary, rec)
        assert "both_pass" in md.lower() or "Both" in md

    def test_promote_hold_table_present(self):
        cross_summary, rec = _make_cross_loto_and_rec()
        md = build_markdown_report(cross_summary, rec)
        assert "promote_count" in md or "promote" in md.lower()
        assert "hold_count" in md or "hold" in md.lower()


# ---------------------------------------------------------------------------
# 2. Variant metrics CSV
# ---------------------------------------------------------------------------


VARIANT_METRICS_REQUIRED_COLS = [
    "rank",
    "variant",
    "logloss_mean",
    "logloss_std",
    "brier_mean",
    "brier_std",
    "ece_mean",
    "ece_std",
    "promote_count_total",
    "hold_count_total",
    "promote_rate",
    "consistent_promote",
    "consistent_hold",
    "loto_types_evaluated",
]


class TestVariantMetricsCsv:
    def test_required_columns_present(self):
        cross_summary, _ = _make_cross_loto_and_rec()
        csv_text = build_variant_metrics_csv(cross_summary)
        reader = csv.DictReader(io.StringIO(csv_text))
        fieldnames = reader.fieldnames or []
        for col in VARIANT_METRICS_REQUIRED_COLS:
            assert col in fieldnames, f"Missing column: {col}"

    def test_row_count_matches_variant_count(self):
        variant_names = ["legacy", "deepsets", "settransformer"]
        cross_summary, _ = _make_cross_loto_and_rec(variant_names=variant_names)
        csv_text = build_variant_metrics_csv(cross_summary)
        reader = csv.DictReader(io.StringIO(csv_text))
        rows = list(reader)
        assert len(rows) == len(variant_names)

    def test_variant_names_in_csv(self):
        variant_names = ["legacy", "deepsets"]
        cross_summary, _ = _make_cross_loto_and_rec(variant_names=variant_names)
        csv_text = build_variant_metrics_csv(cross_summary)
        reader = csv.DictReader(io.StringIO(csv_text))
        variants_in_csv = {row["variant"] for row in reader}
        assert variants_in_csv == set(variant_names)

    def test_consistent_promote_is_boolean_string(self):
        cross_summary, _ = _make_cross_loto_and_rec(promote=False)
        csv_text = build_variant_metrics_csv(cross_summary)
        reader = csv.DictReader(io.StringIO(csv_text))
        for row in reader:
            assert row["consistent_promote"] in ("true", "false")
            assert row["consistent_hold"] in ("true", "false")

    def test_loto_types_evaluated_pipe_separated(self):
        cross_summary, _ = _make_cross_loto_and_rec(loto_types=["loto6", "miniloto"])
        csv_text = build_variant_metrics_csv(cross_summary)
        reader = csv.DictReader(io.StringIO(csv_text))
        rows = list(reader)
        # All rows should have pipe-separated loto_types
        for row in rows:
            lt_eval = row["loto_types_evaluated"]
            # Either single loto or pipe-separated multiple
            assert "|" in lt_eval or lt_eval in {"loto6", "miniloto", "loto7"}


# ---------------------------------------------------------------------------
# 3. Pairwise comparisons CSV
# ---------------------------------------------------------------------------


PAIRWISE_REQUIRED_COLS = [
    "comparison_key",
    "scope",
    "loto_type",
    "run_count",
    "ci_wins",
    "permutation_wins",
    "both_pass_count",
    "both_pass_rate",
]


class TestPairwiseCsv:
    def test_required_columns_present(self):
        cross_summary, _ = _make_cross_loto_and_rec()
        csv_text = build_pairwise_comparisons_csv(cross_summary)
        reader = csv.DictReader(io.StringIO(csv_text))
        fieldnames = reader.fieldnames or []
        for col in PAIRWISE_REQUIRED_COLS:
            assert col in fieldnames, f"Missing column: {col}"

    def test_overall_row_present_for_each_key(self):
        cross_summary, _ = _make_cross_loto_and_rec()
        csv_text = build_pairwise_comparisons_csv(cross_summary)
        reader = csv.DictReader(io.StringIO(csv_text))
        rows = list(reader)
        overall_rows = [r for r in rows if r["scope"] == "overall"]
        pairwise = cross_summary.get("pairwise_comparison_summary") or {}
        assert len(overall_rows) == len(pairwise)

    def test_per_loto_rows_present(self):
        cross_summary, _ = _make_cross_loto_and_rec(loto_types=["loto6", "miniloto"])
        csv_text = build_pairwise_comparisons_csv(cross_summary)
        reader = csv.DictReader(io.StringIO(csv_text))
        rows = list(reader)
        per_loto_rows = [r for r in rows if r["scope"] == "per_loto"]
        assert len(per_loto_rows) > 0

    def test_both_pass_rate_is_numeric(self):
        cross_summary, _ = _make_cross_loto_and_rec()
        csv_text = build_pairwise_comparisons_csv(cross_summary)
        reader = csv.DictReader(io.StringIO(csv_text))
        for row in reader:
            # Should be parseable as float
            rate = float(row["both_pass_rate"])
            assert 0.0 <= rate <= 1.0


# ---------------------------------------------------------------------------
# 4. Recommendation summary CSV
# ---------------------------------------------------------------------------


REC_SUMMARY_REQUIRED_COLS = [
    "generated_at",
    "loto_types",
    "preset",
    "seeds",
    "run_count_total",
    "recommended_next_action",
    "recommended_challenger",
    "keep_production_as_is",
    "whether_to_try_pma_or_isab_next",
    "best_variant_by_logloss",
    "consistent_promote_variants",
    "blockers_count",
    "next_experiment_count",
]


class TestRecommendationSummaryCsv:
    def test_required_columns_present(self):
        cross_summary, rec = _make_cross_loto_and_rec()
        csv_text = build_recommendation_summary_csv(cross_summary, rec)
        reader = csv.DictReader(io.StringIO(csv_text))
        fieldnames = reader.fieldnames or []
        for col in REC_SUMMARY_REQUIRED_COLS:
            assert col in fieldnames, f"Missing column: {col}"

    def test_exactly_one_data_row(self):
        cross_summary, rec = _make_cross_loto_and_rec()
        csv_text = build_recommendation_summary_csv(cross_summary, rec)
        reader = csv.DictReader(io.StringIO(csv_text))
        rows = list(reader)
        assert len(rows) == 1

    def test_keep_production_is_boolean_string(self):
        cross_summary, rec = _make_cross_loto_and_rec(promote=False)
        csv_text = build_recommendation_summary_csv(cross_summary, rec)
        reader = csv.DictReader(io.StringIO(csv_text))
        row = next(reader)
        assert row["keep_production_as_is"] in ("true", "false")
        assert row["whether_to_try_pma_or_isab_next"] in ("true", "false")

    def test_recommended_next_action_matches_recommendation(self):
        cross_summary, rec = _make_cross_loto_and_rec()
        csv_text = build_recommendation_summary_csv(cross_summary, rec)
        reader = csv.DictReader(io.StringIO(csv_text))
        row = next(reader)
        assert row["recommended_next_action"] == rec.get("recommended_next_action")

    def test_blockers_count_is_int_string(self):
        cross_summary, rec = _make_cross_loto_and_rec()
        csv_text = build_recommendation_summary_csv(cross_summary, rec)
        reader = csv.DictReader(io.StringIO(csv_text))
        row = next(reader)
        assert int(row["blockers_count"]) >= 0


# ---------------------------------------------------------------------------
# 5. Run metadata tracking
# ---------------------------------------------------------------------------


class TestRunMetadata:
    def test_required_keys(self):
        meta = build_run_metadata(
            loto_types=["loto6", "miniloto"],
            preset="archcomp",
            seeds=[42, 123],
            evaluation_model_variants="legacy,deepsets,settransformer",
        )
        required_keys = [
            "schema_version",
            "run_id",
            "generated_at",
            "preset",
            "seeds",
            "loto_types",
            "evaluation_model_variants",
            "alpha",
            "source_summary_paths",
        ]
        for key in required_keys:
            assert key in meta, f"Missing key: {key}"

    def test_schema_version(self):
        meta = build_run_metadata(
            loto_types=["loto6"],
            preset="archcomp",
            seeds=[42],
            evaluation_model_variants="legacy",
        )
        assert meta["schema_version"] == CROSS_LOTO_REPORT_SCHEMA_VERSION

    def test_evaluation_model_variants_as_string(self):
        meta = build_run_metadata(
            loto_types=["loto6"],
            preset="archcomp",
            seeds=[42],
            evaluation_model_variants="legacy,deepsets",
        )
        assert isinstance(meta["evaluation_model_variants"], list)
        assert "legacy" in meta["evaluation_model_variants"]
        assert "deepsets" in meta["evaluation_model_variants"]

    def test_evaluation_model_variants_as_list(self):
        meta = build_run_metadata(
            loto_types=["loto6"],
            preset="archcomp",
            seeds=[42],
            evaluation_model_variants=["legacy", "multihot"],
        )
        assert isinstance(meta["evaluation_model_variants"], list)
        assert meta["evaluation_model_variants"] == ["legacy", "multihot"]

    def test_source_summary_paths_preserved(self):
        paths = {"loto6": "data/comparison_summary_loto6.json"}
        meta = build_run_metadata(
            loto_types=["loto6"],
            preset="archcomp",
            seeds=[42],
            evaluation_model_variants="legacy",
            source_summary_paths=paths,
        )
        assert meta["source_summary_paths"] == paths

    def test_run_id_can_be_set(self):
        meta = build_run_metadata(
            loto_types=["loto6"],
            preset="archcomp",
            seeds=[42],
            evaluation_model_variants="legacy",
            run_id="my_test_run",
        )
        assert meta["run_id"] == "my_test_run"

    def test_run_id_defaults_to_none(self):
        meta = build_run_metadata(
            loto_types=["loto6"],
            preset="archcomp",
            seeds=[42],
            evaluation_model_variants="legacy",
        )
        assert meta["run_id"] is None

    def test_loto_types_sorted(self):
        meta = build_run_metadata(
            loto_types=["miniloto", "loto6"],
            preset="archcomp",
            seeds=[42],
            evaluation_model_variants="legacy",
        )
        assert meta["loto_types"] == sorted(["miniloto", "loto6"])

    def test_generated_at_is_iso_format(self):
        from datetime import datetime
        meta = build_run_metadata(
            loto_types=["loto6"],
            preset="archcomp",
            seeds=[42],
            evaluation_model_variants="legacy",
        )
        # Should parse without error
        datetime.fromisoformat(meta["generated_at"])


# ---------------------------------------------------------------------------
# 6. Artifact save / load
# ---------------------------------------------------------------------------


class TestSaveReportArtifacts:
    def test_all_files_created(self, tmp_path):
        cross_summary, rec = _make_cross_loto_and_rec()
        saved = save_report_artifacts(cross_summary, rec, data_dir=tmp_path)
        assert "cross_loto_report.md" in saved
        assert "variant_metrics.csv" in saved
        assert "pairwise_comparisons.csv" in saved
        assert "recommendation_summary.csv" in saved
        for name, path in saved.items():
            assert Path(path).exists(), f"Expected file missing: {name}"

    def test_markdown_file_is_nonempty(self, tmp_path):
        cross_summary, rec = _make_cross_loto_and_rec()
        saved = save_report_artifacts(cross_summary, rec, data_dir=tmp_path)
        md_path = Path(saved["cross_loto_report.md"])
        assert md_path.stat().st_size > 100

    def test_csv_files_are_valid_csv(self, tmp_path):
        cross_summary, rec = _make_cross_loto_and_rec()
        saved = save_report_artifacts(cross_summary, rec, data_dir=tmp_path)
        for name in ["variant_metrics.csv", "pairwise_comparisons.csv", "recommendation_summary.csv"]:
            path = Path(saved[name])
            with open(path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) >= 1, f"Empty CSV: {name}"

    def test_run_metadata_in_markdown(self, tmp_path):
        cross_summary, rec = _make_cross_loto_and_rec()
        meta = build_run_metadata(
            loto_types=["loto6"],
            preset="archcomp",
            seeds=[42],
            evaluation_model_variants="legacy",
            run_id="save_test_run",
        )
        saved = save_report_artifacts(cross_summary, rec, data_dir=tmp_path, run_metadata=meta)
        md_content = Path(saved["cross_loto_report.md"]).read_text(encoding="utf-8")
        assert "save_test_run" in md_content

    def test_returns_path_dict(self, tmp_path):
        cross_summary, rec = _make_cross_loto_and_rec()
        saved = save_report_artifacts(cross_summary, rec, data_dir=tmp_path)
        assert isinstance(saved, dict)
        assert all(isinstance(v, str) for v in saved.values())


# ---------------------------------------------------------------------------
# 7. CLI smoke tests
# ---------------------------------------------------------------------------


class TestCliSmoke:
    def _run_cli(self, *args) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "run_cross_loto.py"), *args],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )

    def test_help_exits_zero(self):
        result = self._run_cli("--help")
        assert result.returncode == 0

    def test_help_mentions_report_only(self):
        result = self._run_cli("--help")
        assert "--report_only" in result.stdout

    def test_help_mentions_skip_training(self):
        result = self._run_cli("--help")
        assert "--skip_training" in result.stdout

    def test_report_only_fails_gracefully_without_artifacts(self, tmp_path):
        result = self._run_cli("--report_only", "--data_dir", str(tmp_path))
        # Should exit with error since no cross_loto_summary.json exists
        assert result.returncode != 0

    def test_report_only_succeeds_with_existing_artifacts(self, tmp_path):
        """Write minimal cross_loto_summary.json and recommendation.json, then --report_only."""
        cross_summary, rec = _make_cross_loto_and_rec()
        import json as _json
        (tmp_path / "cross_loto_summary.json").write_text(
            _json.dumps(cross_summary, indent=2), encoding="utf-8"
        )
        (tmp_path / "recommendation.json").write_text(
            _json.dumps(rec, indent=2), encoding="utf-8"
        )
        result = self._run_cli("--report_only", "--data_dir", str(tmp_path))
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert (tmp_path / "cross_loto_report.md").exists()
        assert (tmp_path / "variant_metrics.csv").exists()

    def test_invalid_loto_type_exits_nonzero(self):
        result = self._run_cli("--loto_types", "invalid_type_xyz")
        assert result.returncode != 0

    def test_invalid_seed_exits_nonzero(self):
        result = self._run_cli("--seeds", "abc", "--skip_training")
        assert result.returncode != 0
