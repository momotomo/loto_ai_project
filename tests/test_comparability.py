"""tests/test_comparability.py

Tests for the benchmark registry and comparability checker:
  - benchmark registry resolution
  - campaign metadata required keys (benchmark_name, evaluation_model_variants, etc.)
  - comparability checker required keys
  - comparable / not comparable cases
  - governance / diff report comparability integration
  - run_campaign comparability artifact smoke
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from benchmark_registry import (
    BENCHMARK_DEFINITIONS,
    BENCHMARK_REGISTRY_SCHEMA_VERSION,
    VALID_BENCHMARK_NAMES,
    benchmarks_are_compatible,
    get_benchmark,
    list_benchmarks,
    resolve_benchmark_for_profile,
    validate_campaign_against_benchmark,
)
from comparability_checker import (
    COMPARABILITY_SCHEMA_VERSION,
    build_comparability_report_md,
    check_history_comparability,
    check_pair_comparability,
    save_comparability_artifacts,
)
from campaign_manager import (
    build_campaign_entry,
    build_diff_report,
)
from governance_layer import (
    build_promotion_gate,
    build_trend_summary,
    build_regression_alert,
    build_governance_report,
    save_governance_artifacts,
)
from campaign_manager import compute_recommendation_stability


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_entry(
    name: str = "c1",
    profile: str = "archcomp",
    loto_types: list[str] | None = None,
    seeds: list[int] | None = None,
    action: str = "hold",
    challenger: str | None = "multihot",
    evaluation_model_variants: str = "legacy,multihot,deepsets,settransformer",
    evaluation_calibration_methods: str = "none,temperature,isotonic",
    data_fingerprints: dict | None = None,
    ranking: list[dict] | None = None,
    pairwise: dict | None = None,
    keep: bool = True,
    blockers: int = 2,
) -> dict:
    if loto_types is None:
        loto_types = ["loto6", "loto7", "miniloto"]
    if seeds is None:
        seeds = [42, 123, 456]
    if ranking is None:
        ranking = [
            {"rank": 1, "variant": "multihot", "logloss_mean": 0.310},
            {"rank": 2, "variant": "legacy", "logloss_mean": 0.315},
            {"rank": 3, "variant": "deepsets", "logloss_mean": 0.320},
            {"rank": 4, "variant": "settransformer", "logloss_mean": 0.325},
        ]
    if pairwise is None:
        pairwise = {
            "settransformer_vs_deepsets": {"run_count": 9, "both_pass_count": 1, "both_pass_rate": 0.111},
            "deepsets_vs_legacy": {"run_count": 9, "both_pass_count": 0, "both_pass_rate": 0.0},
        }
    return {
        "campaign_name": name,
        "profile_name": profile,
        "benchmark_name": resolve_benchmark_for_profile(profile),
        "generated_at": "2026-03-21T00:00:00+00:00",
        "loto_types": loto_types,
        "seeds": seeds,
        "preset": "archcomp",
        "recommended_next_action": action,
        "recommended_challenger": challenger,
        "keep_production_as_is": keep,
        "whether_to_try_pma_or_isab_next": False,
        "best_variant_by_logloss": challenger,
        "consistent_promote_variants": [] if action != "consider_promotion" else [challenger],
        "blockers_count": blockers,
        "variant_ranking_summary": ranking,
        "key_pairwise_signals": pairwise,
        "evaluation_model_variants": evaluation_model_variants,
        "evaluation_calibration_methods": evaluation_calibration_methods,
        "data_fingerprints": data_fingerprints or {},
    }


# ---------------------------------------------------------------------------
# 1. Benchmark registry tests
# ---------------------------------------------------------------------------


class TestBenchmarkRegistry:
    def test_valid_benchmark_names(self):
        assert set(VALID_BENCHMARK_NAMES) == {"archcomp", "archcomp_lite", "archcomp_full"}

    def test_schema_version_is_int(self):
        assert isinstance(BENCHMARK_REGISTRY_SCHEMA_VERSION, int)
        assert BENCHMARK_REGISTRY_SCHEMA_VERSION >= 1

    def test_get_benchmark_returns_copy(self):
        b = get_benchmark("archcomp")
        assert isinstance(b, dict)
        assert b["benchmark_name"] == "archcomp"
        # Modifying the copy must not affect registry
        b["benchmark_name"] = "modified"
        assert BENCHMARK_DEFINITIONS["archcomp"]["benchmark_name"] == "archcomp"

    def test_get_benchmark_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown benchmark"):
            get_benchmark("nonexistent")

    def test_benchmark_required_keys(self):
        required = {
            "benchmark_name", "allowed_profiles", "compatible_benchmarks",
            "expected_loto_types", "expected_variants", "min_seed_count",
            "allowed_seed_sets", "expected_calibration_methods",
            "evaluation_window_policy", "fold_policy",
            "data_freshness_policy", "description",
        }
        for name in VALID_BENCHMARK_NAMES:
            b = get_benchmark(name)
            missing = required - set(b.keys())
            assert not missing, f"{name} missing keys: {missing}"

    def test_resolve_benchmark_for_profile(self):
        assert resolve_benchmark_for_profile("archcomp") == "archcomp"
        assert resolve_benchmark_for_profile("archcomp_lite") == "archcomp_lite"
        assert resolve_benchmark_for_profile("archcomp_full") == "archcomp_full"

    def test_resolve_benchmark_unknown_profile_returns_name(self):
        result = resolve_benchmark_for_profile("my_custom_profile")
        assert result == "my_custom_profile"

    def test_benchmarks_are_compatible_same(self):
        assert benchmarks_are_compatible("archcomp", "archcomp")
        assert benchmarks_are_compatible("archcomp_lite", "archcomp_lite")

    def test_benchmarks_are_compatible_archcomp_and_full(self):
        assert benchmarks_are_compatible("archcomp", "archcomp_full")
        assert benchmarks_are_compatible("archcomp_full", "archcomp")

    def test_benchmarks_not_compatible_lite_and_archcomp(self):
        assert not benchmarks_are_compatible("archcomp_lite", "archcomp")
        assert not benchmarks_are_compatible("archcomp", "archcomp_lite")

    def test_validate_campaign_against_benchmark_pass(self):
        entry = _make_entry(
            loto_types=["loto6", "loto7", "miniloto"],
            seeds=[42, 123, 456],
            evaluation_model_variants="legacy,multihot,deepsets,settransformer",
            evaluation_calibration_methods="none,temperature,isotonic",
        )
        defn = get_benchmark("archcomp")
        failures = validate_campaign_against_benchmark(entry, defn)
        assert failures == []

    def test_validate_campaign_against_benchmark_fail_seeds(self):
        entry = _make_entry(seeds=[42])
        defn = get_benchmark("archcomp")
        failures = validate_campaign_against_benchmark(entry, defn)
        assert any("seed_count" in f for f in failures)

    def test_validate_campaign_against_benchmark_fail_loto(self):
        entry = _make_entry(loto_types=["loto6"])
        defn = get_benchmark("archcomp")
        failures = validate_campaign_against_benchmark(entry, defn)
        assert any("loto_types" in f for f in failures)

    def test_list_benchmarks_runs(self, capsys):
        list_benchmarks()
        out = capsys.readouterr().out
        assert "archcomp" in out
        assert "archcomp_lite" in out


# ---------------------------------------------------------------------------
# 2. Campaign metadata required keys
# ---------------------------------------------------------------------------


class TestCampaignMetadataKeys:
    def test_build_campaign_entry_has_benchmark_name(self):
        from cross_loto_summary import build_cross_loto_summary, build_recommendation
        per_loto = {
            "loto6": {
                "schema_version": 1, "loto_type": "loto6", "preset": "archcomp",
                "seeds": [42], "run_count": 1, "alpha": 0.05,
                "variants": {"legacy": {
                    "run_count": 1,
                    "logloss": {"mean": 0.3, "std": 0.0, "values": [0.3]},
                    "brier": {"mean": 0.04, "std": 0.0, "values": [0.04]},
                    "ece": {"mean": 0.01, "std": 0.0, "values": [0.01]},
                    "calibration_recommendations": {"none": 1},
                    "promote_count": 0, "hold_count": 1,
                }},
                "pairwise_comparisons": {},
            }
        }
        cs = build_cross_loto_summary(per_loto, ["loto6"], "archcomp", [42])
        rec = build_recommendation(cs)
        entry = build_campaign_entry(
            "test_campaign", "archcomp", cs, rec,
            evaluation_model_variants="legacy,multihot,deepsets,settransformer",
            evaluation_calibration_methods="none,temperature,isotonic",
        )
        assert "benchmark_name" in entry
        assert entry["benchmark_name"] == "archcomp"
        assert "evaluation_model_variants" in entry
        assert "evaluation_calibration_methods" in entry
        assert "data_fingerprints" in entry

    def test_build_campaign_entry_benchmark_for_lite(self):
        from cross_loto_summary import build_cross_loto_summary, build_recommendation
        per_loto = {
            "loto6": {
                "schema_version": 1, "loto_type": "loto6", "preset": "fast",
                "seeds": [42], "run_count": 1, "alpha": 0.05,
                "variants": {"legacy": {
                    "run_count": 1,
                    "logloss": {"mean": 0.3, "std": 0.0, "values": [0.3]},
                    "brier": {"mean": 0.04, "std": 0.0, "values": [0.04]},
                    "ece": {"mean": 0.01, "std": 0.0, "values": [0.01]},
                    "calibration_recommendations": {"none": 1},
                    "promote_count": 0, "hold_count": 1,
                }},
                "pairwise_comparisons": {},
            }
        }
        cs = build_cross_loto_summary(per_loto, ["loto6"], "fast", [42])
        rec = build_recommendation(cs)
        entry = build_campaign_entry("test", "archcomp_lite", cs, rec)
        assert entry["benchmark_name"] == "archcomp_lite"


# ---------------------------------------------------------------------------
# 3. Comparability checker required keys
# ---------------------------------------------------------------------------


class TestComparabilityCheckerKeys:
    def test_pair_check_required_keys(self):
        e1 = _make_entry("c1")
        e2 = _make_entry("c2")
        result = check_pair_comparability(e1, e2)
        required = {
            "schema_version", "campaign_a", "campaign_b",
            "benchmark_a", "benchmark_b", "comparable",
            "severity", "failed_checks", "warnings",
            "rationale", "suggested_action",
        }
        assert required <= set(result.keys())

    def test_history_check_required_keys(self):
        history = [_make_entry("c1"), _make_entry("c2")]
        result = check_history_comparability(history)
        required = {
            "schema_version", "generated_at", "total_campaigns",
            "comparable_pairs", "warning_pairs", "incomparable_pairs",
            "overall_comparable", "overall_severity", "pairs", "summary",
        }
        assert required <= set(result.keys())

    def test_schema_version(self):
        e1 = _make_entry("c1")
        e2 = _make_entry("c2")
        result = check_pair_comparability(e1, e2)
        assert result["schema_version"] == COMPARABILITY_SCHEMA_VERSION

    def test_severity_values(self):
        e1 = _make_entry("c1")
        e2 = _make_entry("c2")
        result = check_pair_comparability(e1, e2)
        assert result["severity"] in ("ok", "warning", "error")

    def test_empty_history(self):
        result = check_history_comparability([])
        assert result["total_campaigns"] == 0
        assert result["overall_comparable"] is True
        assert result["pairs"] == []

    def test_single_campaign_history(self):
        result = check_history_comparability([_make_entry("c1")])
        assert result["total_campaigns"] == 1
        assert result["overall_comparable"] is True
        assert result["pairs"] == []


# ---------------------------------------------------------------------------
# 4. Comparable / not comparable cases
# ---------------------------------------------------------------------------


class TestComparabilityLogic:
    def test_identical_entries_fully_comparable(self):
        e1 = _make_entry("c1")
        e2 = _make_entry("c2")
        result = check_pair_comparability(e1, e2)
        assert result["comparable"] is True
        # Data fingerprints not available → warning (not error)
        assert result["severity"] in ("ok", "warning")
        assert result["failed_checks"] == []

    def test_different_loto_types_not_comparable(self):
        e1 = _make_entry("c1", loto_types=["loto6", "loto7", "miniloto"])
        e2 = _make_entry("c2", loto_types=["loto6"])
        result = check_pair_comparability(e1, e2)
        assert result["comparable"] is False
        assert result["severity"] == "error"
        assert any("loto_type" in f for f in result["failed_checks"])

    def test_different_variants_not_comparable(self):
        e1 = _make_entry("c1", evaluation_model_variants="legacy,multihot")
        e2 = _make_entry("c2", evaluation_model_variants="legacy,multihot,deepsets,settransformer")
        result = check_pair_comparability(e1, e2)
        assert result["comparable"] is False
        assert any("variant" in f for f in result["failed_checks"])

    def test_different_calibration_not_comparable(self):
        e1 = _make_entry("c1", evaluation_calibration_methods="none,temperature,isotonic")
        e2 = _make_entry("c2", evaluation_calibration_methods="none")
        result = check_pair_comparability(e1, e2)
        assert result["comparable"] is False
        assert any("calibration" in f for f in result["failed_checks"])

    def test_incompatible_benchmark_not_comparable(self):
        e1 = _make_entry("c1", profile="archcomp")
        e2 = _make_entry("c2", profile="archcomp_lite")
        # Force different loto_types to match the lite benchmark
        e2["loto_types"] = ["loto6"]
        e2["benchmark_name"] = "archcomp_lite"
        result = check_pair_comparability(e1, e2)
        # Should be not comparable due to benchmark mismatch and/or loto mismatch
        assert result["comparable"] is False

    def test_compatible_benchmark_warning(self):
        e1 = _make_entry("c1", profile="archcomp", seeds=[42, 123, 456])
        e2 = _make_entry("c2", profile="archcomp_full", seeds=[42, 123, 456, 789, 999])
        e2["benchmark_name"] = "archcomp_full"
        result = check_pair_comparability(e1, e2)
        # Compatible benchmarks → comparable (possibly with warnings)
        assert result["comparable"] is True
        # Should have a warning about different benchmarks
        all_soft = result["warnings"]
        assert any("compatible" in w for w in all_soft) or result["severity"] == "warning"

    def test_seed_count_mismatch_is_warning_not_error(self):
        e1 = _make_entry("c1", seeds=[42, 123, 456])
        e2 = _make_entry("c2", seeds=[42, 123, 456, 789, 999])
        e2["benchmark_name"] = "archcomp_full"
        e2["profile_name"] = "archcomp_full"
        result = check_pair_comparability(e1, e2)
        # Seed count mismatch = warning, not hard failure
        assert result["comparable"] is True  # compatible benchmarks
        assert any("seed" in w for w in result.get("warnings", []))

    def test_data_fingerprint_mismatch_is_warning(self):
        e1 = _make_entry("c1", data_fingerprints={"loto6": "hash_abc"})
        e2 = _make_entry("c2", data_fingerprints={"loto6": "hash_xyz"})
        result = check_pair_comparability(e1, e2)
        assert result["comparable"] is True  # fingerprint = soft check
        assert any("fingerprint" in w for w in result["warnings"])

    def test_history_all_comparable(self):
        history = [_make_entry(f"c{i}") for i in range(3)]
        result = check_history_comparability(history)
        assert result["overall_comparable"] is True
        assert result["incomparable_pairs"] == 0
        assert result["comparable_pairs"] == 2

    def test_history_with_incomparable_pair(self):
        e1 = _make_entry("c1", loto_types=["loto6", "loto7", "miniloto"])
        e2 = _make_entry("c2", loto_types=["loto6"])
        result = check_history_comparability([e1, e2])
        assert result["overall_comparable"] is False
        assert result["incomparable_pairs"] == 1
        assert result["overall_severity"] == "error"


# ---------------------------------------------------------------------------
# 5. Governance / diff report comparability integration
# ---------------------------------------------------------------------------


class TestGovernanceComparabilityIntegration:
    def _two_entry_history(self):
        return [_make_entry("c1"), _make_entry("c2")]

    def test_trend_summary_has_comparability_fields(self):
        history = self._two_entry_history()
        trend = build_trend_summary(history)
        assert "comparability_overall" in trend
        assert "comparability_severity" in trend
        assert "comparability_note" in trend

    def test_regression_alert_has_comparability_caution(self):
        history = self._two_entry_history()
        alert = build_regression_alert(history)
        assert "comparability_caution" in alert
        assert "comparability_note" in alert

    def test_promotion_gate_blocks_on_incomparable(self):
        # e1 and e2 with different loto_types should cause comparability error
        e1 = _make_entry("c1", loto_types=["loto6", "loto7", "miniloto"])
        e2 = _make_entry("c2", loto_types=["loto6"])
        history = [e1, e2]
        stability = compute_recommendation_stability(history)
        gate = build_promotion_gate(history, stability)
        # comparability error should be a blocker
        assert "comparability_ok" not in " ".join(gate.get("conditions_passed", []))
        assert any("comparability" in b.lower() for b in gate.get("blockers", []))

    def test_promotion_gate_has_comparability_fields(self):
        history = self._two_entry_history()
        stability = compute_recommendation_stability(history)
        gate = build_promotion_gate(history, stability)
        assert "comparability_ok" in gate
        assert "comparability_severity" in gate
        assert "comparability_note" in gate

    def test_promotion_gate_comparability_ok_for_comparable_history(self):
        history = self._two_entry_history()
        stability = compute_recommendation_stability(history)
        gate = build_promotion_gate(history, stability)
        # For comparable history, comparability_ok should be True
        assert gate["comparability_ok"] is True

    def test_governance_report_has_comparability_section(self):
        history = self._two_entry_history()
        stability = compute_recommendation_stability(history)
        trend = build_trend_summary(history)
        alert = build_regression_alert(history)
        gate = build_promotion_gate(history, stability)
        from comparability_checker import check_history_comparability
        comp = check_history_comparability(history)
        md = build_governance_report(trend, alert, gate, stability, comparability_result=comp)
        assert "## Comparability" in md
        assert "# Governance Report" in md

    def test_diff_report_has_comparability_section(self):
        e1 = _make_entry("c1")
        e2 = _make_entry("c2")
        stability = compute_recommendation_stability([e1, e2])
        md = build_diff_report(e1, e2, stability=stability)
        assert "## Comparability" in md
        assert "COMPARABLE" in md or "comparable" in md.lower()

    def test_diff_report_warns_on_incomparable(self):
        e1 = _make_entry("c1", loto_types=["loto6", "loto7", "miniloto"])
        e2 = _make_entry("c2", loto_types=["loto6"])
        md = build_diff_report(e1, e2)
        assert "NOT COMPARABLE" in md or "not comparable" in md.lower()

    def test_save_governance_artifacts_includes_comparability(self):
        history = self._two_entry_history()
        stability = compute_recommendation_stability(history)
        with tempfile.TemporaryDirectory() as d:
            paths = save_governance_artifacts(history, stability, data_dir=d)
        assert "comparability_report.json" in paths
        assert "comparability_report.md" in paths


# ---------------------------------------------------------------------------
# 6. Save comparability artifacts smoke
# ---------------------------------------------------------------------------


class TestSaveComparabilityArtifacts:
    def test_saves_json_and_md(self):
        history = [_make_entry("c1"), _make_entry("c2")]
        with tempfile.TemporaryDirectory() as d:
            paths = save_comparability_artifacts(history, data_dir=d)
            assert "comparability_report.json" in paths
            assert "comparability_report.md" in paths
            assert Path(paths["comparability_report.json"]).exists()
            assert Path(paths["comparability_report.md"]).exists()

    def test_json_is_valid_and_has_required_keys(self):
        history = [_make_entry("c1"), _make_entry("c2")]
        with tempfile.TemporaryDirectory() as d:
            paths = save_comparability_artifacts(history, data_dir=d)
            payload = json.loads(Path(paths["comparability_report.json"]).read_text())
        assert "schema_version" in payload
        assert "overall_comparable" in payload
        assert "pairs" in payload

    def test_md_has_required_sections(self):
        history = [_make_entry("c1"), _make_entry("c2")]
        with tempfile.TemporaryDirectory() as d:
            paths = save_comparability_artifacts(history, data_dir=d)
            md = Path(paths["comparability_report.md"]).read_text()
        assert "# Comparability Report" in md
        assert "## Overall Status" in md

    def test_empty_history_creates_files(self):
        with tempfile.TemporaryDirectory() as d:
            paths = save_comparability_artifacts([], data_dir=d)
            assert "comparability_report.json" in paths
            assert Path(paths["comparability_report.json"]).exists()

    def test_md_report_build(self):
        result = check_history_comparability([_make_entry("c1"), _make_entry("c2")])
        md = build_comparability_report_md(result)
        assert "# Comparability Report" in md
        assert "## Overall Status" in md


# ---------------------------------------------------------------------------
# 7. Module imports and CLI smoke
# ---------------------------------------------------------------------------


class TestModuleImports:
    def test_benchmark_registry_imports(self):
        import benchmark_registry
        assert hasattr(benchmark_registry, "BENCHMARK_DEFINITIONS")
        assert hasattr(benchmark_registry, "BENCHMARK_REGISTRY_SCHEMA_VERSION")
        assert hasattr(benchmark_registry, "resolve_benchmark_for_profile")
        assert hasattr(benchmark_registry, "get_benchmark")
        assert hasattr(benchmark_registry, "benchmarks_are_compatible")

    def test_comparability_checker_imports(self):
        import comparability_checker
        assert hasattr(comparability_checker, "check_pair_comparability")
        assert hasattr(comparability_checker, "check_history_comparability")
        assert hasattr(comparability_checker, "save_comparability_artifacts")
        assert hasattr(comparability_checker, "COMPARABILITY_SCHEMA_VERSION")

    def test_run_campaign_imports_benchmark_registry(self):
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-c",
             "from benchmark_registry import resolve_benchmark_for_profile; "
             "from comparability_checker import check_history_comparability; "
             "print('OK')"],
            capture_output=True, text=True, cwd=str(Path(__file__).parents[1]),
        )
        assert result.returncode == 0, result.stderr
        assert "OK" in result.stdout

    def test_campaign_manager_has_benchmark_name(self):
        from cross_loto_summary import build_cross_loto_summary, build_recommendation
        per_loto = {
            "loto6": {
                "schema_version": 1, "loto_type": "loto6", "preset": "archcomp",
                "seeds": [42], "run_count": 1, "alpha": 0.05,
                "variants": {"legacy": {
                    "run_count": 1,
                    "logloss": {"mean": 0.3, "std": 0.0, "values": [0.3]},
                    "brier": {"mean": 0.04, "std": 0.0, "values": [0.04]},
                    "ece": {"mean": 0.01, "std": 0.0, "values": [0.01]},
                    "calibration_recommendations": {"none": 1},
                    "promote_count": 0, "hold_count": 1,
                }},
                "pairwise_comparisons": {},
            }
        }
        cs = build_cross_loto_summary(per_loto, ["loto6"], "archcomp", [42])
        rec = build_recommendation(cs)
        entry = build_campaign_entry("c", "archcomp", cs, rec)
        assert "benchmark_name" in entry
        assert entry["benchmark_name"] == "archcomp"
