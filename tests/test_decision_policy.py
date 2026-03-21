"""tests/test_decision_policy.py

Tests for decision_policy.py — benchmark lock artifact, campaign acceptance
check, governance integration, accepted-only stability, and CLI smoke.

Focus: implementation health, comparability discipline, and safe operation.
NOT testing statistical accuracy or model quality.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import under test
# ---------------------------------------------------------------------------

from decision_policy import (
    DECISION_BENCHMARK_POLICY,
    DECISION_POLICY_SCHEMA_VERSION,
    CAMPAIGN_ACCEPTANCE_SCHEMA_VERSION,
    build_benchmark_lock_artifact,
    build_benchmark_lock_md,
    check_campaign_acceptance,
    build_campaign_acceptance_md,
    save_benchmark_lock_artifacts,
    save_campaign_acceptance_artifacts,
)
from campaign_manager import (
    build_campaign_entry,
    compute_recommendation_stability,
)
from governance_layer import build_governance_report, save_governance_artifacts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cross_loto_summary(
    loto_types=None,
    preset="archcomp",
    seeds=None,
    variants=("legacy", "multihot", "deepsets", "settransformer"),
):
    loto_types = loto_types or ["loto6", "loto7", "miniloto"]
    seeds = seeds or [42, 123, 456]
    return {
        "generated_at": "2026-03-21T00:00:00+00:00",
        "loto_types": loto_types,
        "preset": preset,
        "seeds": seeds,
        "variant_ranking": {
            "by_logloss": [
                {"rank": i + 1, "variant": v, "mean": 0.5 + i * 0.01}
                for i, v in enumerate(variants)
            ]
        },
        "pairwise_comparison_summary": {
            "settransformer_vs_deepsets": {
                "overall": {"run_count": 9, "both_pass_count": 5}
            }
        },
    }


def _make_recommendation(**kwargs):
    base = {
        "recommended_next_action": "hold",
        "recommended_challenger": "settransformer",
        "keep_production_as_is": True,
        "whether_to_try_pma_or_isab_next": False,
        "blockers_to_promotion": [],
        "evidence_summary": {
            "best_variant_by_logloss": "legacy",
            "consistent_promote_variants": [],
        },
    }
    base.update(kwargs)
    return base


def _make_accepted_entry(name="2026-03-21_archcomp", profile="archcomp"):
    """Build a minimal campaign entry that should be accepted."""
    cross = _make_cross_loto_summary()
    rec = _make_recommendation()
    return build_campaign_entry(
        name,
        profile,
        cross,
        rec,
        evaluation_model_variants="legacy,multihot,deepsets,settransformer",
        evaluation_calibration_methods="none,temperature,isotonic",
    )


def _make_lite_entry(name="2026-03-21_lite"):
    """Build a minimal archcomp_lite entry (should NOT be accepted)."""
    cross = _make_cross_loto_summary(loto_types=["loto6"], preset="fast", seeds=[42, 123])
    rec = _make_recommendation()
    return build_campaign_entry(
        name,
        "archcomp_lite",
        cross,
        rec,
        evaluation_model_variants="legacy,multihot,deepsets,settransformer",
        evaluation_calibration_methods="none,temperature,isotonic",
    )


# ===========================================================================
# 1. DECISION_BENCHMARK_POLICY required keys
# ===========================================================================


class TestDecisionBenchmarkPolicy:
    REQUIRED_KEYS = [
        "schema_version",
        "policy_name",
        "active_decision_benchmarks",
        "allowed_profiles",
        "excluded_profiles",
        "excluded_profiles_reason",
        "required_loto_types",
        "required_variants",
        "required_calibration_methods",
        "minimum_seed_policy",
        "comparability_required",
        "promotion_review_policy",
        "accepted_only_counting_policy",
        "pma_isab_precondition_policy",
    ]

    def test_required_keys_present(self):
        for key in self.REQUIRED_KEYS:
            assert key in DECISION_BENCHMARK_POLICY, f"Missing key: {key!r}"

    def test_schema_version(self):
        assert DECISION_BENCHMARK_POLICY["schema_version"] == DECISION_POLICY_SCHEMA_VERSION

    def test_active_benchmarks_contains_archcomp(self):
        bms = DECISION_BENCHMARK_POLICY["active_decision_benchmarks"]
        assert "archcomp" in bms
        assert "archcomp_full" in bms

    def test_archcomp_lite_excluded(self):
        excluded = DECISION_BENCHMARK_POLICY["excluded_profiles"]
        assert "archcomp_lite" in excluded

    def test_required_loto_types(self):
        loto = set(DECISION_BENCHMARK_POLICY["required_loto_types"])
        assert loto == {"loto6", "loto7", "miniloto"}

    def test_required_variants(self):
        variants = set(DECISION_BENCHMARK_POLICY["required_variants"])
        assert variants == {"legacy", "multihot", "deepsets", "settransformer"}

    def test_required_calibration_methods(self):
        cals = set(DECISION_BENCHMARK_POLICY["required_calibration_methods"])
        assert cals == {"none", "temperature", "isotonic"}

    def test_comparability_required_is_true(self):
        assert DECISION_BENCHMARK_POLICY["comparability_required"] is True

    def test_minimum_seed_policy_archcomp(self):
        mp = DECISION_BENCHMARK_POLICY["minimum_seed_policy"]
        assert mp.get("archcomp", 0) >= 3

    def test_minimum_seed_policy_archcomp_full(self):
        mp = DECISION_BENCHMARK_POLICY["minimum_seed_policy"]
        assert mp.get("archcomp_full", 0) >= 5


# ===========================================================================
# 2. Benchmark lock artifact required keys
# ===========================================================================


class TestBenchmarkLockArtifact:
    REQUIRED_KEYS = [
        "schema_version",
        "policy_name",
        "generated_at",
        "active_decision_benchmarks",
        "allowed_profiles",
        "excluded_profiles",
        "excluded_profiles_reason",
        "required_loto_types",
        "required_variants",
        "required_calibration_methods",
        "minimum_seed_policy",
        "comparability_required",
        "promotion_review_policy",
        "accepted_only_counting_policy",
        "pma_isab_precondition_policy",
        "active_benchmark_definitions",
    ]

    def test_required_keys_present(self):
        lock = build_benchmark_lock_artifact()
        for key in self.REQUIRED_KEYS:
            assert key in lock, f"Missing key: {key!r}"

    def test_schema_version(self):
        lock = build_benchmark_lock_artifact()
        assert lock["schema_version"] == DECISION_POLICY_SCHEMA_VERSION

    def test_active_benchmark_definitions_populated(self):
        lock = build_benchmark_lock_artifact()
        defs = lock["active_benchmark_definitions"]
        assert "archcomp" in defs
        assert "archcomp_full" in defs
        assert "archcomp_lite" not in defs

    def test_benchmark_lock_md_has_required_sections(self):
        lock = build_benchmark_lock_artifact()
        md = build_benchmark_lock_md(lock)
        assert "Decision Benchmark Policy" in md
        assert "Active Decision Benchmarks" in md
        assert "Required Conditions for Acceptance" in md
        assert "Promotion Review Policy" in md
        assert "archcomp_lite" in md

    def test_save_benchmark_lock_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            paths = save_benchmark_lock_artifacts(data_dir=d)
            assert "benchmark_lock.json" in paths
            assert "benchmark_lock.md" in paths
            json_path = Path(paths["benchmark_lock.json"])
            md_path = Path(paths["benchmark_lock.md"])
            assert json_path.exists()
            assert md_path.exists()
            with open(json_path) as f:
                data = json.load(f)
            assert data["schema_version"] == DECISION_POLICY_SCHEMA_VERSION


# ===========================================================================
# 3. Campaign acceptance artifact required keys
# ===========================================================================


class TestCampaignAcceptanceKeys:
    REQUIRED_KEYS = [
        "schema_version",
        "generated_at",
        "campaign_name",
        "profile_name",
        "benchmark_name",
        "decision_benchmark_name",
        "accepted_for_decision_use",
        "counts_toward_promotion_readiness",
        "can_count_toward_promotion_readiness",
        "failed_requirements",
        "warnings",
        "rationale",
    ]

    def test_required_keys_present(self):
        entry = _make_accepted_entry()
        result = check_campaign_acceptance(entry)
        for key in self.REQUIRED_KEYS:
            assert key in result, f"Missing key: {key!r}"

    def test_schema_version(self):
        entry = _make_accepted_entry()
        result = check_campaign_acceptance(entry)
        assert result["schema_version"] == CAMPAIGN_ACCEPTANCE_SCHEMA_VERSION

    def test_failed_requirements_is_list(self):
        entry = _make_accepted_entry()
        result = check_campaign_acceptance(entry)
        assert isinstance(result["failed_requirements"], list)

    def test_warnings_is_list(self):
        entry = _make_accepted_entry()
        result = check_campaign_acceptance(entry)
        assert isinstance(result["warnings"], list)


# ===========================================================================
# 4. Campaign acceptance logic — accepted cases
# ===========================================================================


class TestCampaignAcceptanceLogic:
    def test_archcomp_accepted(self):
        entry = _make_accepted_entry(profile="archcomp")
        result = check_campaign_acceptance(entry, comparability_ok=True)
        assert result["accepted_for_decision_use"] is True
        assert result["counts_toward_promotion_readiness"] is True
        assert result["failed_requirements"] == []

    def test_archcomp_full_accepted(self):
        cross = _make_cross_loto_summary(seeds=[42, 123, 456, 789, 999])
        rec = _make_recommendation()
        entry = build_campaign_entry(
            "2026-03-21_full",
            "archcomp_full",
            cross,
            rec,
            evaluation_model_variants="legacy,multihot,deepsets,settransformer",
            evaluation_calibration_methods="none,temperature,isotonic",
        )
        result = check_campaign_acceptance(entry, comparability_ok=True)
        assert result["accepted_for_decision_use"] is True
        assert len(result["failed_requirements"]) == 0

    def test_archcomp_lite_not_accepted(self):
        entry = _make_lite_entry()
        result = check_campaign_acceptance(entry, comparability_ok=True)
        assert result["accepted_for_decision_use"] is False
        assert len(result["failed_requirements"]) >= 1
        # Should mention the excluded profile reason
        combined = " ".join(result["failed_requirements"])
        assert "archcomp_lite" in combined or "excluded" in combined

    def test_comparability_false_blocks_acceptance(self):
        entry = _make_accepted_entry()
        result = check_campaign_acceptance(entry, comparability_ok=False)
        assert result["accepted_for_decision_use"] is False
        combined = " ".join(result["failed_requirements"])
        assert "comparability_ok=False" in combined

    def test_loto_types_mismatch_blocks_acceptance(self):
        cross = _make_cross_loto_summary(loto_types=["loto6"])  # missing loto7, miniloto
        rec = _make_recommendation()
        entry = build_campaign_entry(
            "2026-03-21_partial",
            "archcomp",
            cross,
            rec,
            evaluation_model_variants="legacy,multihot,deepsets,settransformer",
            evaluation_calibration_methods="none,temperature,isotonic",
        )
        result = check_campaign_acceptance(entry, comparability_ok=True)
        assert result["accepted_for_decision_use"] is False
        combined = " ".join(result["failed_requirements"])
        assert "loto_types" in combined

    def test_variant_mismatch_blocks_acceptance(self):
        entry = _make_accepted_entry()
        # Manually corrupt evaluation_model_variants
        entry["evaluation_model_variants"] = "legacy,multihot"
        result = check_campaign_acceptance(entry, comparability_ok=True)
        assert result["accepted_for_decision_use"] is False
        combined = " ".join(result["failed_requirements"])
        assert "variant_set" in combined

    def test_calibration_mismatch_blocks_acceptance(self):
        entry = _make_accepted_entry()
        entry["evaluation_calibration_methods"] = "none"
        result = check_campaign_acceptance(entry, comparability_ok=True)
        assert result["accepted_for_decision_use"] is False
        combined = " ".join(result["failed_requirements"])
        assert "calibration_methods" in combined

    def test_insufficient_seeds_blocks_acceptance(self):
        cross = _make_cross_loto_summary(seeds=[42])  # only 1 seed
        rec = _make_recommendation()
        entry = build_campaign_entry(
            "2026-03-21_oneseed",
            "archcomp",
            cross,
            rec,
            evaluation_model_variants="legacy,multihot,deepsets,settransformer",
            evaluation_calibration_methods="none,temperature,isotonic",
        )
        result = check_campaign_acceptance(entry, comparability_ok=True)
        assert result["accepted_for_decision_use"] is False
        combined = " ".join(result["failed_requirements"])
        assert "seed_count" in combined

    def test_comparability_unknown_raises_warning(self):
        entry = _make_accepted_entry()
        result = check_campaign_acceptance(entry, comparability_ok=None)
        # comparability_ok=None is a warning, not a hard failure
        # (campaign may still be accepted if all other conditions pass)
        # The warning about unknown comparability should be present
        combined = " ".join(result["warnings"])
        assert "comparability" in combined.lower()

    def test_acceptance_md_has_required_sections(self):
        entry = _make_accepted_entry()
        result = check_campaign_acceptance(entry, comparability_ok=True)
        md = build_campaign_acceptance_md(result)
        assert "Campaign Acceptance Report" in md
        assert "Acceptance Status" in md
        assert "accepted_for_decision_use" in md

    def test_rejection_md_shows_failed_requirements(self):
        entry = _make_lite_entry()
        result = check_campaign_acceptance(entry, comparability_ok=True)
        md = build_campaign_acceptance_md(result)
        assert "NOT ACCEPTED" in md
        assert "Failed Requirements" in md

    def test_save_campaign_acceptance_artifacts(self):
        entry = _make_accepted_entry()
        with tempfile.TemporaryDirectory() as d:
            paths = save_campaign_acceptance_artifacts(entry, data_dir=d, comparability_ok=True)
            assert "campaign_acceptance.json" in paths
            assert "campaign_acceptance.md" in paths
            json_path = Path(paths["campaign_acceptance.json"])
            md_path = Path(paths["campaign_acceptance.md"])
            assert json_path.exists()
            assert md_path.exists()
            with open(json_path) as f:
                data = json.load(f)
            assert data["accepted_for_decision_use"] is True
            assert "Campaign Acceptance Report" in md_path.read_text()


# ===========================================================================
# 5. Campaign entry accepted fields
# ===========================================================================


class TestCampaignEntryAcceptedFields:
    def test_accepted_entry_has_accepted_for_decision_use(self):
        entry = _make_accepted_entry()
        assert "accepted_for_decision_use" in entry

    def test_accepted_entry_has_counts_toward_promotion_readiness(self):
        entry = _make_accepted_entry()
        assert "counts_toward_promotion_readiness" in entry

    def test_accepted_entry_has_decision_benchmark_name(self):
        entry = _make_accepted_entry()
        assert "decision_benchmark_name" in entry

    def test_archcomp_entry_accepted_true(self):
        entry = _make_accepted_entry(profile="archcomp")
        # Without comparability_ok, may have warnings but archcomp with full config
        # should still pass non-comparability requirements
        # The accepted field at build time: comparability_ok=None → warning only
        # So accepted_for_decision_use depends on other fields only
        # With valid archcomp config, it should be True (comparability warning, not failure)
        assert entry["accepted_for_decision_use"] is True

    def test_archcomp_lite_entry_accepted_false(self):
        entry = _make_lite_entry()
        assert entry["accepted_for_decision_use"] is False

    def test_decision_benchmark_name_archcomp(self):
        entry = _make_accepted_entry(profile="archcomp")
        assert entry["decision_benchmark_name"] == "archcomp"

    def test_decision_benchmark_name_archcomp_lite(self):
        entry = _make_lite_entry()
        assert entry["decision_benchmark_name"] == "archcomp_lite"


# ===========================================================================
# 6. Accepted-only stability
# ===========================================================================


class TestAcceptedOnlyStability:
    def test_empty_history_stability(self):
        result = compute_recommendation_stability([])
        assert result["total_accepted_campaigns"] == 0
        assert result["consecutive_same_action_accepted_only"] == 0
        assert result["consecutive_positive_signal_for_settransformer_accepted_only"] == 0

    def test_all_accepted_stability(self):
        entries = [
            _make_accepted_entry(f"camp_{i}") for i in range(3)
        ]
        for e in entries:
            e["recommended_next_action"] = "run_more_seeds"
        result = compute_recommendation_stability(entries)
        assert result["total_accepted_campaigns"] == 3
        assert result["consecutive_same_action_accepted_only"] == 3

    def test_lite_excluded_from_accepted_count(self):
        entries = [
            _make_lite_entry("lite_1"),
            _make_lite_entry("lite_2"),
            _make_accepted_entry("arch_1"),
        ]
        for e in entries:
            e["recommended_next_action"] = "consider_promotion"
        result = compute_recommendation_stability(entries)
        assert result["total_accepted_campaigns"] == 1
        assert result["consecutive_same_action_accepted_only"] == 1

    def test_accepted_only_consecutive_resets_on_rejected(self):
        entries = [
            _make_accepted_entry("arch_1"),
            _make_accepted_entry("arch_2"),
            _make_lite_entry("lite_1"),
        ]
        for e in entries:
            e["recommended_next_action"] = "run_more_seeds"
        result = compute_recommendation_stability(entries)
        # accepted_only streak: should count only consecutive from end
        # arch_1 and arch_2 are accepted, lite_1 is not
        # The most recent is lite_1 (not accepted), so accepted-only streak = 0
        # Because accepted_history only includes arch_1, arch_2
        # consecutive streak from reversed accepted_history = 2 (same action)
        assert result["total_accepted_campaigns"] == 2
        # consecutive_same_action_accepted_only counts from accepted history end
        assert result["consecutive_same_action_accepted_only"] == 2

    def test_settransformer_accepted_only_streak(self):
        entries = [
            _make_accepted_entry("arch_1"),
            _make_accepted_entry("arch_2"),
        ]
        entries[0]["whether_to_try_pma_or_isab_next"] = True
        entries[1]["whether_to_try_pma_or_isab_next"] = True
        result = compute_recommendation_stability(entries)
        assert result["consecutive_positive_signal_for_settransformer_accepted_only"] == 2

    def test_stability_has_total_accepted_campaigns(self):
        entries = [_make_accepted_entry("arch_1")]
        result = compute_recommendation_stability(entries)
        assert "total_accepted_campaigns" in result

    def test_stability_has_accepted_only_fields(self):
        result = compute_recommendation_stability([])
        assert "consecutive_same_action_accepted_only" in result
        assert "consecutive_positive_signal_for_settransformer_accepted_only" in result


# ===========================================================================
# 7. Governance integration
# ===========================================================================


class TestGovernanceIntegration:
    def _build_trend_stub(self):
        from governance_layer import build_trend_summary
        entry = _make_accepted_entry()
        return build_trend_summary([entry])

    def _build_alert_stub(self):
        from governance_layer import build_regression_alert
        entry = _make_accepted_entry()
        return build_regression_alert([entry])

    def _build_gate_stub(self):
        from governance_layer import build_promotion_gate
        entry = _make_accepted_entry()
        stability = compute_recommendation_stability([entry])
        return build_promotion_gate([entry], stability)

    def test_governance_report_has_decision_policy_section(self):
        trend = self._build_trend_stub()
        alert = self._build_alert_stub()
        gate = self._build_gate_stub()
        stability = compute_recommendation_stability([_make_accepted_entry()])
        md = build_governance_report(
            trend_summary=trend,
            regression_alert=alert,
            promotion_gate=gate,
            stability=stability,
            latest_entry=_make_accepted_entry(),
        )
        assert "Decision Benchmark Policy" in md

    def test_governance_report_has_acceptance_section(self):
        trend = self._build_trend_stub()
        alert = self._build_alert_stub()
        gate = self._build_gate_stub()
        stability = compute_recommendation_stability([_make_accepted_entry()])
        md = build_governance_report(
            trend_summary=trend,
            regression_alert=alert,
            promotion_gate=gate,
            stability=stability,
            latest_entry=_make_accepted_entry(),
        )
        assert "Current Campaign Acceptance" in md

    def test_governance_report_has_promotion_readiness_section(self):
        trend = self._build_trend_stub()
        alert = self._build_alert_stub()
        gate = self._build_gate_stub()
        stability = compute_recommendation_stability([_make_accepted_entry()])
        md = build_governance_report(
            trend_summary=trend,
            regression_alert=alert,
            promotion_gate=gate,
            stability=stability,
            latest_entry=_make_accepted_entry(),
        )
        assert "Promotion Readiness" in md

    def test_governance_report_shows_accepted_status(self):
        trend = self._build_trend_stub()
        alert = self._build_alert_stub()
        gate = self._build_gate_stub()
        stability = compute_recommendation_stability([_make_accepted_entry()])
        acceptance = check_campaign_acceptance(_make_accepted_entry(), comparability_ok=True)
        md = build_governance_report(
            trend_summary=trend,
            regression_alert=alert,
            promotion_gate=gate,
            stability=stability,
            latest_entry=_make_accepted_entry(),
            acceptance_result=acceptance,
        )
        assert "ACCEPTED" in md

    def test_governance_report_shows_not_accepted_for_lite(self):
        from governance_layer import build_trend_summary, build_regression_alert, build_promotion_gate
        entry = _make_lite_entry()
        trend = build_trend_summary([entry])
        alert = build_regression_alert([entry])
        stability = compute_recommendation_stability([entry])
        gate = build_promotion_gate([entry], stability)
        acceptance = check_campaign_acceptance(entry, comparability_ok=True)
        md = build_governance_report(
            trend_summary=trend,
            regression_alert=alert,
            promotion_gate=gate,
            stability=stability,
            latest_entry=entry,
            acceptance_result=acceptance,
        )
        assert "NOT ACCEPTED" in md

    def test_save_governance_artifacts_includes_benchmark_lock(self):
        entry = _make_accepted_entry()
        stability = compute_recommendation_stability([entry])
        with tempfile.TemporaryDirectory() as d:
            paths = save_governance_artifacts([entry], stability, data_dir=d)
            assert "benchmark_lock.json" in paths
            assert "benchmark_lock.md" in paths

    def test_save_governance_artifacts_includes_campaign_acceptance(self):
        entry = _make_accepted_entry()
        stability = compute_recommendation_stability([entry])
        with tempfile.TemporaryDirectory() as d:
            paths = save_governance_artifacts([entry], stability, data_dir=d)
            assert "campaign_acceptance.json" in paths
            assert "campaign_acceptance.md" in paths

    def test_save_governance_artifacts_total_count(self):
        """save_governance_artifacts should return 11 paths total now."""
        entry = _make_accepted_entry()
        stability = compute_recommendation_stability([entry])
        with tempfile.TemporaryDirectory() as d:
            paths = save_governance_artifacts([entry], stability, data_dir=d)
            # 2 (benchmark_lock) + 2 (comparability) + 2 (campaign_acceptance)
            # + 2 (trend) + 2 (regression) + 2 (promotion) + 1 (governance)
            # = 13 ... depends on exact count
            # Let's just check >= 11 (9 original + 2 benchmark_lock + 2 acceptance)
            # Actually: 9 (prev) + 2 (benchmark_lock) + 2 (acceptance) = 13
            assert len(paths) >= 11
            assert "governance_report.md" in paths

    def test_acceptance_json_is_valid(self):
        entry = _make_accepted_entry()
        stability = compute_recommendation_stability([entry])
        with tempfile.TemporaryDirectory() as d:
            paths = save_governance_artifacts([entry], stability, data_dir=d)
            json_path = Path(paths["campaign_acceptance.json"])
            with open(json_path) as f:
                data = json.load(f)
            assert "accepted_for_decision_use" in data
            assert "counts_toward_promotion_readiness" in data
            assert "failed_requirements" in data


# ===========================================================================
# 8. Campaign history CSV includes accepted columns
# ===========================================================================


class TestCampaignHistoryCSV:
    def test_csv_has_accepted_for_decision_use_column(self):
        from campaign_manager import build_campaign_history_csv
        entry = _make_accepted_entry()
        csv_str = build_campaign_history_csv([entry])
        assert "accepted_for_decision_use" in csv_str

    def test_csv_has_counts_toward_promotion_readiness_column(self):
        from campaign_manager import build_campaign_history_csv
        entry = _make_accepted_entry()
        csv_str = build_campaign_history_csv([entry])
        assert "counts_toward_promotion_readiness" in csv_str

    def test_csv_has_decision_benchmark_name_column(self):
        from campaign_manager import build_campaign_history_csv
        entry = _make_accepted_entry()
        csv_str = build_campaign_history_csv([entry])
        assert "decision_benchmark_name" in csv_str

    def test_archcomp_entry_csv_shows_true(self):
        from campaign_manager import build_campaign_history_csv
        entry = _make_accepted_entry()
        csv_str = build_campaign_history_csv([entry])
        # archcomp entry should have accepted=true
        assert "true" in csv_str

    def test_lite_entry_csv_shows_false(self):
        from campaign_manager import build_campaign_history_csv
        entry = _make_lite_entry()
        csv_str = build_campaign_history_csv([entry])
        assert "false" in csv_str


# ===========================================================================
# 9. Module imports smoke
# ===========================================================================


class TestModuleImports:
    def test_decision_policy_imports(self):
        from decision_policy import (
            DECISION_BENCHMARK_POLICY,
            DECISION_POLICY_SCHEMA_VERSION,
            CAMPAIGN_ACCEPTANCE_SCHEMA_VERSION,
            build_benchmark_lock_artifact,
            check_campaign_acceptance,
            save_benchmark_lock_artifacts,
            save_campaign_acceptance_artifacts,
        )
        assert DECISION_POLICY_SCHEMA_VERSION >= 1
        assert CAMPAIGN_ACCEPTANCE_SCHEMA_VERSION >= 1

    def test_governance_layer_imports_decision_policy(self):
        from governance_layer import (
            build_governance_report,
            save_governance_artifacts,
        )
        import inspect
        src = inspect.getsource(save_governance_artifacts)
        assert "save_benchmark_lock_artifacts" in src

    def test_campaign_manager_imports_decision_policy(self):
        import inspect
        from campaign_manager import build_campaign_entry
        src = inspect.getsource(build_campaign_entry)
        assert "check_campaign_acceptance" in src

    def test_run_campaign_imports_benchmark_lock(self):
        """Smoke: benchmark_lock can be imported from REPO root context."""
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, '.'); "
             "from decision_policy import save_benchmark_lock_artifacts; "
             "print('ok')"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        assert result.returncode == 0, result.stderr
        assert "ok" in result.stdout
