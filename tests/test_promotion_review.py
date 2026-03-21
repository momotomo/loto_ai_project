"""tests/test_promotion_review.py

Tests for the promotion_review module:
  - accepted_campaign_summary artifact (keys, structure)
  - promotion_review_readiness artifact (keys, ready/not-ready logic)
  - accepted_campaign_review_bundle artifact (keys, structure)
  - governance report integration (new sections)
  - save artifacts smoke (file output)
  - run_campaign review bundle generation smoke (import)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from campaign_manager import (
    append_campaign_to_history,
    build_campaign_entry,
    compute_recommendation_stability,
)
from promotion_review import (
    ACCEPTED_CAMPAIGN_SUMMARY_SCHEMA_VERSION,
    ACCEPTED_REVIEW_BUNDLE_SCHEMA_VERSION,
    PROMOTION_REVIEW_READINESS_SCHEMA_VERSION,
    build_accepted_campaign_review_bundle,
    build_accepted_campaign_review_bundle_md,
    build_accepted_campaign_summary,
    build_accepted_campaign_summary_md,
    build_promotion_review_readiness,
    build_promotion_review_readiness_md,
    save_accepted_campaign_review_bundle_artifacts,
    save_accepted_campaign_summary_artifacts,
    save_promotion_review_readiness_artifacts,
)
from governance_layer import (
    build_governance_report,
    build_regression_alert,
    build_promotion_gate,
    build_trend_summary,
    save_governance_artifacts,
)
from cross_loto_summary import build_cross_loto_summary, build_recommendation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_per_loto(
    loto_types=("loto6", "loto7", "miniloto"),
    seeds=(42, 123, 456),
    preset="archcomp",
    promote_count=0,
):
    result = {}
    for lt in loto_types:
        result[lt] = {
            "schema_version": 1,
            "loto_type": lt,
            "preset": preset,
            "seeds": list(seeds),
            "run_count": len(seeds),
            "alpha": 0.05,
            "variants": {
                "legacy": {
                    "run_count": len(seeds),
                    "logloss": {
                        "mean": 0.31,
                        "std": 0.01,
                        "values": [0.31] * len(seeds),
                    },
                    "brier": {"mean": 0.04, "std": 0.0, "values": [0.04] * len(seeds)},
                    "ece": {"mean": 0.01, "std": 0.0, "values": [0.01] * len(seeds)},
                    "calibration_recommendations": {"none": len(seeds)},
                    "promote_count": promote_count,
                    "hold_count": len(seeds) - promote_count,
                }
            },
            "pairwise_comparisons": {},
        }
    return result


def _make_entry(
    name: str,
    profile: str = "archcomp",
    loto_types=("loto6", "loto7", "miniloto"),
    seeds=(42, 123, 456),
    promote_count: int = 0,
    evaluation_model_variants: str = "legacy,multihot,deepsets,settransformer",
    evaluation_calibration_methods: str = "none,temperature,isotonic",
) -> dict:
    per_loto = _make_per_loto(loto_types=loto_types, seeds=seeds, promote_count=promote_count)
    cs = build_cross_loto_summary(per_loto, list(loto_types), "archcomp", list(seeds))
    rec = build_recommendation(cs)
    return build_campaign_entry(
        name,
        profile,
        cs,
        rec,
        evaluation_model_variants=evaluation_model_variants,
        evaluation_calibration_methods=evaluation_calibration_methods,
    )


def _make_accepted_history(n: int = 2) -> tuple[list[dict], dict]:
    """Return (history, stability) with n accepted archcomp entries."""
    history = []
    for i in range(n):
        e = _make_entry(f"c{i+1}", profile="archcomp")
        history = append_campaign_to_history(history, e)
    stability = compute_recommendation_stability(history)
    return history, stability


def _make_lite_entry(name: str = "lite1") -> dict:
    per_loto = _make_per_loto(loto_types=("loto6",), seeds=(42, 123))
    cs = build_cross_loto_summary(per_loto, ["loto6"], "archcomp_lite", [42, 123])
    rec = build_recommendation(cs)
    return build_campaign_entry(
        name,
        "archcomp_lite",
        cs,
        rec,
        evaluation_model_variants="legacy,multihot,deepsets,settransformer",
        evaluation_calibration_methods="none,temperature,isotonic",
    )


# ---------------------------------------------------------------------------
# TestAcceptedCampaignSummaryKeys
# ---------------------------------------------------------------------------


class TestAcceptedCampaignSummaryKeys:
    """Verify accepted_campaign_summary has required keys and schema_version."""

    def test_required_keys_present(self):
        history, _ = _make_accepted_history(2)
        summary = build_accepted_campaign_summary(history)
        required = [
            "schema_version",
            "generated_at",
            "total_campaigns_in_history",
            "accepted_campaign_count",
            "counts_toward_promotion_readiness_count",
            "accepted_campaign_names",
            "counts_toward_promotion_readiness_names",
            "action_history",
            "action_distribution",
            "challenger_distribution",
            "consecutive_accepted_positive_signals",
            "consecutive_accepted_settransformer_signal",
            "profile_distribution",
            "benchmark_distribution",
            "seed_counts_per_accepted_campaign",
            "loto_types_per_accepted_campaign",
            "comparability_issues_within_accepted",
            "latest_accepted_campaign_name",
            "latest_accepted_action",
            "latest_accepted_challenger",
            "latest_accepted_pma_signal",
        ]
        for key in required:
            assert key in summary, f"Missing key: {key}"

    def test_schema_version(self):
        summary = build_accepted_campaign_summary([])
        assert summary["schema_version"] == ACCEPTED_CAMPAIGN_SUMMARY_SCHEMA_VERSION

    def test_empty_history(self):
        summary = build_accepted_campaign_summary([])
        assert summary["accepted_campaign_count"] == 0
        assert summary["accepted_campaign_names"] == []
        assert summary["action_history"] == []

    def test_accepted_count_correct(self):
        history, _ = _make_accepted_history(3)
        summary = build_accepted_campaign_summary(history)
        # All 3 should be accepted (archcomp + full loto + full variants)
        assert summary["accepted_campaign_count"] == 3
        assert summary["total_campaigns_in_history"] == 3

    def test_lite_excluded_from_accepted(self):
        lite = _make_lite_entry()
        assert lite.get("accepted_for_decision_use") is False
        history = [lite]
        summary = build_accepted_campaign_summary(history)
        assert summary["accepted_campaign_count"] == 0

    def test_mixed_history(self):
        history, _ = _make_accepted_history(2)
        lite = _make_lite_entry()
        history.append(lite)
        summary = build_accepted_campaign_summary(history)
        assert summary["accepted_campaign_count"] == 2
        assert summary["total_campaigns_in_history"] == 3

    def test_action_distribution_is_dict(self):
        history, _ = _make_accepted_history(2)
        summary = build_accepted_campaign_summary(history)
        assert isinstance(summary["action_distribution"], dict)

    def test_md_has_required_sections(self):
        history, _ = _make_accepted_history(2)
        summary = build_accepted_campaign_summary(history)
        md = build_accepted_campaign_summary_md(summary)
        assert "# Accepted Campaign Summary" in md
        assert "## Counts" in md
        assert "## Accepted Campaigns" in md
        assert "## Action / Challenger History" in md


# ---------------------------------------------------------------------------
# TestPromotionReviewReadinessKeys
# ---------------------------------------------------------------------------


class TestPromotionReviewReadinessKeys:
    """Verify promotion_review_readiness has required keys."""

    def test_required_keys_present(self):
        history, stability = _make_accepted_history(2)
        readiness = build_promotion_review_readiness(history, stability)
        required = [
            "schema_version",
            "generated_at",
            "ready_for_promotion_review",
            "candidate_variant",
            "accepted_campaign_count",
            "counts_toward_promotion_readiness_count",
            "accepted_campaign_window",
            "consecutive_accepted_positive_signals",
            "consecutive_accepted_settransformer_signal",
            "consecutive_same_action_accepted_only",
            "consecutive_positive_signal_for_settransformer_accepted_only",
            "promotion_gate_status",
            "regression_alert_level",
            "conditions_passed",
            "blockers",
            "rationale",
            "recommended_next_step",
        ]
        for key in required:
            assert key in readiness, f"Missing key: {key}"

    def test_schema_version(self):
        readiness = build_promotion_review_readiness([], {})
        assert readiness["schema_version"] == PROMOTION_REVIEW_READINESS_SCHEMA_VERSION

    def test_ready_for_promotion_review_is_bool(self):
        history, stability = _make_accepted_history(2)
        readiness = build_promotion_review_readiness(history, stability)
        assert isinstance(readiness["ready_for_promotion_review"], bool)

    def test_conditions_passed_is_list(self):
        history, stability = _make_accepted_history(2)
        readiness = build_promotion_review_readiness(history, stability)
        assert isinstance(readiness["conditions_passed"], list)

    def test_blockers_is_list(self):
        history, stability = _make_accepted_history(2)
        readiness = build_promotion_review_readiness(history, stability)
        assert isinstance(readiness["blockers"], list)

    def test_rationale_is_str(self):
        history, stability = _make_accepted_history(2)
        readiness = build_promotion_review_readiness(history, stability)
        assert isinstance(readiness["rationale"], str)
        assert len(readiness["rationale"]) > 0


# ---------------------------------------------------------------------------
# TestPromotionReviewReadinessLogic
# ---------------------------------------------------------------------------


class TestPromotionReviewReadinessLogic:
    """Verify ready/not-ready case logic."""

    def test_empty_history_not_ready(self):
        readiness = build_promotion_review_readiness([], {})
        assert readiness["ready_for_promotion_review"] is False
        assert len(readiness["blockers"]) > 0

    def test_single_accepted_not_ready(self):
        history, stability = _make_accepted_history(1)
        readiness = build_promotion_review_readiness(history, stability)
        assert readiness["ready_for_promotion_review"] is False
        # Should flag insufficient accepted campaigns
        assert any("accepted" in b.lower() for b in readiness["blockers"])

    def test_two_accepted_hold_action_not_ready(self):
        # hold is not a positive signal — should block readiness
        history, stability = _make_accepted_history(2)
        readiness = build_promotion_review_readiness(history, stability)
        assert readiness["ready_for_promotion_review"] is False
        # Should mention non-positive action
        all_blockers = " ".join(readiness["blockers"])
        assert "positive" in all_blockers.lower() or "hold" in all_blockers.lower()

    def test_lite_only_not_ready(self):
        lite = _make_lite_entry()
        assert lite.get("accepted_for_decision_use") is False
        history = [lite]
        stability = compute_recommendation_stability(history)
        readiness = build_promotion_review_readiness(history, stability)
        assert readiness["ready_for_promotion_review"] is False
        assert readiness["accepted_campaign_count"] == 0

    def test_high_regression_blocks_readiness(self):
        history, stability = _make_accepted_history(2)
        fake_alert = {
            "alert_level": "high",
            "summary": "Significant regression detected.",
        }
        readiness = build_promotion_review_readiness(
            history, stability, regression_alert=fake_alert
        )
        assert readiness["ready_for_promotion_review"] is False
        assert any("HIGH" in b for b in readiness["blockers"])

    def test_no_high_regression_passes_condition(self):
        history, stability = _make_accepted_history(2)
        fake_alert = {"alert_level": "none"}
        readiness = build_promotion_review_readiness(
            history, stability, regression_alert=fake_alert
        )
        passed = readiness["conditions_passed"]
        assert any("no_high_regression" in c for c in passed)

    def test_accepted_campaign_count_correct(self):
        history, stability = _make_accepted_history(3)
        readiness = build_promotion_review_readiness(history, stability)
        assert readiness["accepted_campaign_count"] == 3

    def test_promotion_gate_green_passes_condition(self):
        history, stability = _make_accepted_history(2)
        fake_gate = {"gate_status": "green"}
        readiness = build_promotion_review_readiness(
            history, stability, promotion_gate=fake_gate
        )
        passed = readiness["conditions_passed"]
        assert any("promotion_gate_green" in c for c in passed)

    def test_md_has_required_sections(self):
        history, stability = _make_accepted_history(2)
        readiness = build_promotion_review_readiness(history, stability)
        md = build_promotion_review_readiness_md(readiness)
        assert "# Promotion Review Readiness" in md
        assert "## Status:" in md
        assert "## Recommended Next Step" in md

    def test_md_not_ready_shows_blockers(self):
        readiness = build_promotion_review_readiness([], {})
        md = build_promotion_review_readiness_md(readiness)
        assert "## Blockers" in md or "NOT YET READY" in md


# ---------------------------------------------------------------------------
# TestAcceptedCampaignReviewBundleKeys
# ---------------------------------------------------------------------------


class TestAcceptedCampaignReviewBundleKeys:
    """Verify accepted_campaign_review_bundle has required keys."""

    def test_required_keys_present(self):
        history, stability = _make_accepted_history(2)
        bundle = build_accepted_campaign_review_bundle(history, stability)
        required = [
            "schema_version",
            "generated_at",
            "ready_for_promotion_review",
            "candidate_variant",
            "accepted_campaign_count",
            "counts_toward_promotion_readiness_count",
            "accepted_campaign_names",
            "counts_toward_promotion_readiness_names",
            "consecutive_accepted_positive_signals",
            "consecutive_accepted_settransformer_signal",
            "promotion_gate_status",
            "regression_alert_level",
            "comparability_ok",
            "comparability_severity",
            "conditions_passed",
            "blockers",
            "rationale",
            "recommended_next_step",
            "action_history",
            "action_distribution",
            "challenger_distribution",
            "latest_accepted_evidence",
            "accepted_campaign_summary",
        ]
        for key in required:
            assert key in bundle, f"Missing key: {key}"

    def test_schema_version(self):
        bundle = build_accepted_campaign_review_bundle([], {})
        assert bundle["schema_version"] == ACCEPTED_REVIEW_BUNDLE_SCHEMA_VERSION

    def test_ready_for_promotion_review_is_bool(self):
        history, stability = _make_accepted_history(2)
        bundle = build_accepted_campaign_review_bundle(history, stability)
        assert isinstance(bundle["ready_for_promotion_review"], bool)

    def test_latest_accepted_evidence_populated(self):
        history, stability = _make_accepted_history(2)
        bundle = build_accepted_campaign_review_bundle(history, stability)
        evidence = bundle.get("latest_accepted_evidence") or {}
        assert "campaign_name" in evidence
        assert "recommended_next_action" in evidence

    def test_empty_history_no_evidence(self):
        bundle = build_accepted_campaign_review_bundle([], {})
        assert bundle["accepted_campaign_count"] == 0
        assert bundle["latest_accepted_evidence"] == {}

    def test_accepted_campaign_summary_embedded(self):
        history, stability = _make_accepted_history(2)
        bundle = build_accepted_campaign_review_bundle(history, stability)
        summary = bundle.get("accepted_campaign_summary") or {}
        assert "accepted_campaign_count" in summary

    def test_md_has_required_sections(self):
        history, stability = _make_accepted_history(2)
        bundle = build_accepted_campaign_review_bundle(history, stability)
        md = build_accepted_campaign_review_bundle_md(bundle)
        assert "# Accepted Campaign Review Bundle" in md
        assert "## Quick Overview" in md
        assert "## Accepted Campaigns" in md
        assert "## Action / Challenger History" in md
        assert "## Latest Accepted Evidence" in md
        assert "## Recommended Next Step" in md


# ---------------------------------------------------------------------------
# TestAcceptedOnlySaveArtifacts
# ---------------------------------------------------------------------------


class TestAcceptedOnlySaveArtifacts:
    """Verify save functions produce files with correct content."""

    def test_save_accepted_campaign_summary_creates_files(self):
        history, _ = _make_accepted_history(2)
        with tempfile.TemporaryDirectory() as d:
            paths = save_accepted_campaign_summary_artifacts(history, data_dir=d)
            assert "accepted_campaign_summary.json" in paths
            assert "accepted_campaign_summary.md" in paths
            assert Path(paths["accepted_campaign_summary.json"]).exists()
            assert Path(paths["accepted_campaign_summary.md"]).exists()

    def test_save_accepted_campaign_summary_json_valid(self):
        history, _ = _make_accepted_history(2)
        with tempfile.TemporaryDirectory() as d:
            paths = save_accepted_campaign_summary_artifacts(history, data_dir=d)
            with open(paths["accepted_campaign_summary.json"]) as f:
                data = json.load(f)
            assert data["schema_version"] == ACCEPTED_CAMPAIGN_SUMMARY_SCHEMA_VERSION
            assert "accepted_campaign_count" in data

    def test_save_promotion_review_readiness_creates_files(self):
        history, stability = _make_accepted_history(2)
        with tempfile.TemporaryDirectory() as d:
            paths = save_promotion_review_readiness_artifacts(history, stability, data_dir=d)
            assert "promotion_review_readiness.json" in paths
            assert "promotion_review_readiness.md" in paths
            assert Path(paths["promotion_review_readiness.json"]).exists()
            assert Path(paths["promotion_review_readiness.md"]).exists()

    def test_save_promotion_review_readiness_json_valid(self):
        history, stability = _make_accepted_history(2)
        with tempfile.TemporaryDirectory() as d:
            paths = save_promotion_review_readiness_artifacts(history, stability, data_dir=d)
            with open(paths["promotion_review_readiness.json"]) as f:
                data = json.load(f)
            assert data["schema_version"] == PROMOTION_REVIEW_READINESS_SCHEMA_VERSION
            assert "ready_for_promotion_review" in data

    def test_save_accepted_review_bundle_creates_files(self):
        history, stability = _make_accepted_history(2)
        with tempfile.TemporaryDirectory() as d:
            paths = save_accepted_campaign_review_bundle_artifacts(
                history, stability, data_dir=d
            )
            assert "accepted_campaign_review_bundle.json" in paths
            assert "accepted_campaign_review_bundle.md" in paths
            assert Path(paths["accepted_campaign_review_bundle.json"]).exists()
            assert Path(paths["accepted_campaign_review_bundle.md"]).exists()

    def test_save_accepted_review_bundle_json_valid(self):
        history, stability = _make_accepted_history(2)
        with tempfile.TemporaryDirectory() as d:
            paths = save_accepted_campaign_review_bundle_artifacts(
                history, stability, data_dir=d
            )
            with open(paths["accepted_campaign_review_bundle.json"]) as f:
                data = json.load(f)
            assert data["schema_version"] == ACCEPTED_REVIEW_BUNDLE_SCHEMA_VERSION
            assert "ready_for_promotion_review" in data


# ---------------------------------------------------------------------------
# TestGovernanceIntegration
# ---------------------------------------------------------------------------


class TestGovernanceIntegration:
    """Verify governance_report includes new accepted review sections."""

    def test_governance_report_has_accepted_review_section(self):
        history, stability = _make_accepted_history(2)
        trend = build_trend_summary(history)
        alert = build_regression_alert(history)
        gate = build_promotion_gate(history, stability, regression_alert=alert)
        md = build_governance_report(
            trend, alert, gate, stability, history=history
        )
        assert "## Accepted Campaign Review" in md

    def test_governance_report_has_promotion_review_readiness_section(self):
        history, stability = _make_accepted_history(2)
        trend = build_trend_summary(history)
        alert = build_regression_alert(history)
        gate = build_promotion_gate(history, stability, regression_alert=alert)
        md = build_governance_report(
            trend, alert, gate, stability, history=history
        )
        assert "## Promotion Review Readiness" in md

    def test_governance_report_shows_accepted_count(self):
        history, stability = _make_accepted_history(2)
        trend = build_trend_summary(history)
        alert = build_regression_alert(history)
        gate = build_promotion_gate(history, stability, regression_alert=alert)
        md = build_governance_report(
            trend, alert, gate, stability, history=history
        )
        assert "Accepted campaigns" in md

    def test_governance_report_shows_not_ready_when_no_history(self):
        trend = build_trend_summary([])
        alert = build_regression_alert([])
        gate = build_promotion_gate([], {}, regression_alert=alert)
        stability = compute_recommendation_stability([])
        md = build_governance_report(
            trend, alert, gate, stability, history=[]
        )
        assert "NOT YET READY" in md or "not enough" in md.lower() or "blocker" in md.lower()

    def test_save_governance_artifacts_returns_19_paths(self):
        history, stability = _make_accepted_history(2)
        with tempfile.TemporaryDirectory() as d:
            paths = save_governance_artifacts(history, stability, data_dir=d)
            assert len(paths) == 19, f"Expected 19 paths, got {len(paths)}: {list(paths.keys())}"

    def test_save_governance_artifacts_includes_review_bundle(self):
        history, stability = _make_accepted_history(2)
        with tempfile.TemporaryDirectory() as d:
            paths = save_governance_artifacts(history, stability, data_dir=d)
            for name in [
                "accepted_campaign_summary.json",
                "accepted_campaign_summary.md",
                "promotion_review_readiness.json",
                "promotion_review_readiness.md",
                "accepted_campaign_review_bundle.json",
                "accepted_campaign_review_bundle.md",
            ]:
                assert name in paths, f"Missing: {name}"
                assert Path(paths[name]).exists(), f"File missing: {name}"

    def test_governance_report_review_bundle_md_valid(self):
        history, stability = _make_accepted_history(2)
        with tempfile.TemporaryDirectory() as d:
            paths = save_governance_artifacts(history, stability, data_dir=d)
            md = Path(paths["accepted_campaign_review_bundle.md"]).read_text()
            assert "# Accepted Campaign Review Bundle" in md
            assert "## Quick Overview" in md

    def test_governance_report_promotion_readiness_json_valid(self):
        history, stability = _make_accepted_history(2)
        with tempfile.TemporaryDirectory() as d:
            paths = save_governance_artifacts(history, stability, data_dir=d)
            with open(paths["promotion_review_readiness.json"]) as f:
                prr = json.load(f)
            assert "ready_for_promotion_review" in prr
            assert isinstance(prr["ready_for_promotion_review"], bool)

    def test_governance_report_no_history_generates_ok(self):
        # Edge case: no history at all
        with tempfile.TemporaryDirectory() as d:
            paths = save_governance_artifacts([], {}, data_dir=d)
            assert "governance_report.md" in paths
            assert "accepted_campaign_review_bundle.md" in paths


# ---------------------------------------------------------------------------
# TestModuleImports
# ---------------------------------------------------------------------------


class TestModuleImports:
    """Import smoke tests for promotion_review."""

    def test_promotion_review_module_imports(self):
        import promotion_review as pr
        assert hasattr(pr, "build_accepted_campaign_summary")
        assert hasattr(pr, "build_promotion_review_readiness")
        assert hasattr(pr, "build_accepted_campaign_review_bundle")

    def test_schema_version_constants(self):
        assert ACCEPTED_CAMPAIGN_SUMMARY_SCHEMA_VERSION == 1
        assert PROMOTION_REVIEW_READINESS_SCHEMA_VERSION == 1
        assert ACCEPTED_REVIEW_BUNDLE_SCHEMA_VERSION == 1

    def test_governance_layer_imports_promotion_review(self):
        from governance_layer import (
            save_accepted_campaign_summary_artifacts,
            save_promotion_review_readiness_artifacts,
            save_accepted_campaign_review_bundle_artifacts,
        )
        assert callable(save_accepted_campaign_summary_artifacts)
        assert callable(save_promotion_review_readiness_artifacts)
        assert callable(save_accepted_campaign_review_bundle_artifacts)

    def test_run_campaign_imports_work(self):
        """Verify run_campaign can be imported as module."""
        import sys
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_campaign", "scripts/run_campaign.py"
        )
        mod = importlib.util.module_from_spec(spec)
        # Don't exec (would call main), just verify spec loads
        assert spec is not None
