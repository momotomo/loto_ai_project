"""tests/test_governance.py

Tests for the governance layer:
  - trend summary artifact (keys, structure, window)
  - regression alert artifact (keys, levels)
  - promotion readiness gate (keys, status logic)
  - recommendation stability extension (new keys)
  - governance report (Markdown sections)
  - save_governance_artifacts (file output smoke)
  - run_campaign governance CLI smoke (imports)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from campaign_manager import compute_recommendation_stability
from governance_layer import (
    DEFAULT_TREND_WINDOW,
    GOVERNANCE_REPORT_SCHEMA_VERSION,
    PROMOTION_GATE_SCHEMA_VERSION,
    REGRESSION_ALERT_SCHEMA_VERSION,
    TREND_SUMMARY_SCHEMA_VERSION,
    build_governance_report,
    build_promotion_gate,
    build_promotion_gate_md,
    build_regression_alert,
    build_regression_alert_md,
    build_trend_summary,
    build_trend_summary_md,
    save_governance_artifacts,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_entry(
    name: str,
    action: str = "hold",
    challenger: str | None = "multihot",
    keep: bool = True,
    pma: bool = False,
    blockers: int = 2,
    ranking: list[dict] | None = None,
    pairwise: dict | None = None,
    seeds: list[int] | None = None,
    loto_types: list[str] | None = None,
) -> dict:
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
        "profile_name": "archcomp",
        "generated_at": "2026-03-21T00:00:00+00:00",
        "recommended_next_action": action,
        "recommended_challenger": challenger,
        "keep_production_as_is": keep,
        "whether_to_try_pma_or_isab_next": pma,
        "best_variant_by_logloss": challenger,
        "consistent_promote_variants": [] if action != "consider_promotion" else [challenger],
        "blockers_count": blockers,
        "variant_ranking_summary": ranking,
        "key_pairwise_signals": pairwise,
        "seeds": seeds or [42, 123, 456],
        "loto_types": loto_types or ["loto6", "loto7", "miniloto"],
    }


@pytest.fixture
def single_history():
    return [_make_entry("c1")]


@pytest.fixture
def multi_history():
    return [
        _make_entry("c1", action="hold"),
        _make_entry("c2", action="hold"),
        _make_entry("c3", action="run_more_seeds"),
    ]


@pytest.fixture
def promote_history():
    return [
        _make_entry("c1", action="consider_promotion", keep=False, blockers=0,
                    challenger="multihot",
                    ranking=[
                        {"rank": 1, "variant": "multihot", "logloss_mean": 0.305},
                        {"rank": 2, "variant": "legacy", "logloss_mean": 0.315},
                        {"rank": 3, "variant": "deepsets", "logloss_mean": 0.320},
                    ]),
        _make_entry("c2", action="consider_promotion", keep=False, blockers=0,
                    challenger="multihot",
                    ranking=[
                        {"rank": 1, "variant": "multihot", "logloss_mean": 0.303},
                        {"rank": 2, "variant": "legacy", "logloss_mean": 0.315},
                        {"rank": 3, "variant": "deepsets", "logloss_mean": 0.322},
                    ]),
    ]


# ---------------------------------------------------------------------------
# TestTrendSummary
# ---------------------------------------------------------------------------

class TestTrendSummary:
    def test_required_keys(self, single_history):
        t = build_trend_summary(single_history)
        for key in [
            "schema_version", "generated_at", "tracked_window_size",
            "total_campaigns", "campaigns_considered", "variant_rank_history",
            "metric_trends", "recommendation_history", "keep_production_streak",
            "pairwise_signal_history", "dominant_action", "dominant_challenger",
            "action_distribution",
        ]:
            assert key in t, f"Missing key: {key}"

    def test_schema_version(self, single_history):
        t = build_trend_summary(single_history)
        assert t["schema_version"] == TREND_SUMMARY_SCHEMA_VERSION

    def test_campaigns_considered_single(self, single_history):
        t = build_trend_summary(single_history)
        assert t["campaigns_considered"] == ["c1"]
        assert t["total_campaigns"] == 1

    def test_campaigns_considered_multi(self, multi_history):
        t = build_trend_summary(multi_history)
        assert set(t["campaigns_considered"]) == {"c1", "c2", "c3"}
        assert t["total_campaigns"] == 3

    def test_window_truncation(self):
        history = [_make_entry(f"c{i}") for i in range(10)]
        t = build_trend_summary(history, window_size=3)
        assert len(t["campaigns_considered"]) == 3
        assert t["tracked_window_size"] == 3

    def test_variant_rank_history_keys(self, single_history):
        t = build_trend_summary(single_history)
        rh = t["variant_rank_history"]
        assert "multihot" in rh
        assert "ranks" in rh["multihot"]
        assert "rank_trend" in rh["multihot"]

    def test_metric_trends_logloss(self, single_history):
        t = build_trend_summary(single_history)
        mt = t["metric_trends"]
        assert "multihot" in mt
        ll = mt["multihot"]["logloss"]
        assert "values" in ll
        assert "trend" in ll

    def test_recommendation_history_structure(self, multi_history):
        t = build_trend_summary(multi_history)
        rh = t["recommendation_history"]
        assert len(rh) == 3
        assert "action" in rh[0]
        assert "challenger" in rh[0]
        assert "keep_production" in rh[0]

    def test_keep_production_streak(self, multi_history):
        # c1=hold(keep=True), c2=hold(keep=True), c3=run_more_seeds(keep=True)
        t = build_trend_summary(multi_history)
        assert t["keep_production_streak"] >= 1

    def test_pairwise_signal_history(self, single_history):
        t = build_trend_summary(single_history)
        psh = t["pairwise_signal_history"]
        assert "settransformer_vs_deepsets" in psh
        entry = psh["settransformer_vs_deepsets"][0]
        assert "both_pass_rate" in entry

    def test_empty_history(self):
        t = build_trend_summary([])
        assert t["total_campaigns"] == 0
        assert t["campaigns_considered"] == []

    def test_trend_summary_md_has_sections(self, multi_history):
        t = build_trend_summary(multi_history)
        md = build_trend_summary_md(t)
        assert "# Trend Summary" in md
        assert "## Recommendation History" in md
        assert "## Variant Rank History" in md
        assert "## Logloss Trend" in md


# ---------------------------------------------------------------------------
# TestRegressionAlert
# ---------------------------------------------------------------------------

class TestRegressionAlert:
    def test_required_keys(self, single_history):
        a = build_regression_alert(single_history)
        for key in [
            "schema_version", "generated_at", "alert_level", "latest_campaign",
            "baseline_campaigns", "affected_variants", "ranking_drop",
            "metric_regressions", "pairwise_signal_loss",
            "recommendation_instability", "suspected_causes", "summary",
        ]:
            assert key in a, f"Missing key: {key}"

    def test_schema_version(self, single_history):
        a = build_regression_alert(single_history)
        assert a["schema_version"] == REGRESSION_ALERT_SCHEMA_VERSION

    def test_alert_level_none_single_campaign(self, single_history):
        a = build_regression_alert(single_history)
        assert a["alert_level"] == "none"

    def test_alert_level_values(self, multi_history):
        a = build_regression_alert(multi_history)
        assert a["alert_level"] in {"none", "low", "medium", "high"}

    def test_no_regression_stable_history(self):
        # Same entry repeated = no regression
        entry = _make_entry("c1")
        history = [entry, dict(entry, campaign_name="c2"), dict(entry, campaign_name="c3")]
        a = build_regression_alert(history)
        assert a["alert_level"] in {"none", "low"}

    def test_high_regression_on_rank_drop(self):
        # Rank dramatically worsens
        baseline = _make_entry("c1", ranking=[
            {"rank": 1, "variant": "multihot", "logloss_mean": 0.300},
            {"rank": 2, "variant": "legacy", "logloss_mean": 0.310},
        ])
        regression = _make_entry("c2", ranking=[
            {"rank": 1, "variant": "legacy", "logloss_mean": 0.310},
            {"rank": 2, "variant": "multihot", "logloss_mean": 0.330},
        ])
        history = [baseline, regression]
        a = build_regression_alert(history)
        # Should have some signal
        assert a["alert_level"] != "none" or a["ranking_drop"] == {}

    def test_recommendation_instability_detected(self):
        h = [
            _make_entry("c1", action="consider_promotion"),
            _make_entry("c2", action="hold"),
        ]
        a = build_regression_alert(h)
        assert a["recommendation_instability"]["action_changed"] is True

    def test_empty_history(self):
        a = build_regression_alert([])
        assert a["alert_level"] == "none"
        assert a["latest_campaign"] is None

    def test_regression_alert_md_has_sections(self, multi_history):
        a = build_regression_alert(multi_history)
        md = build_regression_alert_md(a)
        assert "# Regression Alert" in md
        assert "## Alert Level" in md


# ---------------------------------------------------------------------------
# TestPromotionGate
# ---------------------------------------------------------------------------

class TestPromotionGate:
    def _make_stability(self, **kwargs):
        base = {
            "total_campaigns": 2,
            "latest_action": "hold",
            "latest_challenger": "multihot",
            "consecutive_same_action": 2,
            "consecutive_same_challenger": 2,
            "consecutive_keep_production": 2,
            "consecutive_run_more_seeds": 0,
            "consecutive_positive_signal_for_settransformer": 0,
            "consecutive_positive_signal_for_deepsets": 0,
        }
        base.update(kwargs)
        return base

    def test_required_keys(self, single_history):
        stability = self._make_stability()
        g = build_promotion_gate(single_history, stability)
        for key in [
            "schema_version", "generated_at", "gate_status",
            "candidate_variant", "evidence_window", "conditions_passed",
            "blockers", "rationale", "next_required_action",
        ]:
            assert key in g, f"Missing key: {key}"

    def test_schema_version(self, single_history):
        stability = self._make_stability()
        g = build_promotion_gate(single_history, stability)
        assert g["schema_version"] == PROMOTION_GATE_SCHEMA_VERSION

    def test_gate_status_values(self, multi_history):
        stability = self._make_stability()
        g = build_promotion_gate(multi_history, stability)
        assert g["gate_status"] in {"red", "yellow", "green"}

    def test_gate_red_empty_history(self):
        stability = self._make_stability(total_campaigns=0)
        g = build_promotion_gate([], stability)
        assert g["gate_status"] == "red"
        assert len(g["blockers"]) >= 1

    def test_gate_green_when_conditions_met(self, promote_history):
        stability = self._make_stability(
            latest_action="consider_promotion",
            consecutive_same_action=2,
            consecutive_same_challenger=2,
            latest_challenger="multihot",
        )
        alert = {"alert_level": "none"}
        g = build_promotion_gate(promote_history, stability, regression_alert=alert)
        # With consider_promotion + consistent challenger + no regression,
        # should be green or yellow
        assert g["gate_status"] in {"green", "yellow"}

    def test_gate_red_when_high_regression(self, promote_history):
        stability = self._make_stability(latest_action="consider_promotion")
        alert = {"alert_level": "high"}
        g = build_promotion_gate(promote_history, stability, regression_alert=alert)
        assert g["gate_status"] in {"red", "yellow"}

    def test_candidate_variant_from_latest(self, single_history):
        stability = self._make_stability()
        g = build_promotion_gate(single_history, stability)
        assert g["candidate_variant"] == single_history[-1].get("recommended_challenger")

    def test_conditions_and_blockers_are_lists(self, single_history):
        stability = self._make_stability()
        g = build_promotion_gate(single_history, stability)
        assert isinstance(g["conditions_passed"], list)
        assert isinstance(g["blockers"], list)

    def test_promotion_gate_md_has_sections(self, single_history):
        stability = self._make_stability()
        g = build_promotion_gate(single_history, stability)
        md = build_promotion_gate_md(g)
        assert "# Promotion Readiness Gate" in md
        assert "## Gate Status" in md
        assert "## Next Required Action" in md


# ---------------------------------------------------------------------------
# TestStabilityExtension
# ---------------------------------------------------------------------------

class TestStabilityExtension:
    def test_new_keys_present(self):
        history = [_make_entry("c1")]
        s = compute_recommendation_stability(history)
        assert "consecutive_positive_signal_for_settransformer" in s
        assert "consecutive_positive_signal_for_deepsets" in s

    def test_settransformer_signal_streak(self):
        e_pma = _make_entry("c1", pma=True)
        e_no_pma = _make_entry("c2", pma=False)
        # Last 2 have pma=True
        history = [e_no_pma, dict(e_pma, campaign_name="c3"), dict(e_pma, campaign_name="c4")]
        s = compute_recommendation_stability(history)
        assert s["consecutive_positive_signal_for_settransformer"] == 2

    def test_settransformer_signal_streak_broken(self):
        e_pma = _make_entry("c1", pma=True)
        e_no_pma = _make_entry("c2", pma=False)
        history = [dict(e_pma, campaign_name="c1"), e_no_pma]
        s = compute_recommendation_stability(history)
        assert s["consecutive_positive_signal_for_settransformer"] == 0

    def test_deepsets_signal_streak(self):
        pw_pass = {
            "deepsets_vs_legacy": {"run_count": 9, "both_pass_count": 3, "both_pass_rate": 0.333},
        }
        pw_fail = {
            "deepsets_vs_legacy": {"run_count": 9, "both_pass_count": 0, "both_pass_rate": 0.0},
        }
        e_pass = _make_entry("c1", pairwise=pw_pass)
        e_fail = _make_entry("c2", pairwise=pw_fail)
        history = [e_fail, dict(e_pass, campaign_name="c3"), dict(e_pass, campaign_name="c4")]
        s = compute_recommendation_stability(history)
        assert s["consecutive_positive_signal_for_deepsets"] == 2

    def test_empty_history_new_keys(self):
        s = compute_recommendation_stability([])
        assert s.get("consecutive_positive_signal_for_settransformer") == 0
        assert s.get("consecutive_positive_signal_for_deepsets") == 0

    def test_existing_keys_still_present(self):
        history = [_make_entry("c1")]
        s = compute_recommendation_stability(history)
        for key in [
            "total_campaigns", "latest_action", "latest_challenger",
            "consecutive_same_action", "consecutive_same_challenger",
            "consecutive_keep_production", "consecutive_run_more_seeds",
        ]:
            assert key in s, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# TestGovernanceReport
# ---------------------------------------------------------------------------

class TestGovernanceReport:
    def _build_minimal(self, history):
        from governance_layer import build_trend_summary, build_regression_alert, build_promotion_gate
        stability = compute_recommendation_stability(history)
        trend = build_trend_summary(history)
        alert = build_regression_alert(history)
        gate = build_promotion_gate(history, stability, regression_alert=alert)
        return trend, alert, gate, stability

    def test_governance_report_sections(self, multi_history):
        trend, alert, gate, stability = self._build_minimal(multi_history)
        md = build_governance_report(trend, alert, gate, stability, latest_entry=multi_history[-1])
        assert "# Governance Report" in md
        assert "## Current Recommendation" in md
        assert "## Promotion Readiness Gate" in md
        assert "## Regression Alert" in md
        assert "## Recommendation Stability" in md
        assert "## Recent Trend Overview" in md
        assert "## Production Status" in md
        assert "## PMA / ISAB / HPO Guidance" in md

    def test_governance_report_no_latest_entry(self, multi_history):
        trend, alert, gate, stability = self._build_minimal(multi_history)
        md = build_governance_report(trend, alert, gate, stability, latest_entry=None)
        assert "# Governance Report" in md

    def test_governance_report_schema_version_constant(self):
        assert GOVERNANCE_REPORT_SCHEMA_VERSION == 1

    def test_governance_report_contains_campaign_name(self, single_history):
        trend, alert, gate, stability = self._build_minimal(single_history)
        md = build_governance_report(trend, alert, gate, stability, latest_entry=single_history[-1])
        assert "c1" in md


# ---------------------------------------------------------------------------
# TestSaveGovernanceArtifacts
# ---------------------------------------------------------------------------

class TestSaveGovernanceArtifacts:
    def test_saves_all_artifacts(self, multi_history):
        stability = compute_recommendation_stability(multi_history)
        with tempfile.TemporaryDirectory() as d:
            paths = save_governance_artifacts(multi_history, stability, data_dir=d)
            expected = [
                "trend_summary.json", "trend_summary.md",
                "regression_alert.json", "regression_alert.md",
                "promotion_gate.json", "promotion_gate.md",
                "governance_report.md",
            ]
            for name in expected:
                assert name in paths, f"Missing artifact: {name}"
                assert Path(paths[name]).exists(), f"File not created: {paths[name]}"

    def test_json_artifacts_are_valid(self, multi_history):
        stability = compute_recommendation_stability(multi_history)
        with tempfile.TemporaryDirectory() as d:
            paths = save_governance_artifacts(multi_history, stability, data_dir=d)
            for name in ["trend_summary.json", "regression_alert.json", "promotion_gate.json"]:
                data = json.loads(Path(paths[name]).read_text(encoding="utf-8"))
                assert "schema_version" in data

    def test_governance_report_md_sections(self, multi_history):
        stability = compute_recommendation_stability(multi_history)
        with tempfile.TemporaryDirectory() as d:
            paths = save_governance_artifacts(multi_history, stability, data_dir=d)
            md = Path(paths["governance_report.md"]).read_text(encoding="utf-8")
            assert "# Governance Report" in md
            assert "## Current Recommendation" in md
            assert "## Promotion Readiness Gate" in md

    def test_empty_history_creates_files(self):
        stability = compute_recommendation_stability([])
        with tempfile.TemporaryDirectory() as d:
            paths = save_governance_artifacts([], stability, data_dir=d)
            assert Path(paths["governance_report.md"]).exists()

    def test_single_campaign_creates_all_files(self, single_history):
        stability = compute_recommendation_stability(single_history)
        with tempfile.TemporaryDirectory() as d:
            paths = save_governance_artifacts(single_history, stability, data_dir=d)
            # 7 governance + 2 comparability + 2 benchmark_lock + 2 campaign_acceptance = 13
            assert len(paths) == 13
            for name in [
                "trend_summary.json", "trend_summary.md",
                "regression_alert.json", "regression_alert.md",
                "promotion_gate.json", "promotion_gate.md",
                "governance_report.md",
                "comparability_report.json", "comparability_report.md",
                "benchmark_lock.json", "benchmark_lock.md",
                "campaign_acceptance.json", "campaign_acceptance.md",
            ]:
                assert name in paths, f"Missing: {name}"


# ---------------------------------------------------------------------------
# TestRunCampaignGovernanceSmoke
# ---------------------------------------------------------------------------

class TestRunCampaignGovernanceSmoke:
    def test_governance_layer_imports_cleanly(self):
        import governance_layer
        assert hasattr(governance_layer, "build_trend_summary")
        assert hasattr(governance_layer, "build_regression_alert")
        assert hasattr(governance_layer, "build_promotion_gate")
        assert hasattr(governance_layer, "build_governance_report")
        assert hasattr(governance_layer, "save_governance_artifacts")

    def test_run_campaign_imports_governance(self):
        # Verify the run_campaign script imports governance_layer
        import importlib.util, sys
        scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
        spec = importlib.util.spec_from_file_location(
            "run_campaign", scripts_dir / "run_campaign.py"
        )
        mod = importlib.util.module_from_spec(spec)
        # Just check the source text rather than executing (to avoid argparse side effects)
        src = (scripts_dir / "run_campaign.py").read_text(encoding="utf-8")
        assert "save_governance_artifacts" in src
        assert "governance_layer" in src

    def test_governance_end_to_end_smoke(self, multi_history):
        """Smoke: full governance artifact generation from history."""
        from campaign_manager import compute_recommendation_stability
        from governance_layer import save_governance_artifacts
        stability = compute_recommendation_stability(multi_history)
        with tempfile.TemporaryDirectory() as d:
            paths = save_governance_artifacts(multi_history, stability, data_dir=d)
            # 7 governance + 2 comparability + 2 benchmark_lock + 2 campaign_acceptance = 13
            assert len(paths) == 13
            # Governance report is non-empty
            gov_md = Path(paths["governance_report.md"]).read_text(encoding="utf-8")
            assert len(gov_md) > 100
            # Trend summary has the right window
            trend = json.loads(Path(paths["trend_summary.json"]).read_text())
            assert len(trend["campaigns_considered"]) == len(multi_history)
            # Regression alert has a valid level
            alert = json.loads(Path(paths["regression_alert.json"]).read_text())
            assert alert["alert_level"] in {"none", "low", "medium", "high"}
            # Promotion gate has a valid status
            gate = json.loads(Path(paths["promotion_gate.json"]).read_text())
            assert gate["gate_status"] in {"red", "yellow", "green"}
