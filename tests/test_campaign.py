"""tests/test_campaign.py

Tests for campaign_profiles.py and campaign_manager.py.

Covers:
- Campaign profile resolution (get_profile, resolve_profile, overrides)
- Campaign entry building (required keys, values)
- Recommendation stability computation
- History append / dedup
- Diff report generation (required Markdown sections)
- Campaign history CSV columns
- CLI smoke tests (--list_profiles, --help, --campaign_name)
"""

from __future__ import annotations

import csv
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def _make_per_loto(loto_type: str = "loto6", promote_count: int = 0) -> dict:
    return {
        "schema_version": 1,
        "loto_type": loto_type,
        "preset": "archcomp",
        "seeds": [42, 123],
        "run_count": 2,
        "alpha": 0.05,
        "variants": {
            "legacy": {
                "run_count": 2,
                "logloss": {"mean": 0.30, "std": 0.01, "values": [0.30, 0.30]},
                "brier": {"mean": 0.04, "std": 0.0, "values": [0.04, 0.04]},
                "ece": {"mean": 0.01, "std": 0.0, "values": [0.01, 0.01]},
                "calibration_recommendations": {"none": 2},
                "promote_count": 0,
                "hold_count": 2,
            },
            "multihot": {
                "run_count": 2,
                "logloss": {"mean": 0.28, "std": 0.01, "values": [0.28, 0.28]},
                "brier": {"mean": 0.03, "std": 0.0, "values": [0.03, 0.03]},
                "ece": {"mean": 0.01, "std": 0.0, "values": [0.01, 0.01]},
                "calibration_recommendations": {"none": 2},
                "promote_count": promote_count,
                "hold_count": 2 - promote_count,
            },
        },
        "pairwise_comparisons": {
            "multihot_vs_legacy": {
                "run_count": 2,
                "ci_wins": 1,
                "permutation_wins": 1,
                "both_pass_count": 1,
            },
            "settransformer_vs_deepsets": {
                "run_count": 2,
                "ci_wins": 0,
                "permutation_wins": 0,
                "both_pass_count": 0,
            },
        },
    }


def _make_cross_loto_and_rec(loto_types=("loto6",), promote_count=0):
    from cross_loto_summary import build_cross_loto_summary, build_recommendation

    per_loto = {lt: _make_per_loto(lt, promote_count) for lt in loto_types}
    cs = build_cross_loto_summary(per_loto, list(loto_types), "archcomp", [42, 123])
    rec = build_recommendation(cs)
    return cs, rec


# ---------------------------------------------------------------------------
# campaign_profiles
# ---------------------------------------------------------------------------


class TestCampaignProfiles:
    def test_valid_profile_names_exists(self):
        from campaign_profiles import VALID_PROFILE_NAMES

        assert len(VALID_PROFILE_NAMES) >= 3

    def test_archcomp_lite_profile_exists(self):
        from campaign_profiles import get_profile

        p = get_profile("archcomp_lite")
        assert p["preset"] == "fast"
        assert isinstance(p["seeds"], list)
        assert len(p["seeds"]) >= 1

    def test_archcomp_profile_exists(self):
        from campaign_profiles import get_profile

        p = get_profile("archcomp")
        assert p["preset"] == "archcomp"
        assert len(p["seeds"]) >= 2
        assert len(p["loto_types"]) >= 1

    def test_archcomp_full_profile_exists(self):
        from campaign_profiles import get_profile

        p = get_profile("archcomp_full")
        assert p["preset"] == "default"
        assert len(p["seeds"]) >= 4

    def test_profile_required_keys(self):
        from campaign_profiles import CAMPAIGN_PROFILES

        required_keys = {
            "preset",
            "seeds",
            "loto_types",
            "evaluation_model_variants",
            "saved_calibration_method",
            "evaluation_calibration_methods",
            "description",
        }
        for name, profile in CAMPAIGN_PROFILES.items():
            missing = required_keys - set(profile.keys())
            assert not missing, f"Profile {name!r} missing keys: {missing}"

    def test_get_profile_unknown_raises(self):
        from campaign_profiles import get_profile

        with pytest.raises(ValueError, match="Unknown campaign profile"):
            get_profile("nonexistent_profile")

    def test_resolve_profile_no_overrides(self):
        from campaign_profiles import resolve_profile

        p = resolve_profile("archcomp")
        assert p["preset"] == "archcomp"

    def test_resolve_profile_with_seed_override(self):
        from campaign_profiles import resolve_profile

        p = resolve_profile("archcomp", {"seeds": [42]})
        assert p["seeds"] == [42]
        # Other keys unchanged
        assert p["preset"] == "archcomp"

    def test_resolve_profile_does_not_mutate_original(self):
        from campaign_profiles import CAMPAIGN_PROFILES, resolve_profile

        original_seeds = list(CAMPAIGN_PROFILES["archcomp"]["seeds"])
        _ = resolve_profile("archcomp", {"seeds": [999]})
        assert CAMPAIGN_PROFILES["archcomp"]["seeds"] == original_seeds

    def test_archcomp_full_has_more_seeds_than_archcomp(self):
        from campaign_profiles import get_profile

        lite = get_profile("archcomp_lite")
        standard = get_profile("archcomp")
        full = get_profile("archcomp_full")
        assert len(full["seeds"]) >= len(standard["seeds"]) >= len(lite["seeds"])

    def test_profile_loto_types_are_strings(self):
        from campaign_profiles import CAMPAIGN_PROFILES

        for name, profile in CAMPAIGN_PROFILES.items():
            for lt in profile["loto_types"]:
                assert isinstance(lt, str), f"Profile {name!r}: loto_type must be str"

    def test_profile_seeds_are_ints(self):
        from campaign_profiles import CAMPAIGN_PROFILES

        for name, profile in CAMPAIGN_PROFILES.items():
            for s in profile["seeds"]:
                assert isinstance(s, int), f"Profile {name!r}: seed must be int"


# ---------------------------------------------------------------------------
# campaign_manager — build_campaign_entry
# ---------------------------------------------------------------------------


class TestBuildCampaignEntry:
    def test_required_keys_present(self):
        from campaign_manager import build_campaign_entry

        cs, rec = _make_cross_loto_and_rec()
        entry = build_campaign_entry("test_campaign", "archcomp", cs, rec)

        required = {
            "campaign_name",
            "profile_name",
            "generated_at",
            "started_at",
            "finished_at",
            "campaign_dir",
            "loto_types",
            "preset",
            "seeds",
            "recommended_next_action",
            "recommended_challenger",
            "keep_production_as_is",
            "whether_to_try_pma_or_isab_next",
            "best_variant_by_logloss",
            "consistent_promote_variants",
            "blockers_count",
            "variant_ranking_summary",
            "key_pairwise_signals",
        }
        missing = required - set(entry.keys())
        assert not missing, f"Missing keys in campaign entry: {missing}"

    def test_campaign_name_is_preserved(self):
        from campaign_manager import build_campaign_entry

        cs, rec = _make_cross_loto_and_rec()
        entry = build_campaign_entry("my_campaign", "archcomp", cs, rec)
        assert entry["campaign_name"] == "my_campaign"

    def test_profile_name_is_preserved(self):
        from campaign_manager import build_campaign_entry

        cs, rec = _make_cross_loto_and_rec()
        entry = build_campaign_entry("x", "archcomp_full", cs, rec)
        assert entry["profile_name"] == "archcomp_full"

    def test_variant_ranking_summary_is_list(self):
        from campaign_manager import build_campaign_entry

        cs, rec = _make_cross_loto_and_rec()
        entry = build_campaign_entry("x", "archcomp", cs, rec)
        assert isinstance(entry["variant_ranking_summary"], list)
        assert len(entry["variant_ranking_summary"]) >= 1

    def test_variant_ranking_summary_has_required_keys(self):
        from campaign_manager import build_campaign_entry

        cs, rec = _make_cross_loto_and_rec()
        entry = build_campaign_entry("x", "archcomp", cs, rec)
        for item in entry["variant_ranking_summary"]:
            assert "rank" in item
            assert "variant" in item
            assert "logloss_mean" in item

    def test_key_pairwise_signals_is_dict(self):
        from campaign_manager import build_campaign_entry

        cs, rec = _make_cross_loto_and_rec()
        entry = build_campaign_entry("x", "archcomp", cs, rec)
        assert isinstance(entry["key_pairwise_signals"], dict)

    def test_pairwise_signals_have_both_pass_rate(self):
        from campaign_manager import build_campaign_entry

        cs, rec = _make_cross_loto_and_rec()
        entry = build_campaign_entry("x", "archcomp", cs, rec)
        for key, sig in entry["key_pairwise_signals"].items():
            assert "both_pass_rate" in sig, f"key {key!r} missing both_pass_rate"
            assert 0.0 <= sig["both_pass_rate"] <= 1.0

    def test_blockers_count_is_int(self):
        from campaign_manager import build_campaign_entry

        cs, rec = _make_cross_loto_and_rec()
        entry = build_campaign_entry("x", "archcomp", cs, rec)
        assert isinstance(entry["blockers_count"], int)

    def test_optional_fields_accepted(self):
        from campaign_manager import build_campaign_entry

        cs, rec = _make_cross_loto_and_rec()
        entry = build_campaign_entry(
            "x",
            "archcomp",
            cs,
            rec,
            started_at="2026-01-01T00:00:00Z",
            finished_at="2026-01-01T01:00:00Z",
            campaign_dir="/tmp/campaigns/x",
        )
        assert entry["started_at"] == "2026-01-01T00:00:00Z"
        assert entry["finished_at"] == "2026-01-01T01:00:00Z"
        assert entry["campaign_dir"] == "/tmp/campaigns/x"


# ---------------------------------------------------------------------------
# campaign_manager — recommendation_stability
# ---------------------------------------------------------------------------


class TestRecommendationStability:
    def _make_entry(self, action: str, challenger: str | None, keep: bool) -> dict:
        cs, rec = _make_cross_loto_and_rec()
        from campaign_manager import build_campaign_entry

        entry = build_campaign_entry("x", "archcomp", cs, rec)
        entry["recommended_next_action"] = action
        entry["recommended_challenger"] = challenger
        entry["keep_production_as_is"] = keep
        return entry

    def test_empty_history_returns_zero_counts(self):
        from campaign_manager import compute_recommendation_stability

        stability = compute_recommendation_stability([])
        assert stability["total_campaigns"] == 0
        assert stability["consecutive_same_action"] == 0

    def test_single_entry(self):
        from campaign_manager import compute_recommendation_stability

        e = self._make_entry("hold", "multihot", True)
        stability = compute_recommendation_stability([e])
        assert stability["total_campaigns"] == 1
        assert stability["consecutive_same_action"] == 1
        assert stability["latest_action"] == "hold"

    def test_consecutive_same_action_count(self):
        from campaign_manager import compute_recommendation_stability

        entries = [
            self._make_entry("hold", "multihot", True),
            self._make_entry("hold", "multihot", True),
            self._make_entry("hold", "multihot", True),
        ]
        stability = compute_recommendation_stability(entries)
        assert stability["consecutive_same_action"] == 3

    def test_consecutive_breaks_on_different_action(self):
        from campaign_manager import compute_recommendation_stability

        entries = [
            self._make_entry("run_more_seeds", "multihot", True),
            self._make_entry("run_more_seeds", "multihot", True),
            self._make_entry("hold", "multihot", True),
        ]
        stability = compute_recommendation_stability(entries)
        assert stability["consecutive_same_action"] == 1
        assert stability["latest_action"] == "hold"

    def test_consecutive_run_more_seeds_count(self):
        from campaign_manager import compute_recommendation_stability

        entries = [
            self._make_entry("run_more_seeds", "multihot", True),
            self._make_entry("run_more_seeds", "multihot", True),
        ]
        stability = compute_recommendation_stability(entries)
        assert stability["consecutive_run_more_seeds"] == 2

    def test_consecutive_keep_production(self):
        from campaign_manager import compute_recommendation_stability

        entries = [
            self._make_entry("hold", "multihot", True),
            self._make_entry("hold", "multihot", True),
            self._make_entry("consider_promotion", None, False),
        ]
        stability = compute_recommendation_stability(entries)
        assert stability["consecutive_keep_production"] == 0

    def test_consecutive_challenger_counts(self):
        from campaign_manager import compute_recommendation_stability

        entries = [
            self._make_entry("hold", "multihot", True),
            self._make_entry("hold", "deepsets", True),
            self._make_entry("hold", "deepsets", True),
        ]
        stability = compute_recommendation_stability(entries)
        assert stability["consecutive_same_challenger"] == 2
        assert stability["latest_challenger"] == "deepsets"

    def test_stability_required_keys(self):
        from campaign_manager import compute_recommendation_stability

        stability = compute_recommendation_stability([self._make_entry("hold", "multihot", True)])
        required = {
            "total_campaigns",
            "latest_action",
            "latest_challenger",
            "consecutive_same_action",
            "consecutive_same_challenger",
            "consecutive_keep_production",
            "consecutive_run_more_seeds",
        }
        missing = required - set(stability.keys())
        assert not missing


# ---------------------------------------------------------------------------
# campaign_manager — history append / dedup
# ---------------------------------------------------------------------------


class TestHistoryAppend:
    def _entry(self, name: str) -> dict:
        cs, rec = _make_cross_loto_and_rec()
        from campaign_manager import build_campaign_entry

        return build_campaign_entry(name, "archcomp", cs, rec)

    def test_append_to_empty_history(self):
        from campaign_manager import append_campaign_to_history

        history = append_campaign_to_history([], self._entry("campaign_1"))
        assert len(history) == 1
        assert history[0]["campaign_name"] == "campaign_1"

    def test_append_multiple(self):
        from campaign_manager import append_campaign_to_history

        history = []
        for i in range(3):
            history = append_campaign_to_history(history, self._entry(f"campaign_{i}"))
        assert len(history) == 3

    def test_duplicate_campaign_name_replaces(self):
        from campaign_manager import append_campaign_to_history

        history = [self._entry("c1"), self._entry("c2")]
        updated = self._entry("c1")
        updated["recommended_next_action"] = "run_more_seeds"
        history = append_campaign_to_history(history, updated)
        assert len(history) == 2
        names = [e["campaign_name"] for e in history]
        assert names.count("c1") == 1
        # Updated entry is at the end
        assert history[-1]["campaign_name"] == "c1"
        assert history[-1]["recommended_next_action"] == "run_more_seeds"

    def test_original_history_not_mutated(self):
        from campaign_manager import append_campaign_to_history

        original = [self._entry("c1")]
        _ = append_campaign_to_history(original, self._entry("c2"))
        assert len(original) == 1


# ---------------------------------------------------------------------------
# campaign_manager — campaign history CSV
# ---------------------------------------------------------------------------


class TestCampaignHistoryCsv:
    def _entry(self, name: str, action: str = "hold") -> dict:
        cs, rec = _make_cross_loto_and_rec()
        from campaign_manager import build_campaign_entry

        entry = build_campaign_entry(name, "archcomp", cs, rec)
        entry["recommended_next_action"] = action
        return entry

    def test_required_columns_present(self):
        from campaign_manager import build_campaign_history_csv

        history = [self._entry("c1")]
        csv_text = build_campaign_history_csv(history)
        reader = csv.DictReader(io.StringIO(csv_text))
        cols = reader.fieldnames or []
        required = [
            "campaign_name",
            "profile_name",
            "generated_at",
            "loto_types",
            "preset",
            "seeds",
            "recommended_next_action",
            "recommended_challenger",
            "keep_production_as_is",
            "best_variant_by_logloss",
            "consistent_promote_variants",
            "blockers_count",
        ]
        for col in required:
            assert col in cols, f"Column {col!r} missing from campaign history CSV"

    def test_row_count_matches_history(self):
        from campaign_manager import build_campaign_history_csv

        history = [self._entry(f"c{i}") for i in range(3)]
        csv_text = build_campaign_history_csv(history)
        reader = csv.DictReader(io.StringIO(csv_text))
        rows = list(reader)
        assert len(rows) == 3

    def test_campaign_names_in_csv(self):
        from campaign_manager import build_campaign_history_csv

        history = [self._entry("alpha"), self._entry("beta")]
        csv_text = build_campaign_history_csv(history)
        assert "alpha" in csv_text
        assert "beta" in csv_text


# ---------------------------------------------------------------------------
# campaign_manager — diff report
# ---------------------------------------------------------------------------


class TestDiffReport:
    def _entry(self, name: str, action: str = "hold", challenger: str = "multihot") -> dict:
        cs, rec = _make_cross_loto_and_rec()
        from campaign_manager import build_campaign_entry

        entry = build_campaign_entry(name, "archcomp", cs, rec)
        entry["recommended_next_action"] = action
        entry["recommended_challenger"] = challenger
        return entry

    def test_required_sections_present(self):
        from campaign_manager import build_diff_report

        prev = self._entry("c1")
        curr = self._entry("c2")
        md = build_diff_report(prev, curr)
        for section in [
            "# Campaign Diff Report",
            "## Campaign Overview",
            "## Recommendation Change",
            "## Variant Ranking Change",
            "## Pairwise Signal Change",
            "## PMA / ISAB Next Steps",
        ]:
            assert section in md, f"Missing section: {section!r}"

    def test_campaign_names_appear_in_report(self):
        from campaign_manager import build_diff_report

        prev = self._entry("prev_campaign")
        curr = self._entry("curr_campaign")
        md = build_diff_report(prev, curr)
        assert "prev_campaign" in md
        assert "curr_campaign" in md

    def test_changed_action_shows_changed_marker(self):
        from campaign_manager import build_diff_report

        prev = self._entry("c1", action="hold")
        curr = self._entry("c2", action="run_more_seeds")
        md = build_diff_report(prev, curr)
        assert "CHANGED" in md

    def test_unchanged_action_shows_unchanged_marker(self):
        from campaign_manager import build_diff_report

        prev = self._entry("c1", action="hold")
        curr = self._entry("c2", action="hold")
        md = build_diff_report(prev, curr)
        assert "unchanged" in md

    def test_stability_section_when_provided(self):
        from campaign_manager import build_diff_report, compute_recommendation_stability

        prev = self._entry("c1")
        curr = self._entry("c2")
        stability = compute_recommendation_stability([prev, curr])
        md = build_diff_report(prev, curr, stability=stability)
        assert "## Recommendation Stability" in md

    def test_no_stability_section_when_omitted(self):
        from campaign_manager import build_diff_report

        prev = self._entry("c1")
        curr = self._entry("c2")
        md = build_diff_report(prev, curr, stability=None)
        assert "## Recommendation Stability" not in md

    def test_pma_signal_message_when_newly_appears(self):
        from campaign_manager import build_diff_report

        prev = self._entry("c1")
        curr = self._entry("c2")
        prev["whether_to_try_pma_or_isab_next"] = False
        curr["whether_to_try_pma_or_isab_next"] = True
        md = build_diff_report(prev, curr)
        assert "newly appeared" in md

    def test_variant_ranking_table_has_variants(self):
        from campaign_manager import build_diff_report

        prev = self._entry("c1")
        curr = self._entry("c2")
        md = build_diff_report(prev, curr)
        # Both variants from _make_per_loto appear
        assert "legacy" in md
        assert "multihot" in md


# ---------------------------------------------------------------------------
# campaign_manager — save_campaign_artifacts
# ---------------------------------------------------------------------------


class TestSaveCampaignArtifacts:
    def _entry(self, name: str) -> dict:
        cs, rec = _make_cross_loto_and_rec()
        from campaign_manager import build_campaign_entry

        return build_campaign_entry(name, "archcomp", cs, rec)

    def test_saves_history_json(self):
        from campaign_manager import save_campaign_artifacts

        history = [self._entry("c1")]
        with tempfile.TemporaryDirectory() as d:
            paths = save_campaign_artifacts(history, data_dir=d)
            assert "campaign_history.json" in paths
            assert Path(paths["campaign_history.json"]).exists()

    def test_saves_history_csv(self):
        from campaign_manager import save_campaign_artifacts

        history = [self._entry("c1")]
        with tempfile.TemporaryDirectory() as d:
            paths = save_campaign_artifacts(history, data_dir=d)
            assert "campaign_history.csv" in paths
            assert Path(paths["campaign_history.csv"]).exists()

    def test_saves_diff_report_when_two_entries(self):
        from campaign_manager import save_campaign_artifacts

        history = [self._entry("c1"), self._entry("c2")]
        with tempfile.TemporaryDirectory() as d:
            paths = save_campaign_artifacts(history, data_dir=d)
            assert "campaign_diff_report.md" in paths
            assert Path(paths["campaign_diff_report.md"]).exists()

    def test_no_diff_report_when_single_entry(self):
        from campaign_manager import save_campaign_artifacts

        history = [self._entry("c1")]
        with tempfile.TemporaryDirectory() as d:
            paths = save_campaign_artifacts(history, data_dir=d)
            assert "campaign_diff_report.md" not in paths

    def test_history_json_has_stability(self):
        from campaign_manager import save_campaign_artifacts

        history = [self._entry("c1"), self._entry("c2")]
        with tempfile.TemporaryDirectory() as d:
            paths = save_campaign_artifacts(history, data_dir=d)
            with open(paths["campaign_history.json"], encoding="utf-8") as f:
                payload = json.load(f)
            assert "recommendation_stability" in payload

    def test_history_json_has_campaigns_list(self):
        from campaign_manager import save_campaign_artifacts

        history = [self._entry("c1"), self._entry("c2")]
        with tempfile.TemporaryDirectory() as d:
            paths = save_campaign_artifacts(history, data_dir=d)
            with open(paths["campaign_history.json"], encoding="utf-8") as f:
                payload = json.load(f)
            assert isinstance(payload.get("campaigns"), list)
            assert len(payload["campaigns"]) == 2

    def test_load_save_roundtrip(self):
        from campaign_manager import (
            load_campaign_history,
            save_campaign_artifacts,
        )

        history = [self._entry("c1"), self._entry("c2")]
        with tempfile.TemporaryDirectory() as d:
            save_campaign_artifacts(history, data_dir=d)
            loaded = load_campaign_history(d)
        assert len(loaded) == 2
        assert loaded[0]["campaign_name"] == "c1"
        assert loaded[1]["campaign_name"] == "c2"


# ---------------------------------------------------------------------------
# CLI smoke tests
# ---------------------------------------------------------------------------


class TestRunCampaignCli:
    SCRIPT = str(REPO_ROOT / "scripts" / "run_campaign.py")

    def test_help_exits_zero(self):
        result = subprocess.run(
            [sys.executable, self.SCRIPT, "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    def test_help_mentions_list_profiles(self):
        result = subprocess.run(
            [sys.executable, self.SCRIPT, "--help"],
            capture_output=True,
            text=True,
        )
        assert "--list_profiles" in result.stdout

    def test_help_mentions_profile(self):
        result = subprocess.run(
            [sys.executable, self.SCRIPT, "--help"],
            capture_output=True,
            text=True,
        )
        assert "--profile" in result.stdout

    def test_list_profiles_exits_zero(self):
        result = subprocess.run(
            [sys.executable, self.SCRIPT, "--list_profiles"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    def test_list_profiles_shows_profile_names(self):
        result = subprocess.run(
            [sys.executable, self.SCRIPT, "--list_profiles"],
            capture_output=True,
            text=True,
        )
        assert "archcomp_lite" in result.stdout
        assert "archcomp_full" in result.stdout

    def test_missing_campaign_name_exits_nonzero(self):
        result = subprocess.run(
            [sys.executable, self.SCRIPT, "--profile", "archcomp"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

    def test_invalid_profile_exits_nonzero(self):
        result = subprocess.run(
            [sys.executable, self.SCRIPT, "--campaign_name", "x", "--profile", "nonexistent"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
