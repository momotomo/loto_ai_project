"""Tests for cross_loto_summary.py and scripts/run_cross_loto.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cross_loto_summary import (
    CROSS_LOTO_SUMMARY_SCHEMA_VERSION,
    RECOMMENDATION_SCHEMA_VERSION,
    build_cross_loto_summary,
    build_overall_variant_summary,
    build_pairwise_comparison_summary,
    build_promotion_recommendation_summary,
    build_recommendation,
    build_variant_ranking,
    get_cross_loto_summary_path,
    get_recommendation_path,
    save_json,
)


# ---------------------------------------------------------------------------
# Helpers: build minimal per-loto comparison summaries
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


# ---------------------------------------------------------------------------
# 1. build_overall_variant_summary
# ---------------------------------------------------------------------------

def test_overall_variant_summary_empty():
    result = build_overall_variant_summary({})
    assert result == {}


def test_overall_variant_summary_single_loto():
    per_loto = {
        "loto6": _make_comparison_summary("loto6", ["deepsets", "legacy"]),
    }
    result = build_overall_variant_summary(per_loto)
    assert "deepsets" in result
    assert "legacy" in result
    assert result["deepsets"]["loto_types_evaluated"] == ["loto6"]
    assert result["deepsets"]["logloss"]["mean"] == pytest.approx(0.300)
    assert result["deepsets"]["logloss"]["per_loto"]["loto6"] == pytest.approx(0.300)
    assert result["deepsets"]["promote_count_total"] == 0


def test_overall_variant_summary_multiple_lotos():
    per_loto = {
        "loto6": _make_comparison_summary("loto6", ["deepsets", "legacy"], logloss_base=0.300),
        "loto7": _make_comparison_summary("loto7", ["deepsets", "legacy"], logloss_base=0.320),
    }
    result = build_overall_variant_summary(per_loto)
    assert result["deepsets"]["logloss"]["mean"] == pytest.approx(0.310, abs=1e-6)
    assert set(result["deepsets"]["loto_types_evaluated"]) == {"loto6", "loto7"}


def test_overall_variant_summary_missing_variant_in_one_loto():
    per_loto = {
        "loto6": _make_comparison_summary("loto6", ["deepsets", "settransformer"]),
        "loto7": _make_comparison_summary("loto7", ["deepsets"]),
    }
    result = build_overall_variant_summary(per_loto)
    assert result["settransformer"]["loto_types_evaluated"] == ["loto6"]
    assert result["deepsets"]["loto_types_evaluated"] == sorted(["loto6", "loto7"])


def test_overall_variant_summary_promote_counts():
    per_loto = {
        "loto6": _make_comparison_summary("loto6", ["deepsets", "legacy"], promote=True),
        "loto7": _make_comparison_summary("loto7", ["deepsets", "legacy"], promote=False),
    }
    result = build_overall_variant_summary(per_loto)
    # deepsets promoted in loto6 (promote_count=1), held in loto7 (hold_count=1)
    assert result["deepsets"]["promote_count_total"] == 1
    assert result["deepsets"]["hold_count_total"] == 1


# ---------------------------------------------------------------------------
# 2. build_variant_ranking
# ---------------------------------------------------------------------------

def test_variant_ranking_by_logloss_order():
    per_loto = {
        "loto6": _make_comparison_summary("loto6", ["legacy", "deepsets", "settransformer"]),
    }
    # Give different logloss values by hand
    per_loto["loto6"]["variants"]["legacy"]["logloss"]["mean"] = 0.290
    per_loto["loto6"]["variants"]["deepsets"]["logloss"]["mean"] = 0.300
    per_loto["loto6"]["variants"]["settransformer"]["logloss"]["mean"] = 0.310

    overall = build_overall_variant_summary(per_loto)
    ranking = build_variant_ranking(overall, per_loto)

    by_logloss = ranking["by_logloss"]
    assert by_logloss[0]["variant"] == "legacy"
    assert by_logloss[0]["rank"] == 1
    assert by_logloss[1]["variant"] == "deepsets"
    assert by_logloss[2]["variant"] == "settransformer"


def test_variant_ranking_contains_required_keys():
    per_loto = {"loto6": _make_comparison_summary("loto6", ["deepsets", "legacy"])}
    overall = build_overall_variant_summary(per_loto)
    ranking = build_variant_ranking(overall, per_loto)
    assert "by_logloss" in ranking
    assert "by_brier" in ranking
    assert "by_ece" in ranking
    assert "promote_counts" in ranking
    assert "calibration_recommendations" in ranking


def test_variant_ranking_promote_rate():
    per_loto = {
        "loto6": _make_comparison_summary("loto6", ["deepsets", "legacy"], promote=True),
    }
    overall = build_overall_variant_summary(per_loto)
    ranking = build_variant_ranking(overall, per_loto)
    # deepsets: promote_count=1, hold_count=0 → rate=1.0
    assert ranking["promote_counts"]["deepsets"]["promote_rate"] == pytest.approx(1.0)
    assert ranking["promote_counts"]["deepsets"]["promote_count"] == 1


def test_variant_ranking_none_mean_sorted_last():
    per_loto = {
        "loto6": _make_comparison_summary("loto6", ["deepsets", "legacy"]),
    }
    # Remove logloss mean for legacy to simulate missing data
    per_loto["loto6"]["variants"]["legacy"]["logloss"]["mean"] = None
    overall = build_overall_variant_summary(per_loto)
    # Manually clear per_loto mean to simulate None overall
    overall["legacy"]["logloss"]["mean"] = None
    ranking = build_variant_ranking(overall, per_loto)
    assert ranking["by_logloss"][-1]["variant"] == "legacy"


# ---------------------------------------------------------------------------
# 3. build_pairwise_comparison_summary
# ---------------------------------------------------------------------------

def test_pairwise_comparison_summary_empty():
    result = build_pairwise_comparison_summary({})
    assert result == {}


def test_pairwise_comparison_summary_single_loto():
    per_loto = {"loto6": _make_comparison_summary("loto6", ["deepsets", "settransformer"])}
    result = build_pairwise_comparison_summary(per_loto)
    assert "settransformer_vs_deepsets" in result
    comp = result["settransformer_vs_deepsets"]
    assert "per_loto" in comp
    assert "overall" in comp
    assert "loto6" in comp["per_loto"]
    assert comp["overall"]["run_count"] == 1


def test_pairwise_comparison_summary_aggregates_across_lotos():
    per_loto = {
        "loto6": _make_comparison_summary("loto6", ["deepsets", "settransformer"], promote=True),
        "loto7": _make_comparison_summary("loto7", ["deepsets", "settransformer"], promote=False),
    }
    result = build_pairwise_comparison_summary(per_loto)
    overall = result["settransformer_vs_deepsets"]["overall"]
    assert overall["run_count"] == 2
    assert overall["ci_wins"] == 1    # only loto6 passes
    assert overall["both_pass_count"] == 1


# ---------------------------------------------------------------------------
# 4. build_promotion_recommendation_summary
# ---------------------------------------------------------------------------

def test_promotion_recommendation_summary_consistent_hold():
    per_loto = {
        "loto6": _make_comparison_summary("loto6", ["deepsets", "legacy"], promote=False),
        "loto7": _make_comparison_summary("loto7", ["deepsets", "legacy"], promote=False),
    }
    overall = build_overall_variant_summary(per_loto)
    result = build_promotion_recommendation_summary(overall, per_loto)
    assert result["deepsets"]["consistent_hold"] is True
    assert result["deepsets"]["consistent_promote"] is False
    assert result["deepsets"]["promoted_in"] == []
    assert set(result["deepsets"]["held_in"]) == {"loto6", "loto7"}


def test_promotion_recommendation_summary_consistent_promote():
    per_loto = {
        "loto6": _make_comparison_summary("loto6", ["deepsets", "legacy"], promote=True),
        "loto7": _make_comparison_summary("loto7", ["deepsets", "legacy"], promote=True),
    }
    overall = build_overall_variant_summary(per_loto)
    result = build_promotion_recommendation_summary(overall, per_loto)
    assert result["deepsets"]["consistent_promote"] is True
    assert set(result["deepsets"]["promoted_in"]) == {"loto6", "loto7"}


# ---------------------------------------------------------------------------
# 5. build_cross_loto_summary — required keys
# ---------------------------------------------------------------------------

def test_build_cross_loto_summary_required_keys():
    per_loto = {
        "loto6": _make_comparison_summary("loto6", ["deepsets", "legacy"]),
    }
    summary = build_cross_loto_summary(
        per_loto_summaries=per_loto,
        loto_types=["loto6"],
        preset="archcomp",
        seeds=[42],
    )
    assert summary["schema_version"] == CROSS_LOTO_SUMMARY_SCHEMA_VERSION
    assert "generated_at" in summary
    assert summary["loto_types"] == ["loto6"]
    assert summary["preset"] == "archcomp"
    assert summary["seeds"] == [42]
    assert "run_count_per_loto" in summary
    assert "overall_summary" in summary
    assert "per_loto_summary" in summary
    assert "variant_ranking" in summary
    assert "pairwise_comparison_summary" in summary
    assert "promotion_recommendation_summary" in summary


def test_build_cross_loto_summary_multiple_lotos():
    per_loto = {
        "loto6": _make_comparison_summary("loto6", ["deepsets", "legacy"], seeds=[42, 123]),
        "loto7": _make_comparison_summary("loto7", ["deepsets", "legacy"], seeds=[42, 123]),
    }
    summary = build_cross_loto_summary(
        per_loto_summaries=per_loto,
        loto_types=["loto6", "loto7"],
        preset="archcomp",
        seeds=[42, 123],
    )
    assert set(summary["loto_types"]) == {"loto6", "loto7"}
    assert "deepsets" in summary["overall_summary"]["variants"]
    assert "loto6" in summary["per_loto_summary"]
    assert "loto7" in summary["per_loto_summary"]


# ---------------------------------------------------------------------------
# 6. build_recommendation — required keys and logic
# ---------------------------------------------------------------------------

def test_build_recommendation_required_keys():
    per_loto = {
        "loto6": _make_comparison_summary("loto6", ["deepsets", "legacy"]),
    }
    cross = build_cross_loto_summary(
        per_loto_summaries=per_loto,
        loto_types=["loto6"],
        preset="archcomp",
        seeds=[42],
    )
    rec = build_recommendation(cross)
    assert rec["schema_version"] == RECOMMENDATION_SCHEMA_VERSION
    assert "generated_at" in rec
    assert rec["based_on"] == "cross_loto_summary"
    assert "recommended_next_action" in rec
    assert "recommended_challenger" in rec
    assert "keep_production_as_is" in rec
    assert "evidence_summary" in rec
    assert "blockers_to_promotion" in rec
    assert isinstance(rec["blockers_to_promotion"], list)
    assert "whether_to_try_pma_or_isab_next" in rec
    assert isinstance(rec["whether_to_try_pma_or_isab_next"], bool)
    assert "next_experiment_recommendations" in rec
    assert isinstance(rec["next_experiment_recommendations"], list)


def test_build_recommendation_hold_when_no_promotions():
    per_loto = {
        "loto6": _make_comparison_summary("loto6", ["deepsets", "legacy"], promote=False),
    }
    cross = build_cross_loto_summary(
        per_loto_summaries=per_loto,
        loto_types=["loto6"],
        preset="archcomp",
        seeds=[42],
    )
    rec = build_recommendation(cross)
    assert rec["keep_production_as_is"] is True
    assert rec["recommended_next_action"] in ("hold", "run_more_seeds")


def test_build_recommendation_consider_promotion_when_consistent_promote():
    per_loto = {
        "loto6": _make_comparison_summary("loto6", ["deepsets", "legacy"], promote=True),
        "loto7": _make_comparison_summary("loto7", ["deepsets", "legacy"], promote=True),
    }
    cross = build_cross_loto_summary(
        per_loto_summaries=per_loto,
        loto_types=["loto6", "loto7"],
        preset="archcomp",
        seeds=[42],
    )
    rec = build_recommendation(cross)
    assert rec["recommended_next_action"] == "consider_promotion"
    assert rec["keep_production_as_is"] is False


def test_build_recommendation_pma_signal_when_settransformer_beats_deepsets():
    per_loto = {
        "loto6": _make_comparison_summary("loto6", ["deepsets", "settransformer"], promote=True, run_count=3),
        "loto7": _make_comparison_summary("loto7", ["deepsets", "settransformer"], promote=True, run_count=3),
    }
    cross = build_cross_loto_summary(
        per_loto_summaries=per_loto,
        loto_types=["loto6", "loto7"],
        preset="archcomp",
        seeds=[42, 123, 456],
    )
    rec = build_recommendation(cross)
    # promote=True means settransformer_vs_deepsets both_pass_count = run_count = 3
    # overall both_pass_count / run_count = 6/6 = 1.0 >= threshold
    assert rec["whether_to_try_pma_or_isab_next"] is True


def test_build_recommendation_no_pma_signal_when_no_wins():
    per_loto = {
        "loto6": _make_comparison_summary("loto6", ["deepsets", "settransformer"], promote=False),
    }
    cross = build_cross_loto_summary(
        per_loto_summaries=per_loto,
        loto_types=["loto6"],
        preset="archcomp",
        seeds=[42],
    )
    rec = build_recommendation(cross)
    assert rec["whether_to_try_pma_or_isab_next"] is False


# ---------------------------------------------------------------------------
# 7. Persistence helpers
# ---------------------------------------------------------------------------

def test_save_and_load_cross_loto_summary(tmp_path):
    per_loto = {
        "loto6": _make_comparison_summary("loto6", ["deepsets", "legacy"]),
    }
    summary = build_cross_loto_summary(
        per_loto_summaries=per_loto,
        loto_types=["loto6"],
        preset="archcomp",
        seeds=[42],
    )
    output_path = tmp_path / "data" / "cross_loto_summary.json"
    save_json(summary, output_path)
    assert output_path.exists()
    with open(output_path, "r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    assert loaded["schema_version"] == CROSS_LOTO_SUMMARY_SCHEMA_VERSION
    assert loaded["loto_types"] == ["loto6"]


def test_save_and_load_recommendation(tmp_path):
    per_loto = {"loto6": _make_comparison_summary("loto6", ["deepsets", "legacy"])}
    cross = build_cross_loto_summary(
        per_loto_summaries=per_loto,
        loto_types=["loto6"],
        preset="archcomp",
        seeds=[42],
    )
    rec = build_recommendation(cross)
    output_path = tmp_path / "data" / "recommendation.json"
    save_json(rec, output_path)
    assert output_path.exists()
    with open(output_path, "r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    assert loaded["based_on"] == "cross_loto_summary"
    assert loaded["schema_version"] == RECOMMENDATION_SCHEMA_VERSION


def test_get_cross_loto_summary_path():
    path = get_cross_loto_summary_path()
    assert "cross_loto_summary.json" in str(path)

    path_custom = get_cross_loto_summary_path(data_dir="/tmp/mydata")
    assert "cross_loto_summary.json" in str(path_custom)


def test_get_recommendation_path():
    path = get_recommendation_path()
    assert "recommendation.json" in str(path)


# ---------------------------------------------------------------------------
# 8. CLI smoke: scripts/run_cross_loto.py parse helpers
# ---------------------------------------------------------------------------

def test_parse_loto_types_valid():
    from scripts.run_cross_loto import parse_loto_types
    result = parse_loto_types("loto6,loto7")
    assert result == ["loto6", "loto7"]


def test_parse_loto_types_invalid_raises():
    from scripts.run_cross_loto import parse_loto_types
    with pytest.raises(SystemExit):
        parse_loto_types("loto99")


def test_parse_loto_types_empty_raises():
    from scripts.run_cross_loto import parse_loto_types
    with pytest.raises(SystemExit):
        parse_loto_types("")


def test_parse_seeds_valid():
    from scripts.run_cross_loto import parse_seeds
    assert parse_seeds("42,123") == [42, 123]


def test_parse_seeds_invalid_raises():
    from scripts.run_cross_loto import parse_seeds
    with pytest.raises(SystemExit):
        parse_seeds("42,xyz")
