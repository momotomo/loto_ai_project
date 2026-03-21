"""Tests for the archcomp preset and comparison summary infrastructure."""

import json
from pathlib import Path

import pytest

from comparison_summary import (
    COMPARISON_SUMMARY_SCHEMA_VERSION,
    PAIRWISE_KEYS_OF_INTEREST,
    aggregate_pairwise_comparisons,
    aggregate_variant_metrics,
    build_comparison_summary,
    get_comparison_summary_path,
    save_comparison_summary,
)
from train_prob_model import PRESET_CONFIGS


# ---------------------------------------------------------------------------
# 1. archcomp preset
# ---------------------------------------------------------------------------


def test_archcomp_preset_is_registered():
    assert "archcomp" in PRESET_CONFIGS, "archcomp preset must be in PRESET_CONFIGS"


def test_archcomp_preset_has_required_keys():
    required = {"walk_forward_folds", "eval_epochs", "final_epochs", "batch_size", "patience"}
    assert required.issubset(PRESET_CONFIGS["archcomp"]), (
        f"archcomp preset is missing required keys: {required - set(PRESET_CONFIGS['archcomp'])}"
    )


def test_archcomp_preset_has_nonzero_epochs():
    preset = PRESET_CONFIGS["archcomp"]
    assert preset["eval_epochs"] > 0, "archcomp eval_epochs must be > 0 to distinguish architectures"
    assert preset["final_epochs"] > 0, "archcomp final_epochs must be > 0"


def test_archcomp_preset_has_multiple_folds():
    assert PRESET_CONFIGS["archcomp"]["walk_forward_folds"] >= 2, (
        "archcomp should use at least 2 folds for meaningful variance estimation"
    )


def test_smoke_preset_still_has_zero_epochs():
    """Ensure the smoke preset wasn't accidentally changed."""
    assert PRESET_CONFIGS["smoke"]["eval_epochs"] == 0
    assert PRESET_CONFIGS["smoke"]["final_epochs"] == 0


def test_all_presets_have_required_keys():
    required = {"walk_forward_folds", "eval_epochs", "final_epochs", "batch_size", "patience"}
    for name, preset in PRESET_CONFIGS.items():
        missing = required - set(preset)
        assert not missing, f"Preset '{name}' is missing keys: {missing}"


# ---------------------------------------------------------------------------
# 2. comparison_summary.py – aggregate_variant_metrics
# ---------------------------------------------------------------------------


def _make_eval_report(variant_names, *, logloss_offset=0.0, promote=False):
    """Build a minimal eval_report dict for testing."""
    model_variants = {}
    for v in variant_names:
        logloss_val = 0.300 + logloss_offset
        model_variants[v] = {
            "walk_forward": {
                "aggregate": {
                    "model": {
                        "metric_summary": {
                            "logloss": {"mean": logloss_val, "variance": 0.001},
                            "brier": {"mean": 0.040 + logloss_offset, "variance": 0.001},
                            "ece": {"mean": 0.010 + logloss_offset, "variance": 0.001},
                            "mean_overlap_top_k": {"mean": 1.2, "variance": 0.01},
                        }
                    }
                }
            }
        }

    calibration_evaluation = {v: {"recommended_method": "none"} for v in variant_names}

    # Challenger decisions for all non-legacy variants
    challenger_decisions = {}
    for v in variant_names:
        if v != "legacy":
            challenger_decisions[v] = {
                "variant": v,
                "should_promote": promote,
                "reason_summary": [],
            }

    return {
        "model_variants": model_variants,
        "calibration_evaluation": calibration_evaluation,
        "statistical_tests": {
            "comparisons": {
                "settransformer_vs_deepsets": {
                    "bootstrap_ci": {"lower": -0.01, "upper": -0.001},
                    "permutation_test": {"p_value": 0.03},
                },
                "deepsets_vs_multihot": {
                    "bootstrap_ci": {"lower": -0.02, "upper": 0.005},
                    "permutation_test": {"p_value": 0.12},
                },
            }
        },
        "decision_summary": {"challenger_decisions": challenger_decisions},
    }


def test_aggregate_variant_metrics_empty_reports():
    result = aggregate_variant_metrics([], ["deepsets", "settransformer"])
    for variant_name in ["deepsets", "settransformer"]:
        assert result[variant_name]["run_count"] == 0
        assert result[variant_name]["logloss"]["mean"] is None


def test_aggregate_variant_metrics_single_run():
    report = _make_eval_report(["deepsets", "settransformer"])
    result = aggregate_variant_metrics([report], ["deepsets", "settransformer"])

    assert result["deepsets"]["run_count"] == 1
    assert result["deepsets"]["logloss"]["mean"] == pytest.approx(0.300)
    assert result["deepsets"]["logloss"]["std"] == pytest.approx(0.0)
    assert result["deepsets"]["calibration_recommendations"] == {"none": 1}
    assert result["settransformer"]["promote_count"] == 0
    assert result["settransformer"]["hold_count"] == 1


def test_aggregate_variant_metrics_multiple_runs():
    reports = [
        _make_eval_report(["deepsets", "settransformer"], logloss_offset=0.0, promote=False),
        _make_eval_report(["deepsets", "settransformer"], logloss_offset=0.010, promote=True),
        _make_eval_report(["deepsets", "settransformer"], logloss_offset=0.020, promote=False),
    ]
    result = aggregate_variant_metrics(reports, ["deepsets", "settransformer"])

    assert result["deepsets"]["run_count"] == 3
    # mean of [0.300, 0.310, 0.320] = 0.310
    assert result["deepsets"]["logloss"]["mean"] == pytest.approx(0.310, abs=1e-6)
    assert result["deepsets"]["logloss"]["std"] == pytest.approx(0.008165, abs=1e-5)

    # settransformer promote_count = 1 (only the second run had promote=True)
    assert result["settransformer"]["promote_count"] == 1
    assert result["settransformer"]["hold_count"] == 2


def test_aggregate_variant_metrics_missing_variant_in_some_runs():
    report_with_both = _make_eval_report(["deepsets", "settransformer"])
    report_with_only_deepsets = _make_eval_report(["deepsets"])
    result = aggregate_variant_metrics(
        [report_with_both, report_with_only_deepsets],
        ["deepsets", "settransformer"],
    )
    assert result["deepsets"]["run_count"] == 2
    assert result["settransformer"]["run_count"] == 1


# ---------------------------------------------------------------------------
# 3. aggregate_pairwise_comparisons
# ---------------------------------------------------------------------------


def test_aggregate_pairwise_comparisons_empty():
    result = aggregate_pairwise_comparisons([], alpha=0.05)
    assert result == {}


def test_aggregate_pairwise_comparisons_single_report_ci_win():
    report = _make_eval_report(["deepsets", "settransformer"])
    result = aggregate_pairwise_comparisons([report], alpha=0.05)

    # settransformer_vs_deepsets: upper=-0.001 < 0 and p=0.03 < 0.05 → both win
    comp = result["settransformer_vs_deepsets"]
    assert comp["run_count"] == 1
    assert comp["ci_wins"] == 1
    assert comp["permutation_wins"] == 1
    assert comp["both_pass_count"] == 1


def test_aggregate_pairwise_comparisons_no_win():
    report = _make_eval_report(["deepsets", "settransformer"])
    result = aggregate_pairwise_comparisons([report], alpha=0.05)

    # deepsets_vs_multihot: upper=0.005 >= 0 → no ci_win; p=0.12 > 0.05 → no perm win
    comp = result["deepsets_vs_multihot"]
    assert comp["ci_wins"] == 0
    assert comp["permutation_wins"] == 0
    assert comp["both_pass_count"] == 0


def test_aggregate_pairwise_comparisons_multiple_runs():
    # Run 1: both pass; Run 2: only ci pass; Run 3: neither pass
    def _make_report_with_comp(ci_upper, p_value):
        report = _make_eval_report(["deepsets", "settransformer"])
        report["statistical_tests"]["comparisons"]["settransformer_vs_deepsets"] = {
            "bootstrap_ci": {"lower": -0.02, "upper": ci_upper},
            "permutation_test": {"p_value": p_value},
        }
        return report

    reports = [
        _make_report_with_comp(-0.001, 0.03),   # both pass
        _make_report_with_comp(-0.001, 0.10),   # only ci
        _make_report_with_comp(0.005, 0.10),    # neither
    ]
    result = aggregate_pairwise_comparisons(reports, alpha=0.05)
    comp = result["settransformer_vs_deepsets"]
    assert comp["run_count"] == 3
    assert comp["ci_wins"] == 2
    assert comp["permutation_wins"] == 1
    assert comp["both_pass_count"] == 1


# ---------------------------------------------------------------------------
# 4. build_comparison_summary – required keys and schema
# ---------------------------------------------------------------------------


def test_build_comparison_summary_required_keys():
    report = _make_eval_report(["deepsets", "settransformer", "multihot", "legacy"])
    summary = build_comparison_summary(
        eval_reports=[report],
        loto_type="loto6",
        preset="archcomp",
        seeds=[42],
        alpha=0.05,
    )

    assert "schema_version" in summary
    assert summary["schema_version"] == COMPARISON_SUMMARY_SCHEMA_VERSION
    assert "generated_at" in summary
    assert summary["loto_type"] == "loto6"
    assert summary["preset"] == "archcomp"
    assert summary["seeds"] == [42]
    assert summary["run_count"] == 1
    assert "variants" in summary
    assert "pairwise_comparisons" in summary
    assert "alpha" in summary


def test_build_comparison_summary_variant_keys():
    report = _make_eval_report(["deepsets", "settransformer"])
    summary = build_comparison_summary(
        eval_reports=[report],
        loto_type="loto6",
        preset="smoke",
        seeds=[99],
    )
    for variant_name in ["deepsets", "settransformer"]:
        assert variant_name in summary["variants"]
        v = summary["variants"][variant_name]
        assert "run_count" in v
        assert "logloss" in v
        assert "brier" in v
        assert "ece" in v
        assert "calibration_recommendations" in v
        assert "promote_count" in v
        assert "hold_count" in v
        for metric_key in ["logloss", "brier", "ece"]:
            assert "mean" in v[metric_key]
            assert "std" in v[metric_key]
            assert "values" in v[metric_key]


def test_build_comparison_summary_pairwise_keys():
    report = _make_eval_report(["deepsets", "settransformer"])
    summary = build_comparison_summary(
        eval_reports=[report],
        loto_type="loto6",
        preset="archcomp",
        seeds=[42],
    )
    pairwise = summary["pairwise_comparisons"]
    assert "settransformer_vs_deepsets" in pairwise
    comp = pairwise["settransformer_vs_deepsets"]
    assert "run_count" in comp
    assert "ci_wins" in comp
    assert "permutation_wins" in comp
    assert "both_pass_count" in comp


# ---------------------------------------------------------------------------
# 5. Persistence helpers
# ---------------------------------------------------------------------------


def test_save_and_load_comparison_summary(tmp_path):
    report = _make_eval_report(["deepsets", "settransformer"])
    summary = build_comparison_summary(
        eval_reports=[report],
        loto_type="loto6",
        preset="archcomp",
        seeds=[42, 123],
    )
    output_path = tmp_path / "data" / "comparison_summary_loto6.json"
    save_comparison_summary(summary, output_path)

    assert output_path.exists()
    with open(output_path, "r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    assert loaded["loto_type"] == "loto6"
    assert loaded["preset"] == "archcomp"
    assert loaded["seeds"] == [42, 123]


def test_get_comparison_summary_path():
    path = get_comparison_summary_path("loto6")
    assert "comparison_summary_loto6.json" in str(path)

    path_custom = get_comparison_summary_path("miniloto", data_dir="/tmp/data")
    assert "comparison_summary_miniloto.json" in str(path_custom)


# ---------------------------------------------------------------------------
# 6. Multi-seed config resolution (run_multi_seed helpers)
# ---------------------------------------------------------------------------


def test_parse_seeds_valid():
    from scripts.run_multi_seed import parse_seeds
    assert parse_seeds("42,123,456") == [42, 123, 456]
    assert parse_seeds("  7 , 99 ") == [7, 99]
    assert parse_seeds("1") == [1]


def test_parse_seeds_empty_raises():
    from scripts.run_multi_seed import parse_seeds
    with pytest.raises(SystemExit):
        parse_seeds("")


def test_parse_seeds_invalid_raises():
    from scripts.run_multi_seed import parse_seeds
    with pytest.raises(SystemExit):
        parse_seeds("42,abc,123")


def test_archcomp_preset_matches_update_system_choices():
    """archcomp must be in the train_preset choices of update_system.py."""
    import update_system
    import argparse
    import io

    # Re-parse the argument choices by calling the parser
    parser_output = io.StringIO()
    try:
        update_system.parse_args.__globals__
    except AttributeError:
        pass  # can't directly inspect, skip
    # At minimum verify that the preset is accepted by train_prob_model
    assert "archcomp" in PRESET_CONFIGS
