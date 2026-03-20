import numpy as np

from evaluation_statistics import (
    bootstrap_mean_confidence_interval,
    build_logloss_comparison_summary,
    paired_permutation_test,
)


def test_bootstrap_ci_returns_expected_structure():
    ci = bootstrap_mean_confidence_interval([-0.2, -0.1, 0.0, 0.1], confidence=0.9, n_bootstrap=200, seed=7)

    assert ci["confidence"] == 0.9
    assert ci["n_bootstrap"] == 200
    assert ci["lower"] <= ci["mean"] <= ci["upper"]


def test_paired_permutation_test_returns_probability_payload():
    result = paired_permutation_test([-0.4, -0.2, -0.1, 0.0], n_permutations=200, seed=11)

    assert result["n_permutations"] == 200
    assert 0.0 <= result["p_value"] <= 1.0
    assert isinstance(result["observed_mean"], float)


def test_logloss_comparison_summary_contains_ci_and_permutation_sections():
    candidate_losses = np.array([0.10, 0.20, 0.30], dtype=np.float32)
    reference_losses = np.array([0.20, 0.20, 0.40], dtype=np.float32)

    summary = build_logloss_comparison_summary(
        candidate_losses,
        reference_losses,
        candidate_name="multihot",
        reference_name="legacy",
        confidence=0.95,
        n_bootstrap=200,
        n_permutations=200,
        seed=13,
    )

    assert summary["metric"] == "per_draw_logloss"
    assert summary["sample_count"] == 3
    assert summary["candidate_name"] == "multihot"
    assert summary["reference_name"] == "legacy"
    assert "bootstrap_ci" in summary
    assert "permutation_test" in summary
