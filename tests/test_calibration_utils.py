import numpy as np

from calibration_utils import (
    apply_calibration_artifact,
    build_calibration_quality_summary,
    fit_calibrator,
)
from train_prob_model import build_calibration_selection_report


def build_metrics(logloss, brier, ece):
    return {
        "logloss": logloss,
        "brier": brier,
        "ece": ece,
        "mean_overlap_top_k": 1.0,
        "overlap_dist": {"0": 1},
        "calibration": [],
        "calibration_summary": {"ece": ece, "sample_count": 24},
    }


def test_calibrator_fit_and_apply_keep_shape_and_probability_bounds():
    probs = np.array(
        [
            [0.05, 0.20, 0.80, 0.60],
            [0.15, 0.40, 0.70, 0.55],
            [0.10, 0.35, 0.75, 0.65],
        ],
        dtype=np.float32,
    )
    targets = np.array(
        [
            [0, 0, 1, 1],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
        ],
        dtype=np.float32,
    )

    for method_name in ["temperature", "isotonic"]:
        artifact = fit_calibrator(method_name, probs, targets)
        calibrated = apply_calibration_artifact(artifact, probs)

        assert artifact["method"] == method_name
        assert calibrated.shape == probs.shape
        assert np.all(calibrated >= 0.0)
        assert np.all(calibrated <= 1.0)


def test_reliability_summary_contains_bins_and_ece():
    probs = np.array([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]], dtype=np.float32)
    targets = np.array([[0, 1], [0, 1], [1, 0]], dtype=np.float32)

    summary = build_calibration_quality_summary(probs, targets, num_bins=5)

    assert summary["num_bins"] == 5
    assert summary["sample_count"] == probs.size
    assert 0.0 <= summary["ece"] <= 1.0
    assert len(summary["reliability_bins"]) == 5


def test_calibration_selection_prefers_guardrailed_method():
    legacy_holdout = {
        "model": build_metrics(logloss=0.110, brier=0.030, ece=0.030),
        "calibration_methods": {
            "temperature": {
                "status": "fitted",
                "calibration_sample_count": 48,
                "calibrator_summary": {"method": "temperature", "status": "fitted"},
                "test_metrics": build_metrics(logloss=0.100, brier=0.028, ece=0.020),
                "delta_vs_raw": {"logloss": -0.010, "brier": -0.002, "ece": -0.010},
            },
            "isotonic": {
                "status": "fitted",
                "calibration_sample_count": 48,
                "calibrator_summary": {"method": "isotonic", "status": "fitted"},
                "test_metrics": build_metrics(logloss=0.095, brier=0.034, ece=0.050),
                "delta_vs_raw": {"logloss": -0.015, "brier": 0.004, "ece": 0.020},
            },
        },
    }

    selection = build_calibration_selection_report(
        variant_name="legacy",
        legacy_holdout=legacy_holdout,
        walk_forward=None,
        evaluation_calibration_methods=["none", "temperature", "isotonic"],
        ece_threshold=0.03,
    )

    assert selection["recommended_method"] == "temperature"
    rows_by_method = {row["method"]: row for row in selection["methods"]}
    assert rows_by_method["temperature"]["eligible"] is True
    assert rows_by_method["isotonic"]["eligible"] is False
