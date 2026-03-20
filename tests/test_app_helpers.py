import pandas as pd

from app import get_missing_prediction_artifacts, manifest_requires_calibrator, normalize_prediction_history_df


def test_app_helpers_gracefully_handle_legacy_prediction_history():
    normalized = normalize_prediction_history_df(pd.DataFrame([{"draw_id": 1, "predicted_top_k_hit_count": 0}]))

    assert normalized.loc[0, "model_variant"] == "legacy"
    assert normalized.loc[0, "calibration_method"] == "none"
    assert normalized.loc[0, "evaluation_mode"] == "unknown"


def test_app_helpers_require_calibrator_only_when_manifest_enables_it(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    manifest = {
        "training_context": {"saved_calibration_method": "temperature"},
        "calibration": {"saved_method": "temperature"},
    }

    missing = get_missing_prediction_artifacts(
        loto_type="loto6",
        df=pd.DataFrame({"num1": [1]}),
        feature_cols=["num1"],
        model=object(),
        scaler=object(),
        manifest=manifest,
    )

    assert manifest_requires_calibrator(manifest) is True
    assert any("calibrator" in path for path in missing)
    assert manifest_requires_calibrator({"training_context": {"saved_calibration_method": "none"}}) is False


def test_app_helpers_preserve_new_variant_history_rows():
    normalized = normalize_prediction_history_df(
        pd.DataFrame(
            [
                {
                    "draw_id": 2,
                    "predicted_top_k_hit_count": 1,
                    "model_variant": "deepsets",
                    "calibration_method": "temperature",
                    "evaluation_mode": "walk_forward",
                }
            ]
        )
    )

    assert normalized.loc[0, "model_variant"] == "deepsets"
    assert normalized.loc[0, "calibration_method"] == "temperature"
