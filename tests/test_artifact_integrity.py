import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from artifact_utils import inspect_prediction_artifact_integrity
from config import ARTIFACT_SCHEMA_VERSION, LOOKBACK_WINDOW


def build_feature_frame():
    rows = LOOKBACK_WINDOW + 3
    return pd.DataFrame(
        {
            "num1": np.arange(1, rows + 1, dtype=np.float32),
            "num2": np.arange(11, 11 + rows, dtype=np.float32),
            "sum_val": np.arange(21, 21 + rows, dtype=np.float32),
        }
    )


def build_model(feature_count):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(LOOKBACK_WINDOW, feature_count)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(4),
        ]
    )


def build_deepsets_model(set_cardinality, element_feature_count):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(LOOKBACK_WINDOW, set_cardinality, element_feature_count)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(4),
        ]
    )


def test_bundle_ids_match_across_json_artifacts():
    bundle_id = "bundle-loto6-001"
    manifest = {"bundle_id": bundle_id, "artifact_schema_version": ARTIFACT_SCHEMA_VERSION}
    eval_report = {"bundle_id": bundle_id, "artifact_schema_version": ARTIFACT_SCHEMA_VERSION}
    prediction_history = {"bundle_id": bundle_id, "artifact_schema_version": ARTIFACT_SCHEMA_VERSION}

    assert len({manifest["bundle_id"], eval_report["bundle_id"], prediction_history["bundle_id"]}) == 1
    assert {
        manifest["artifact_schema_version"],
        eval_report["artifact_schema_version"],
        prediction_history["artifact_schema_version"],
    } == {ARTIFACT_SCHEMA_VERSION}


def test_prediction_artifact_integrity_checks_model_and_scaler_shape():
    df = build_feature_frame()
    feature_cols = ["num1", "num2", "sum_val"]
    scaler = MinMaxScaler().fit(df[feature_cols].to_numpy(dtype=np.float32))
    model = build_model(len(feature_cols) + 1)

    issues = inspect_prediction_artifact_integrity(
        df=df,
        feature_cols=feature_cols,
        model=model,
        scaler=scaler,
        lookback_window=LOOKBACK_WINDOW,
    )

    assert any(issue["kind"] == "model_feature_mismatch" for issue in issues)


def test_prediction_artifact_integrity_detects_prepared_feature_count_mismatch():
    df = build_feature_frame()
    feature_cols = ["num1", "num2", "sum_val"]
    scaler = MinMaxScaler().fit(df[feature_cols].to_numpy(dtype=np.float32))
    model = build_model(len(feature_cols))

    issues = inspect_prediction_artifact_integrity(
        df=df,
        feature_cols=feature_cols,
        model=model,
        scaler=scaler,
        lookback_window=LOOKBACK_WINDOW,
        feature_strategy="derived_multihot",
        prepared_feature_count=len(feature_cols) + 2,
    )

    assert any(issue["kind"] == "prepared_feature_mismatch" for issue in issues)


def test_prediction_artifact_integrity_supports_deepsets_input_shapes():
    df = build_feature_frame()
    feature_cols = [
        "slot01_number_norm",
        "slot01_frequency",
        "slot01_gap",
        "slot02_number_norm",
        "slot02_frequency",
        "slot02_gap",
    ]
    scaler = MinMaxScaler().fit(np.random.rand(len(df) * 2, 3).astype(np.float32))
    model = build_deepsets_model(set_cardinality=2, element_feature_count=3)

    issues = inspect_prediction_artifact_integrity(
        df=df,
        feature_cols=feature_cols,
        model=model,
        scaler=scaler,
        lookback_window=LOOKBACK_WINDOW,
        feature_strategy="set_sequence_deepsets",
        prepared_feature_count=len(feature_cols),
        scaler_feature_count=3,
        input_summary={"set_cardinality": 2, "element_feature_count": 3},
    )

    assert issues == []
