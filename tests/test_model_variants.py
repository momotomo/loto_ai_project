import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from config import LOOKBACK_WINDOW, LOTO_CONFIG
from train_prob_model import build_prob_model
from model_variants import (
    build_model_samples_from_scaled_rows,
    build_number_level_feature_planes,
    fit_scaler_for_variant,
    prepare_model_dataset,
    transform_features_for_variant,
)


def build_loto6_history(row_count=LOOKBACK_WINDOW + 3):
    records = []
    max_num = LOTO_CONFIG["loto6"]["max_num"]
    for draw_id in range(1, row_count + 1):
        numbers = [((draw_id + offset - 1) % max_num) + 1 for offset in range(6)]
        records.append(
            {
                "draw_id": draw_id,
                "date": f"2024/01/{draw_id:02d}",
                "num1": numbers[0],
                "num2": numbers[1],
                "num3": numbers[2],
                "num4": numbers[3],
                "num5": numbers[4],
                "num6": numbers[5],
            }
        )
    return pd.DataFrame(records)


def test_multihot_dataset_has_expected_shape_and_target_alignment():
    df = build_loto6_history()
    dataset = prepare_model_dataset(df, "loto6", "multihot")
    max_num = LOTO_CONFIG["loto6"]["max_num"]

    assert dataset["raw_features"].shape == (len(df), max_num * 3)
    assert dataset["targets"].shape == (len(df) - LOOKBACK_WINDOW, max_num)
    assert dataset["feature_cols"][:6] == [
        "n01_hit",
        "n01_frequency",
        "n01_gap",
        "n02_hit",
        "n02_frequency",
        "n02_gap",
    ]
    assert dataset["dataset_metadata"]["feature_strategy"] == "derived_multihot"
    expected_numbers = df.loc[LOOKBACK_WINDOW, ["num1", "num2", "num3", "num4", "num5", "num6"]].to_numpy(dtype=np.int32)
    np.testing.assert_allclose(
        dataset["targets"][0],
        np.eye(max_num, dtype=np.float32)[expected_numbers - 1].sum(axis=0),
    )


def test_number_level_features_are_prefix_stable_without_future_leakage():
    df = build_loto6_history(LOOKBACK_WINDOW + 5)
    target_numbers = df[[f"num{i}" for i in range(1, 7)]].to_numpy(dtype=np.int32)
    max_num = LOTO_CONFIG["loto6"]["max_num"]

    shorter = build_number_level_feature_planes(target_numbers[: LOOKBACK_WINDOW + 1], max_num)
    longer = build_number_level_feature_planes(target_numbers, max_num)

    for channel_name in ["hit", "frequency", "gap"]:
        np.testing.assert_allclose(
            shorter[channel_name],
            longer[channel_name][: LOOKBACK_WINDOW + 1],
        )

    first_row_numbers = set(target_numbers[0].tolist())
    hit_row = shorter["hit"][0]
    frequency_row = shorter["frequency"][0]
    gap_row = shorter["gap"][0]
    assert hit_row[0] == 1.0
    assert frequency_row[0] == 1.0
    assert gap_row[0] == 0.0
    absent_number = next(number for number in range(1, max_num + 1) if number not in first_row_numbers)
    assert hit_row[absent_number - 1] == 0.0
    assert frequency_row[absent_number - 1] == 0.0
    assert gap_row[absent_number - 1] == 1.0


def test_deepsets_dataset_has_expected_shape_and_input_summary():
    df = build_loto6_history()
    dataset = prepare_model_dataset(df, "loto6", "deepsets")
    pick_count = LOTO_CONFIG["loto6"]["pick_count"]

    assert dataset["raw_features"].shape == (len(df), pick_count * 3)
    assert dataset["targets"].shape == (len(df) - LOOKBACK_WINDOW, LOTO_CONFIG["loto6"]["max_num"])
    assert dataset["dataset_metadata"]["feature_strategy"] == "set_sequence_deepsets"
    assert dataset["dataset_metadata"]["row_shape"] == [pick_count, 3]
    assert dataset["dataset_metadata"]["input_summary"]["pooling"] == "mean"
    assert dataset["feature_cols"][:6] == [
        "slot01_number_norm",
        "slot01_frequency",
        "slot01_gap",
        "slot02_number_norm",
        "slot02_frequency",
        "slot02_gap",
    ]


def test_deepsets_rows_are_prefix_stable_and_window_shape_is_4d():
    df = build_loto6_history(LOOKBACK_WINDOW + 5)
    shorter_dataset = prepare_model_dataset(df.iloc[: LOOKBACK_WINDOW + 1].copy(), "loto6", "deepsets")
    longer_dataset = prepare_model_dataset(df, "loto6", "deepsets")
    np.testing.assert_allclose(
        shorter_dataset["raw_features"],
        longer_dataset["raw_features"][: LOOKBACK_WINDOW + 1],
    )

    scaler = MinMaxScaler()
    scaler.fit(fit_scaler_for_variant(longer_dataset["raw_features"], longer_dataset["dataset_metadata"]))
    scaled = transform_features_for_variant(longer_dataset["raw_features"], scaler, longer_dataset["dataset_metadata"])
    X = build_model_samples_from_scaled_rows(scaled, 0, 2, longer_dataset["dataset_metadata"])
    assert X.shape == (2, LOOKBACK_WINDOW, LOTO_CONFIG["loto6"]["pick_count"], 3)


def test_deepsets_model_forward_shape_and_probability_range():
    model = build_prob_model(
        input_shape=(LOOKBACK_WINDOW, LOTO_CONFIG["loto6"]["pick_count"], 3),
        max_num=LOTO_CONFIG["loto6"]["max_num"],
        model_variant="deepsets",
        compile_model=False,
    )
    outputs = model(
        tf.convert_to_tensor(
            np.random.rand(2, LOOKBACK_WINDOW, LOTO_CONFIG["loto6"]["pick_count"], 3).astype(np.float32)
        ),
        training=False,
    ).numpy()

    assert outputs.shape == (2, LOTO_CONFIG["loto6"]["max_num"])
    assert float(outputs.min()) >= 0.0
    assert float(outputs.max()) <= 1.0
