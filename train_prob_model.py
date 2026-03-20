import argparse
import json
import os
import pickle
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input, LSTM
from tensorflow.keras.models import Sequential

from artifact_utils import build_data_fingerprint, collect_file_metadata, collect_runtime_environment
from calibration_utils import (
    CALIBRATION_METHOD_CHOICES,
    DEFAULT_EVALUATION_CALIBRATION_METHODS,
    DEFAULT_SAVED_CALIBRATION_METHOD,
    NO_CALIBRATION_METHOD,
    apply_calibration_artifact,
    build_calibration_quality_summary,
    build_identity_calibrator_artifact,
    fit_calibrator,
    get_calibration_method_label,
    get_calibrator_candidate_paths,
    get_calibrator_file_name,
    load_calibrator_artifact,
    resolve_calibration_method,
    resolve_evaluation_calibration_methods,
    summarize_calibrator_artifact,
)
from config import (
    ARTIFACT_SCHEMA_VERSION,
    EVAL_REPORT_SCHEMA_VERSION,
    LOOKBACK_WINDOW,
    LOTO_CONFIG,
    MANIFEST_SCHEMA_VERSION,
    PREPROCESSING_VERSION,
)
from evaluation_statistics import build_logloss_comparison_summary
from model_variants import (
    DEFAULT_MODEL_VARIANT,
    LEGACY_MODEL_VARIANT,
    MODEL_VARIANT_CHOICES,
    MULTIHOT_MODEL_VARIANT,
    create_multi_hot as build_target_multi_hot,
    get_model_variant_label,
    prepare_model_dataset,
    resolve_model_variant,
)

# フリーズ回避
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

DATA_DIR = "data"
MODEL_DIR = "models"
DEFAULT_INITIAL_TRAIN_FRACTION = 0.60
DEFAULT_WALK_FORWARD_TEST_WINDOW = 10
DEFAULT_MAX_WALK_FORWARD_FOLDS = 5
DEFAULT_EVAL_EPOCHS = 8
DEFAULT_FINAL_EPOCHS = 12
DEFAULT_BATCH_SIZE = 64
DEFAULT_PATIENCE = 2
DEFAULT_SEED = 42
DEFAULT_STATISTICAL_ALPHA = 0.05
DEFAULT_BOOTSTRAP_SAMPLES = 2000
DEFAULT_PERMUTATION_SAMPLES = 2000
DEFAULT_CALIBRATION_FRACTION = 0.15
DEFAULT_CALIBRATION_MIN_SAMPLES = 48
DEFAULT_CALIBRATION_MAX_SAMPLES = 160
DEFAULT_CALIBRATION_NUM_BINS = 10
DEFAULT_ECE_THRESHOLD = 0.03
EPSILON = 1e-6
STATIC_BASELINE_NAMES = ["uniform", "frequency", "gap"]
ONLINE_BASELINE_NAMES = ["frequency_online", "gap_online"]
PRESET_CONFIGS = {
    "default": {
        "walk_forward_folds": DEFAULT_MAX_WALK_FORWARD_FOLDS,
        "eval_epochs": DEFAULT_EVAL_EPOCHS,
        "final_epochs": DEFAULT_FINAL_EPOCHS,
        "batch_size": DEFAULT_BATCH_SIZE,
        "patience": DEFAULT_PATIENCE,
    },
    "fast": {
        "walk_forward_folds": 3,
        "eval_epochs": 4,
        "final_epochs": 6,
        "batch_size": 64,
        "patience": 2,
    },
    "smoke": {
        "walk_forward_folds": 1,
        "eval_epochs": 0,
        "final_epochs": 0,
        "batch_size": 32,
        "patience": 1,
    },
}

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def build_bundle_id(loto_type, generated_at):
    compact_timestamp = (
        str(generated_at)
        .replace("-", "")
        .replace(":", "")
        .replace("+00:00", "Z")
        .replace(".", "")
    )
    return f"{loto_type}-{compact_timestamp}"


def set_reproducible_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)


def create_multi_hot(targets, max_num):
    return build_target_multi_hot(targets, max_num)


def sanitize_probabilities(preds):
    return np.clip(np.asarray(preds, dtype=np.float32), EPSILON, 1.0 - EPSILON)


def calculate_calibration(preds, targets, num_bins=DEFAULT_CALIBRATION_NUM_BINS):
    quality_summary = build_calibration_quality_summary(preds, targets, num_bins=num_bins)
    return quality_summary["reliability_bins"]


def calculate_overlap_distribution(preds, targets, k):
    top_k_preds = np.argsort(preds, axis=1)[:, -k:]
    overlaps = []

    for pred_indices, target_vector in zip(top_k_preds, targets):
        true_indices = set(np.flatnonzero(target_vector > 0.5).tolist())
        overlaps.append(len(true_indices & set(pred_indices.tolist())))

    overlap_dist = {str(i): int(overlaps.count(i)) for i in range(k + 1)}
    return overlaps, overlap_dist


def calculate_metrics(preds, targets, k, calibration_num_bins=DEFAULT_CALIBRATION_NUM_BINS):
    preds = sanitize_probabilities(preds)
    targets = np.asarray(targets, dtype=np.float32)
    logloss = -np.mean(targets * np.log(preds) + (1.0 - targets) * np.log(1.0 - preds))
    brier = np.mean((preds - targets) ** 2)
    overlaps, overlap_dist = calculate_overlap_distribution(preds, targets, k)
    quality_summary = build_calibration_quality_summary(preds, targets, num_bins=calibration_num_bins)

    return {
        "logloss": float(logloss),
        "brier": float(brier),
        "ece": quality_summary["ece"],
        "mean_overlap_top_k": float(np.mean(overlaps)) if overlaps else 0.0,
        "overlap_dist": overlap_dist,
        "calibration": quality_summary["reliability_bins"],
        "reliability_bins": quality_summary["reliability_bins"],
        "calibration_summary": {
            key: value for key, value in quality_summary.items() if key != "reliability_bins"
        },
    }


def calculate_per_draw_logloss(preds, targets):
    preds = sanitize_probabilities(preds)
    targets = np.asarray(targets, dtype=np.float32)
    return -np.mean(targets * np.log(preds) + (1.0 - targets) * np.log(1.0 - preds), axis=1)


def parse_model_variants_csv(value):
    variants = []
    for chunk in str(value or "").split(","):
        normalized = chunk.strip()
        if not normalized:
            continue
        resolved = resolve_model_variant(normalized)
        if resolved not in variants:
            variants.append(resolved)
    return variants


def resolve_evaluation_model_variants(args):
    variants = parse_model_variants_csv(args.evaluation_model_variants)
    if not variants:
        variants = [DEFAULT_MODEL_VARIANT, MULTIHOT_MODEL_VARIANT]
    if args.model_variant not in variants:
        variants.insert(0, args.model_variant)
    return variants


def summarize_metrics_for_decision(metrics):
    if not metrics:
        return {}
    calibration_summary = metrics.get("calibration_summary") or {}
    return {
        "logloss": metrics.get("logloss"),
        "brier": metrics.get("brier"),
        "ece": calibration_summary.get("ece", metrics.get("ece")),
        "sample_count": calibration_summary.get("sample_count"),
    }


def summarize_metric_summary_for_decision(metric_summary_entry, calibration_summary=None):
    if not metric_summary_entry:
        return {}
    calibration_payload = calibration_summary or {}
    return {
        "logloss": ((metric_summary_entry.get("logloss") or {}).get("mean")),
        "brier": ((metric_summary_entry.get("brier") or {}).get("mean")),
        "ece": calibration_payload.get("ece", ((metric_summary_entry.get("ece") or {}).get("mean"))),
        "sample_count": calibration_payload.get("sample_count"),
    }


def build_metric_delta_summary(post_metrics, raw_metrics):
    post_summary = summarize_metrics_for_decision(post_metrics)
    raw_summary = summarize_metrics_for_decision(raw_metrics)
    return {
        "logloss": (
            None
            if post_summary.get("logloss") is None or raw_summary.get("logloss") is None
            else float(post_summary["logloss"] - raw_summary["logloss"])
        ),
        "brier": (
            None
            if post_summary.get("brier") is None or raw_summary.get("brier") is None
            else float(post_summary["brier"] - raw_summary["brier"])
        ),
        "ece": (
            None
            if post_summary.get("ece") is None or raw_summary.get("ece") is None
            else float(post_summary["ece"] - raw_summary["ece"])
        ),
    }


def resolve_calibration_sample_count(
    train_sample_count,
    evaluation_calibration_methods,
    calibration_fraction,
    calibration_min_samples,
    calibration_max_samples,
):
    active_methods = [method for method in evaluation_calibration_methods if method != NO_CALIBRATION_METHOD]
    if not active_methods:
        return 0

    proposed = int(round(float(train_sample_count) * float(calibration_fraction)))
    proposed = max(int(calibration_min_samples), proposed)
    proposed = min(int(calibration_max_samples), proposed)
    minimum_model_train_samples = max(LOOKBACK_WINDOW * 2, 32)
    available = int(train_sample_count) - minimum_model_train_samples
    if available <= 0:
        return 0
    return max(0, min(proposed, available))


def build_calibration_method_reports(
    evaluation_calibration_methods,
    calibration_preds,
    calibration_targets,
    test_preds,
    test_targets,
    pick_count,
    calibration_num_bins,
):
    calibration_preds = sanitize_probabilities(calibration_preds)
    test_preds = sanitize_probabilities(test_preds)
    raw_test_metrics = calculate_metrics(test_preds, test_targets, pick_count, calibration_num_bins=calibration_num_bins)
    if len(calibration_targets):
        raw_calibration_metrics = calculate_metrics(
            calibration_preds,
            calibration_targets,
            pick_count,
            calibration_num_bins=calibration_num_bins,
        )
    else:
        raw_calibration_metrics = None

    reports = {}
    calibrated_test_preds_map = {}
    for method_name in evaluation_calibration_methods:
        if method_name == NO_CALIBRATION_METHOD:
            calibrator_artifact = build_identity_calibrator_artifact()
        else:
            calibrator_artifact = fit_calibrator(method_name, calibration_preds, calibration_targets)

        calibrated_calibration_preds = (
            apply_calibration_artifact(calibrator_artifact, calibration_preds)
            if len(calibration_targets)
            else calibration_preds
        )
        calibrated_test_preds = apply_calibration_artifact(calibrator_artifact, test_preds)
        calibrated_test_metrics = calculate_metrics(
            calibrated_test_preds,
            test_targets,
            pick_count,
            calibration_num_bins=calibration_num_bins,
        )
        calibrated_test_preds_map[method_name] = calibrated_test_preds

        if raw_calibration_metrics is not None and len(calibration_targets):
            calibrated_calibration_metrics = calculate_metrics(
                calibrated_calibration_preds,
                calibration_targets,
                pick_count,
                calibration_num_bins=calibration_num_bins,
            )
        else:
            calibrated_calibration_metrics = None

        reports[method_name] = {
            "method": method_name,
            "label": get_calibration_method_label(method_name),
            "status": calibrator_artifact.get("status", "unknown"),
            "calibration_sample_count": int(len(calibration_targets)),
            "calibrator_summary": summarize_calibrator_artifact(calibrator_artifact),
            "calibration_fit_metrics": {
                "pre": raw_calibration_metrics,
                "post": calibrated_calibration_metrics,
            },
            "test_metrics": calibrated_test_metrics,
            "delta_vs_raw": build_metric_delta_summary(calibrated_test_metrics, raw_test_metrics),
        }

    return raw_test_metrics, reports, calibrated_test_preds_map


def summarize_status_counts(status_values):
    counts = {}
    for value in status_values:
        key = str(value or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return counts


def target_vector_to_numbers(target_vector):
    indices = np.flatnonzero(np.asarray(target_vector, dtype=np.float32) > 0.5)
    return [int(index) + 1 for index in indices.tolist()]


def build_prediction_history_records(
    df,
    preds,
    targets,
    loto_type,
    model_variant,
    calibration_method,
    pick_count,
    max_num,
    sample_start,
    evaluation_mode,
    fold_index=None,
):
    top_limit = min(10, max_num)
    records = []

    for offset, (pred_vector, target_vector) in enumerate(zip(preds, targets)):
        sample_index = sample_start + offset
        draw_row = df.iloc[LOOKBACK_WINDOW + sample_index]
        ranked_indices = np.argsort(pred_vector)[::-1]
        predicted_top_k = [int(index) + 1 for index in ranked_indices[:pick_count].tolist()]
        actual_numbers = target_vector_to_numbers(target_vector)
        hit_numbers = sorted(set(predicted_top_k) & set(actual_numbers))

        records.append(
            {
                "draw_id": int(draw_row["draw_id"]),
                "date": str(draw_row["date"]),
                "loto_type": loto_type,
                "model_variant": model_variant,
                "calibration_method": calibration_method,
                "evaluation_mode": evaluation_mode,
                "fold_index": int(fold_index) if fold_index is not None else None,
                "actual_numbers": actual_numbers,
                "predicted_top_k": predicted_top_k,
                "predicted_top_k_hit_count": len(hit_numbers),
                "predicted_top_k_hit_numbers": hit_numbers,
                "top_probability_numbers": [int(index) + 1 for index in ranked_indices[:top_limit].tolist()],
                "top_probability_scores": [float(pred_vector[index]) for index in ranked_indices[:top_limit].tolist()],
                "pick_count": int(pick_count),
                "max_num": int(max_num),
                "hit_rate_any": bool(hit_numbers),
                "hit_rate_two_plus": len(hit_numbers) >= 2,
            }
        )

    return records


def build_prediction_history_artifact(loto_type, records, generated_at, bundle_id):
    sorted_records = sorted(
        records,
        key=lambda item: (
            int(item["draw_id"]),
            str(item["evaluation_mode"]),
            -1 if item["fold_index"] is None else int(item["fold_index"]),
        ),
    )
    return {
        "schema_version": 3,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "bundle_id": bundle_id,
        "generated_at": generated_at,
        "loto_type": loto_type,
        "record_count": len(sorted_records),
        "records": sorted_records,
    }


def metrics_to_summary_entry(metrics):
    return {
        "fold_count": 1,
        "test_samples": None,
        "metric_summary": {
            "logloss": {"mean": metrics["logloss"], "variance": 0.0},
            "brier": {"mean": metrics["brier"], "variance": 0.0},
            "ece": {"mean": metrics.get("ece"), "variance": 0.0},
            "mean_overlap_top_k": {"mean": metrics["mean_overlap_top_k"], "variance": 0.0},
        },
        "overlap_dist_total": metrics["overlap_dist"],
        "calibration": metrics["calibration"],
        "calibration_summary": metrics.get("calibration_summary"),
    }


def summary_entry_to_metrics(summary_entry):
    metric_summary = summary_entry["metric_summary"]
    return {
        "logloss": metric_summary["logloss"]["mean"],
        "brier": metric_summary["brier"]["mean"],
        "ece": ((metric_summary.get("ece") or {}).get("mean")),
        "mean_overlap_top_k": metric_summary["mean_overlap_top_k"]["mean"],
        "overlap_dist": summary_entry["overlap_dist_total"],
        "calibration": summary_entry.get("calibration", []),
        "calibration_summary": summary_entry.get("calibration_summary"),
    }


def summarize_metric_series(values):
    array = np.asarray(values, dtype=np.float32)
    return {
        "mean": float(array.mean()) if array.size else None,
        "variance": float(array.var()) if array.size else None,
    }


def build_prob_model(input_shape, max_num, compile_model=True):
    model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(64, return_sequences=True),
            BatchNormalization(),
            Dropout(0.2),
            LSTM(32),
            BatchNormalization(),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(max_num, activation="sigmoid"),
        ]
    )
    if compile_model:
        model.compile(optimizer="adam", loss="binary_crossentropy")
    return model


def fit_prob_model(X_train, y_train, max_num, epochs, batch_size, patience):
    tf.keras.backend.clear_session()
    model = build_prob_model((X_train.shape[1], X_train.shape[2]), max_num, compile_model=epochs > 0)
    if epochs <= 0:
        return model

    callbacks = []
    fit_kwargs = {
        "x": X_train,
        "y": y_train,
        "epochs": epochs,
        "batch_size": batch_size,
        "verbose": 0,
        "shuffle": False,
    }

    if len(X_train) >= 20:
        callbacks.append(EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True))
        fit_kwargs["validation_split"] = 0.1
    else:
        callbacks.append(EarlyStopping(monitor="loss", patience=max(1, patience // 2), restore_best_weights=True))

    fit_kwargs["callbacks"] = callbacks
    model.fit(**fit_kwargs)
    return model


def select_walk_forward_starts(num_samples, initial_train_fraction, test_window, max_folds):
    if num_samples <= test_window:
        return []

    initial_train_size = max(int(num_samples * initial_train_fraction), test_window * 3)
    initial_train_size = min(initial_train_size, num_samples - test_window)
    possible_starts = list(range(initial_train_size, num_samples - test_window + 1, test_window))

    if not possible_starts:
        return [num_samples - test_window]

    if len(possible_starts) <= max_folds:
        return possible_starts

    indices = np.linspace(0, len(possible_starts) - 1, num=max_folds, dtype=int)
    selected = []
    for index in indices:
        start = possible_starts[int(index)]
        if not selected or selected[-1] != start:
            selected.append(start)
    return selected


def compute_frequency_probs(counts, observed_draws):
    observed = max(int(observed_draws), 1)
    return sanitize_probabilities(counts / observed)


def compute_gap_state(train_targets):
    train_length = len(train_targets)
    gap_state = np.full(train_targets.shape[1], train_length, dtype=np.float32)

    for num_index in range(train_targets.shape[1]):
        hit_indices = np.flatnonzero(train_targets[:, num_index] > 0.5)
        if hit_indices.size:
            gap_state[num_index] = float(train_length - 1 - hit_indices[-1])

    return gap_state


def compute_gap_probs(gap_state, max_num, pick_count):
    weights = np.asarray(gap_state, dtype=np.float32) + 1.0
    weights_mean = float(np.mean(weights)) if len(weights) else 1.0
    base_rate = pick_count / max_num
    return sanitize_probabilities(base_rate * (weights / max(weights_mean, EPSILON)))


def generate_baseline_predictions(train_targets, test_targets, max_num, pick_count):
    n_test = len(test_targets)
    counts = train_targets.sum(axis=0).astype(np.float32)
    gap_state = compute_gap_state(train_targets)
    base_rate = pick_count / max_num

    static_predictions = {
        "uniform": np.full((n_test, max_num), base_rate, dtype=np.float32),
        "frequency": np.tile(compute_frequency_probs(counts, len(train_targets)), (n_test, 1)),
        "gap": np.tile(compute_gap_probs(gap_state, max_num, pick_count), (n_test, 1)),
    }

    online_frequency_preds = np.zeros((n_test, max_num), dtype=np.float32)
    online_gap_preds = np.zeros((n_test, max_num), dtype=np.float32)
    online_counts = counts.copy()
    online_gap_state = gap_state.copy()
    observed_draws = len(train_targets)

    for idx in range(n_test):
        online_frequency_preds[idx] = compute_frequency_probs(online_counts, observed_draws)
        online_gap_preds[idx] = compute_gap_probs(online_gap_state, max_num, pick_count)
        online_counts += test_targets[idx]
        observed_draws += 1
        online_gap_state += 1
        online_gap_state[test_targets[idx] > 0.5] = 0.0

    return {
        "static_baselines": static_predictions,
        "online_baselines": {
            "frequency_online": online_frequency_preds,
            "gap_online": online_gap_preds,
        },
    }


def evaluate_baselines(train_targets, test_targets, max_num, pick_count):
    baseline_predictions = generate_baseline_predictions(train_targets, test_targets, max_num, pick_count)
    evaluated = {}

    for category, named_predictions in baseline_predictions.items():
        evaluated[category] = {}
        for name, preds in named_predictions.items():
            evaluated[category][name] = {
                "preds": preds,
                "metrics": calculate_metrics(preds, test_targets, pick_count),
            }

    return evaluated


def aggregate_named_reports(metrics_by_name, preds_by_name, targets_by_name, pick_count):
    aggregate = {}

    for name, fold_metrics in metrics_by_name.items():
        merged_preds = np.concatenate(preds_by_name[name], axis=0)
        merged_targets = np.concatenate(targets_by_name[name], axis=0)
        merged_metrics = calculate_metrics(merged_preds, merged_targets, pick_count)
        aggregate[name] = {
            "fold_count": len(fold_metrics),
            "test_samples": int(len(merged_targets)),
            "metric_summary": {
                "logloss": summarize_metric_series([item["logloss"] for item in fold_metrics]),
                "brier": summarize_metric_series([item["brier"] for item in fold_metrics]),
                "ece": summarize_metric_series([item.get("ece") for item in fold_metrics if item.get("ece") is not None]),
                "mean_overlap_top_k": summarize_metric_series([item["mean_overlap_top_k"] for item in fold_metrics]),
            },
            "overlap_dist_total": merged_metrics["overlap_dist"],
            "calibration": merged_metrics["calibration"],
            "calibration_summary": merged_metrics.get("calibration_summary"),
        }

    return aggregate


def build_range_metadata(df, start_row, end_row):
    return {
        "start_draw_id": int(df.iloc[start_row]["draw_id"]),
        "start_date": str(df.iloc[start_row]["date"]),
        "end_draw_id": int(df.iloc[end_row]["draw_id"]),
        "end_date": str(df.iloc[end_row]["date"]),
    }


def build_sample_range_metadata(df, sample_start, sample_end):
    start_row = LOOKBACK_WINDOW + sample_start
    end_row = LOOKBACK_WINDOW + sample_end - 1
    return build_range_metadata(df, start_row, end_row)


def prepare_dataset(df, loto_type, model_variant):
    return prepare_model_dataset(df, loto_type, model_variant)


def fit_scaler_for_split(raw_features, train_sample_count):
    observed_rows = min(len(raw_features), LOOKBACK_WINDOW + train_sample_count)
    scaler = MinMaxScaler()
    scaler.fit(raw_features[:observed_rows])
    return scaler


def build_samples_from_scaled_features(scaled_features, targets, sample_start, sample_end):
    X = [scaled_features[index : index + LOOKBACK_WINDOW] for index in range(sample_start, sample_end)]
    y = targets[sample_start:sample_end]
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def evaluate_split(
    df,
    raw_features,
    targets,
    loto_type,
    model_variant,
    pick_count,
    max_num,
    train_end,
    test_end,
    epochs,
    batch_size,
    patience,
    evaluation_calibration_methods,
    calibration_fraction,
    calibration_min_samples,
    calibration_max_samples,
    calibration_num_bins,
    evaluation_mode,
    fold_index=None,
):
    calibration_sample_count = resolve_calibration_sample_count(
        train_sample_count=train_end,
        evaluation_calibration_methods=evaluation_calibration_methods,
        calibration_fraction=calibration_fraction,
        calibration_min_samples=calibration_min_samples,
        calibration_max_samples=calibration_max_samples,
    )
    model_train_end = train_end - calibration_sample_count if calibration_sample_count > 0 else train_end

    scaler = fit_scaler_for_split(raw_features, train_end)
    required_rows = LOOKBACK_WINDOW + test_end
    scaled_features = scaler.transform(raw_features[:required_rows])

    X_train, y_train = build_samples_from_scaled_features(scaled_features, targets, 0, model_train_end)
    X_calibration, y_calibration = build_samples_from_scaled_features(
        scaled_features,
        targets,
        model_train_end,
        train_end,
    )
    X_test, y_test = build_samples_from_scaled_features(scaled_features, targets, train_end, test_end)

    model = fit_prob_model(X_train, y_train, max_num, epochs, batch_size, patience)
    preds_test = sanitize_probabilities(model(tf.convert_to_tensor(X_test), training=False).numpy())
    preds_calibration = (
        sanitize_probabilities(model(tf.convert_to_tensor(X_calibration), training=False).numpy())
        if len(X_calibration)
        else np.zeros((0, max_num), dtype=np.float32)
    )
    model_metrics, calibration_method_reports, calibrated_test_preds_map = build_calibration_method_reports(
        evaluation_calibration_methods=evaluation_calibration_methods,
        calibration_preds=preds_calibration,
        calibration_targets=y_calibration,
        test_preds=preds_test,
        test_targets=y_test,
        pick_count=pick_count,
        calibration_num_bins=calibration_num_bins,
    )
    baseline_outputs = evaluate_baselines(targets[:train_end], y_test, max_num, pick_count)
    history_records = []
    for method_name in evaluation_calibration_methods:
        history_records.extend(
            build_prediction_history_records(
                df=df,
                preds=calibrated_test_preds_map[method_name],
                targets=y_test,
                loto_type=loto_type,
                model_variant=model_variant,
                calibration_method=method_name,
                pick_count=pick_count,
                max_num=max_num,
                sample_start=train_end,
                evaluation_mode=evaluation_mode,
                fold_index=fold_index,
            )
        )

    split_report = {
        "train_samples": int(train_end),
        "model_train_samples": int(len(y_train)),
        "calibration_samples": int(len(y_calibration)),
        "test_samples": int(len(y_test)),
        "train_range": build_sample_range_metadata(df, 0, train_end),
        "model_train_range": build_sample_range_metadata(df, 0, model_train_end) if model_train_end > 0 else None,
        "calibration_range": (
            build_sample_range_metadata(df, model_train_end, train_end) if len(y_calibration) else None
        ),
        "test_range": build_sample_range_metadata(df, train_end, test_end),
        "model": model_metrics,
        "calibration_methods": calibration_method_reports,
        "static_baselines": {name: payload["metrics"] for name, payload in baseline_outputs["static_baselines"].items()},
        "online_baselines": {name: payload["metrics"] for name, payload in baseline_outputs["online_baselines"].items()},
    }

    comparison_payload = {
        "targets": y_test,
        "model_preds": preds_test,
        "calibration_method_preds": calibrated_test_preds_map,
    }

    return split_report, scaler, model, preds_test, y_test, baseline_outputs, history_records, comparison_payload


def evaluate_walk_forward(
    df,
    raw_features,
    targets,
    loto_type,
    model_variant,
    pick_count,
    max_num,
    initial_train_fraction,
    test_window,
    max_folds,
    epochs,
    batch_size,
    patience,
    evaluation_calibration_methods,
    calibration_fraction,
    calibration_min_samples,
    calibration_max_samples,
    calibration_num_bins,
):
    fold_starts = select_walk_forward_starts(len(targets), initial_train_fraction, test_window, max_folds)
    if not fold_starts:
        raise ValueError("walk-forward に必要な fold を作成できませんでした。")
    folds = []
    history_records = []

    metrics_by_name = {"model": []}
    preds_by_name = {"model": []}
    targets_by_name = {"model": []}
    for method_name in evaluation_calibration_methods:
        metrics_by_name[f"calibration:{method_name}"] = []
        preds_by_name[f"calibration:{method_name}"] = []
        targets_by_name[f"calibration:{method_name}"] = []

    for baseline_name in STATIC_BASELINE_NAMES:
        metrics_by_name[f"static_baselines:{baseline_name}"] = []
        preds_by_name[f"static_baselines:{baseline_name}"] = []
        targets_by_name[f"static_baselines:{baseline_name}"] = []

    for baseline_name in ONLINE_BASELINE_NAMES:
        metrics_by_name[f"online_baselines:{baseline_name}"] = []
        preds_by_name[f"online_baselines:{baseline_name}"] = []
        targets_by_name[f"online_baselines:{baseline_name}"] = []

    for fold_index, train_end in enumerate(fold_starts, start=1):
        test_end = min(train_end + test_window, len(targets))
        print(
            f"    - fold {fold_index}/{len(fold_starts)}: "
            f"train_samples={train_end}, test_samples={test_end - train_end}"
        )
        (
            split_report,
            _,
            model,
            preds_test,
            y_test,
            baseline_outputs,
            split_history_records,
            split_comparison_payload,
        ) = evaluate_split(
            df=df,
            raw_features=raw_features,
            targets=targets,
            loto_type=loto_type,
            model_variant=model_variant,
            pick_count=pick_count,
            max_num=max_num,
            train_end=train_end,
            test_end=test_end,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            evaluation_calibration_methods=evaluation_calibration_methods,
            calibration_fraction=calibration_fraction,
            calibration_min_samples=calibration_min_samples,
            calibration_max_samples=calibration_max_samples,
            calibration_num_bins=calibration_num_bins,
            evaluation_mode="walk_forward",
            fold_index=fold_index,
        )
        split_report["fold"] = fold_index
        folds.append(split_report)
        history_records.extend(split_history_records)

        metrics_by_name["model"].append(split_report["model"])
        preds_by_name["model"].append(preds_test)
        targets_by_name["model"].append(y_test)
        for method_name, method_payload in split_report.get("calibration_methods", {}).items():
            key = f"calibration:{method_name}"
            metrics_by_name[key].append(method_payload["test_metrics"])
            preds_by_name[key].append(split_comparison_payload["calibration_method_preds"][method_name])
            targets_by_name[key].append(y_test)

        for category, named_outputs in baseline_outputs.items():
            for name, payload in named_outputs.items():
                key = f"{category}:{name}"
                metrics_by_name[key].append(payload["metrics"])
                preds_by_name[key].append(payload["preds"])
                targets_by_name[key].append(y_test)

        del model
        tf.keras.backend.clear_session()

    aggregate = {
        "model": aggregate_named_reports(
            {"model": metrics_by_name["model"]},
            {"model": preds_by_name["model"]},
            {"model": targets_by_name["model"]},
            pick_count,
        )["model"],
        "calibration_methods": {},
        "static_baselines": {},
        "online_baselines": {},
    }

    for method_name in evaluation_calibration_methods:
        key = f"calibration:{method_name}"
        calibration_aggregate = aggregate_named_reports(
            {method_name: metrics_by_name[key]},
            {method_name: preds_by_name[key]},
            {method_name: targets_by_name[key]},
            pick_count,
        )[method_name]
        status_values = [
            (fold.get("calibration_methods", {}).get(method_name, {}) or {}).get("status") for fold in folds
        ]
        calibration_sample_counts = [
            int((fold.get("calibration_methods", {}).get(method_name, {}) or {}).get("calibration_sample_count", 0))
            for fold in folds
        ]
        calibration_aggregate["status_counts"] = summarize_status_counts(status_values)
        calibration_aggregate["calibration_sample_summary"] = {
            "min": min(calibration_sample_counts) if calibration_sample_counts else 0,
            "max": max(calibration_sample_counts) if calibration_sample_counts else 0,
            "mean": float(np.mean(calibration_sample_counts)) if calibration_sample_counts else 0.0,
        }
        calibration_aggregate["delta_vs_raw"] = {
            "logloss": (
                None
                if calibration_aggregate["metric_summary"]["logloss"]["mean"] is None
                or aggregate["model"]["metric_summary"]["logloss"]["mean"] is None
                else float(
                    calibration_aggregate["metric_summary"]["logloss"]["mean"]
                    - aggregate["model"]["metric_summary"]["logloss"]["mean"]
                )
            ),
            "brier": (
                None
                if calibration_aggregate["metric_summary"]["brier"]["mean"] is None
                or aggregate["model"]["metric_summary"]["brier"]["mean"] is None
                else float(
                    calibration_aggregate["metric_summary"]["brier"]["mean"]
                    - aggregate["model"]["metric_summary"]["brier"]["mean"]
                )
            ),
            "ece": (
                None
                if (calibration_aggregate.get("calibration_summary") or {}).get("ece") is None
                or (aggregate["model"].get("calibration_summary") or {}).get("ece") is None
                else float(
                    calibration_aggregate["calibration_summary"]["ece"]
                    - aggregate["model"]["calibration_summary"]["ece"]
                )
            ),
        }
        aggregate["calibration_methods"][method_name] = calibration_aggregate

    for category in ["static_baselines", "online_baselines"]:
        category_metric_map = {}
        category_pred_map = {}
        category_target_map = {}
        for name in folds[0][category].keys():
            key = f"{category}:{name}"
            category_metric_map[name] = metrics_by_name[key]
            category_pred_map[name] = preds_by_name[key]
            category_target_map[name] = targets_by_name[key]
        aggregate[category] = aggregate_named_reports(category_metric_map, category_pred_map, category_target_map, pick_count)

    comparison_payload = {
        "targets": np.concatenate(targets_by_name["model"], axis=0),
        "model_preds": np.concatenate(preds_by_name["model"], axis=0),
        "calibration_method_preds": {
            method_name: np.concatenate(preds_by_name[f"calibration:{method_name}"], axis=0)
            for method_name in evaluation_calibration_methods
        },
        "static_baseline_preds": {
            name: np.concatenate(preds_by_name[f"static_baselines:{name}"], axis=0) for name in STATIC_BASELINE_NAMES
        },
    }

    return (
        {
            "settings": {
                "initial_train_fraction": initial_train_fraction,
                "test_window": test_window,
                "max_folds": max_folds,
                "fold_selection_policy": "expanding-window rolling-origin, evenly sampled after the initial train period",
            },
            "folds": folds,
            "aggregate": aggregate,
        },
        history_records,
        comparison_payload,
    )


def train_final_model(
    raw_features,
    targets,
    max_num,
    epochs,
    batch_size,
    patience,
    saved_calibration_method,
    calibration_fraction,
    calibration_min_samples,
    calibration_max_samples,
):
    scaler = MinMaxScaler()
    scaler.fit(raw_features)
    scaled_features = scaler.transform(raw_features)
    calibration_sample_count = resolve_calibration_sample_count(
        train_sample_count=len(targets),
        evaluation_calibration_methods=[saved_calibration_method],
        calibration_fraction=calibration_fraction,
        calibration_min_samples=calibration_min_samples,
        calibration_max_samples=calibration_max_samples,
    )
    model_train_end = len(targets) - calibration_sample_count if calibration_sample_count > 0 else len(targets)
    X_train, y_train = build_samples_from_scaled_features(scaled_features, targets, 0, model_train_end)
    X_calibration, y_calibration = build_samples_from_scaled_features(
        scaled_features,
        targets,
        model_train_end,
        len(targets),
    )
    model = fit_prob_model(X_train, y_train, max_num, epochs, batch_size, patience)
    calibration_preds = (
        sanitize_probabilities(model(tf.convert_to_tensor(X_calibration), training=False).numpy())
        if len(X_calibration)
        else np.zeros((0, max_num), dtype=np.float32)
    )
    calibrator_artifact = fit_calibrator(saved_calibration_method, calibration_preds, y_calibration)
    effective_calibration_method = (
        saved_calibration_method if calibrator_artifact.get("status") == "fitted" else NO_CALIBRATION_METHOD
    )
    return scaler, model, calibrator_artifact, {
        "requested_calibration_method": saved_calibration_method,
        "effective_calibration_method": effective_calibration_method,
        "model_train_samples": int(len(y_train)),
        "calibration_samples": int(len(y_calibration)),
        "calibrator_summary": summarize_calibrator_artifact(calibrator_artifact),
    }


def load_existing_feature_cols(loto_type):
    for path in [
        os.path.join(MODEL_DIR, f"{loto_type}_feature_cols.json"),
        os.path.join(DATA_DIR, f"{loto_type}_feature_cols.json"),
    ]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
    return None


def save_feature_cols(loto_type, feature_cols):
    save_json(os.path.join(DATA_DIR, f"{loto_type}_feature_cols.json"), feature_cols)
    save_json(os.path.join(MODEL_DIR, f"{loto_type}_feature_cols.json"), feature_cols)


def remove_calibrator_artifacts(loto_type):
    for path in get_calibrator_candidate_paths(loto_type, data_dir=DATA_DIR, model_dir=MODEL_DIR):
        if os.path.exists(path):
            os.remove(path)


def save_calibrator_artifact(loto_type, calibrator_artifact):
    for path in get_calibrator_candidate_paths(loto_type, data_dir=DATA_DIR, model_dir=MODEL_DIR):
        save_json(path, calibrator_artifact)


def existing_prediction_artifacts_match_feature_layout(model_path, scaler_path, feature_cols):
    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        return False

    try:
        with open(scaler_path, "rb") as handle:
            scaler = pickle.load(handle)
        scaler_feature_count = getattr(scaler, "n_features_in_", None)
        if scaler_feature_count is not None and int(scaler_feature_count) != len(feature_cols):
            return False
    except Exception:
        return False

    model = None
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        input_shape = getattr(model, "input_shape", None)
        if isinstance(input_shape, list):
            input_shape = input_shape[0] if input_shape else None
        if not isinstance(input_shape, tuple) or len(input_shape) < 3:
            return False
        lookback_size = input_shape[1]
        feature_count = input_shape[2]
        if lookback_size is not None and int(lookback_size) != int(LOOKBACK_WINDOW):
            return False
        if feature_count is not None and int(feature_count) != len(feature_cols):
            return False
        return True
    except Exception:
        return False
    finally:
        if model is not None:
            del model
        tf.keras.backend.clear_session()


def persist_final_artifacts(
    loto_type,
    raw_features,
    targets,
    feature_cols,
    max_num,
    epochs,
    batch_size,
    patience,
    skip_final_train,
    saved_calibration_method,
    calibration_fraction,
    calibration_min_samples,
    calibration_max_samples,
):
    model_path = os.path.join(MODEL_DIR, f"{loto_type}_prob.keras")
    scaler_path = os.path.join(MODEL_DIR, f"{loto_type}_scaler.pkl")
    existing_feature_cols = load_existing_feature_cols(loto_type)
    feature_cols_match = existing_feature_cols == feature_cols if existing_feature_cols is not None else False
    artifacts_exist = existing_prediction_artifacts_match_feature_layout(model_path, scaler_path, feature_cols)
    existing_calibrator_artifact, existing_calibrator_path = load_calibrator_artifact(
        loto_type,
        data_dir=DATA_DIR,
        model_dir=MODEL_DIR,
    )
    existing_calibration_method = NO_CALIBRATION_METHOD
    if isinstance(existing_calibrator_artifact, dict) and existing_calibrator_artifact.get("status") == "fitted":
        existing_calibration_method = resolve_calibration_method(
            existing_calibrator_artifact.get("method", NO_CALIBRATION_METHOD)
        )

    save_feature_cols(loto_type, feature_cols)

    reuse_requested = (
        skip_final_train
        and artifacts_exist
        and feature_cols_match
        and (
            saved_calibration_method == NO_CALIBRATION_METHOD
            or existing_calibration_method == saved_calibration_method
        )
    )
    if reuse_requested:
        if saved_calibration_method == NO_CALIBRATION_METHOD:
            remove_calibrator_artifacts(loto_type)
        calibration_path = existing_calibrator_path if saved_calibration_method != NO_CALIBRATION_METHOD else None
        return {
            "status": "reused_existing_artifacts",
            "requested_calibration_method": saved_calibration_method,
            "effective_calibration_method": existing_calibration_method if calibration_path else NO_CALIBRATION_METHOD,
            "calibration_artifact_path": calibration_path,
            "calibrator_summary": summarize_calibrator_artifact(existing_calibrator_artifact)
            if calibration_path
            else summarize_calibrator_artifact(build_identity_calibrator_artifact()),
        }

    effective_epochs = 0 if skip_final_train else epochs
    final_scaler, final_model, calibrator_artifact, calibration_fit_summary = train_final_model(
        raw_features=raw_features,
        targets=targets,
        max_num=max_num,
        epochs=effective_epochs,
        batch_size=batch_size,
        patience=patience,
        saved_calibration_method=saved_calibration_method,
        calibration_fraction=calibration_fraction,
        calibration_min_samples=calibration_min_samples,
        calibration_max_samples=calibration_max_samples,
    )
    final_model.save(model_path)
    with open(scaler_path, "wb") as handle:
        pickle.dump(final_scaler, handle)
    calibration_artifact_path = None
    if calibration_fit_summary["effective_calibration_method"] != NO_CALIBRATION_METHOD:
        save_calibrator_artifact(loto_type, calibrator_artifact)
        calibration_artifact_path = os.path.join(MODEL_DIR, get_calibrator_file_name(loto_type))
    else:
        remove_calibrator_artifacts(loto_type)
    del final_model
    tf.keras.backend.clear_session()

    status = "initialized_without_training" if skip_final_train else "trained_on_full_data"
    return {
        "status": status,
        "requested_calibration_method": saved_calibration_method,
        "effective_calibration_method": calibration_fit_summary["effective_calibration_method"],
        "calibration_artifact_path": calibration_artifact_path,
        "calibrator_summary": calibration_fit_summary["calibrator_summary"],
        "model_train_samples": calibration_fit_summary["model_train_samples"],
        "calibration_samples": calibration_fit_summary["calibration_samples"],
    }


def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def build_training_context(args, loto_type, feature_cols, pick_count, max_num, dataset_metadata):
    return {
        "preset": args.preset,
        "seed": int(args.seed),
        "loto_type": loto_type,
        "model_variant": args.model_variant,
        "evaluation_model_variants": list(args.evaluation_model_variants),
        "saved_calibration_method_requested": args.saved_calibration_method,
        "saved_calibration_method": NO_CALIBRATION_METHOD,
        "evaluation_calibration_methods": list(args.evaluation_calibration_methods),
        "feature_strategy": dataset_metadata.get("feature_strategy"),
        "feature_channels": dataset_metadata.get("feature_channels"),
        "lookback_window": int(LOOKBACK_WINDOW),
        "pick_count": int(pick_count),
        "max_num": int(max_num),
        "feature_column_count": int(len(feature_cols)),
        "hyperparameters": {
            "initial_train_fraction": args.initial_train_fraction,
            "walk_forward_test_window": args.walk_forward_test_window,
            "walk_forward_folds": args.walk_forward_folds,
            "eval_epochs": args.eval_epochs,
            "final_epochs": args.final_epochs,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "calibration_fraction": args.calibration_fraction,
            "calibration_min_samples": args.calibration_min_samples,
            "calibration_max_samples": args.calibration_max_samples,
            "calibration_num_bins": args.calibration_num_bins,
            "ece_threshold": args.ece_threshold,
        },
        "flags": {
            "skip_final_train": bool(args.skip_final_train),
            "skip_legacy_holdout": bool(args.skip_legacy_holdout),
            "skip_walk_forward": bool(args.skip_walk_forward),
        },
    }


def build_model_variant_payload(variant_name, dataset, legacy_holdout, walk_forward, final_artifact_status=None):
    return {
        "variant": variant_name,
        "label": get_model_variant_label(variant_name),
        "dataset_variant": variant_name,
        "feature_strategy": dataset["dataset_metadata"].get("feature_strategy"),
        "feature_channels": dataset["dataset_metadata"].get("feature_channels"),
        "feature_column_count": len(dataset["feature_cols"]),
        "legacy_holdout": legacy_holdout,
        "walk_forward": walk_forward,
        "final_artifact_status": final_artifact_status,
    }


def build_raw_calibration_method_entry(metrics):
    return {
        "method": NO_CALIBRATION_METHOD,
        "label": get_calibration_method_label(NO_CALIBRATION_METHOD),
        "status": "raw_model",
        "status_usable": True,
        "calibration_sample_count": 0,
        "calibrator_summary": summarize_calibrator_artifact(build_identity_calibrator_artifact(status="disabled")),
        "metrics": summarize_metrics_for_decision(metrics),
        "calibration_summary": metrics.get("calibration_summary") or {},
        "delta_vs_raw": build_metric_delta_summary(metrics, metrics),
        "selected": False,
        "eligible": True,
        "guardrails": {
            "logloss_non_worse": True,
            "brier_non_worse": True,
            "ece_not_worse": True,
            "ece_within_threshold": True,
            "ece_guardrail": True,
        },
    }


def get_calibration_method_entry(selection, method_name):
    if not selection:
        return None
    for row in selection.get("methods", []):
        if row.get("method") == method_name:
            return row
    return None


def build_calibration_selection_report(
    variant_name,
    legacy_holdout,
    walk_forward,
    evaluation_calibration_methods,
    ece_threshold,
):
    source = "walk_forward" if walk_forward is not None else "legacy_holdout"
    if walk_forward is not None:
        raw_metrics = summary_entry_to_metrics(walk_forward["aggregate"]["model"])
        aggregate_calibration_methods = (walk_forward.get("aggregate") or {}).get("calibration_methods") or {}
        fold_count = len(walk_forward.get("folds", []))
        method_payloads = {
            method_name: {
                "status": (
                    "fitted"
                    if int(((aggregate_calibration_methods.get(method_name) or {}).get("status_counts") or {}).get("fitted", 0)) > 0
                    else "skipped"
                ),
                "status_counts": (aggregate_calibration_methods.get(method_name) or {}).get("status_counts"),
                "calibration_sample_count": int(
                    round(
                        float(
                            ((aggregate_calibration_methods.get(method_name) or {}).get("calibration_sample_summary") or {}).get(
                                "mean", 0.0
                            )
                        )
                    )
                ),
                "calibrator_summary": {
                    "method": method_name,
                    "status_counts": (aggregate_calibration_methods.get(method_name) or {}).get("status_counts"),
                },
                "metrics": summarize_metrics_for_decision(
                    summary_entry_to_metrics(aggregate_calibration_methods.get(method_name) or metrics_to_summary_entry(raw_metrics))
                ),
                "calibration_summary": (aggregate_calibration_methods.get(method_name) or {}).get("calibration_summary") or {},
                "delta_vs_raw": (aggregate_calibration_methods.get(method_name) or {}).get("delta_vs_raw")
                or build_metric_delta_summary(raw_metrics, raw_metrics),
            }
            for method_name in evaluation_calibration_methods
            if method_name != NO_CALIBRATION_METHOD and method_name in aggregate_calibration_methods
        }
    elif legacy_holdout is not None:
        raw_metrics = legacy_holdout["model"]
        fold_count = 1
        method_payloads = {
            method_name: {
                "status": (payload or {}).get("status", "unknown"),
                "status_counts": summarize_status_counts([(payload or {}).get("status")]),
                "calibration_sample_count": int((payload or {}).get("calibration_sample_count", 0)),
                "calibrator_summary": (payload or {}).get("calibrator_summary") or {"method": method_name},
                "metrics": summarize_metrics_for_decision((payload or {}).get("test_metrics") or raw_metrics),
                "calibration_summary": (((payload or {}).get("test_metrics") or {}).get("calibration_summary") or {}),
                "delta_vs_raw": (payload or {}).get("delta_vs_raw") or build_metric_delta_summary(raw_metrics, raw_metrics),
            }
            for method_name, payload in ((legacy_holdout.get("calibration_methods") or {}).items())
            if method_name != NO_CALIBRATION_METHOD
        }
    else:
        raise ValueError(f"[{variant_name}] calibration selection requires walk-forward or legacy holdout results.")

    raw_summary = summarize_metrics_for_decision(raw_metrics)
    rows = [build_raw_calibration_method_entry(raw_metrics)]

    for method_name in evaluation_calibration_methods:
        if method_name == NO_CALIBRATION_METHOD:
            continue

        payload = method_payloads.get(method_name)
        metrics = (payload or {}).get("metrics") or {}
        status_counts = (payload or {}).get("status_counts") or {}
        fitted_count = int(status_counts.get("fitted", 0))
        status_usable = fitted_count > 0 or str((payload or {}).get("status")) == "fitted"
        logloss_non_worse = (
            None
            if metrics.get("logloss") is None or raw_summary.get("logloss") is None
            else bool(metrics["logloss"] <= raw_summary["logloss"] + EPSILON)
        )
        brier_non_worse = (
            None
            if metrics.get("brier") is None or raw_summary.get("brier") is None
            else bool(metrics["brier"] <= raw_summary["brier"] + EPSILON)
        )
        ece_not_worse = (
            None
            if metrics.get("ece") is None or raw_summary.get("ece") is None
            else bool(metrics["ece"] <= raw_summary["ece"] + EPSILON)
        )
        ece_within_threshold = None if metrics.get("ece") is None else bool(metrics["ece"] <= ece_threshold + EPSILON)
        ece_guardrail = (
            None
            if ece_not_worse is None and ece_within_threshold is None
            else bool(ece_not_worse or ece_within_threshold)
        )
        eligible = bool(
            status_usable
            and logloss_non_worse is True
            and brier_non_worse is True
            and ece_guardrail is True
        )
        rows.append(
            {
                "method": method_name,
                "label": get_calibration_method_label(method_name),
                "status": (payload or {}).get("status", "missing"),
                "status_usable": status_usable,
                "status_counts": status_counts or None,
                "calibration_sample_count": int((payload or {}).get("calibration_sample_count", 0)),
                "calibrator_summary": (payload or {}).get("calibrator_summary") or {"method": method_name},
                "metrics": metrics,
                "calibration_summary": (payload or {}).get("calibration_summary") or {},
                "delta_vs_raw": (payload or {}).get("delta_vs_raw") or build_metric_delta_summary(raw_metrics, raw_metrics),
                "selected": False,
                "eligible": eligible,
                "guardrails": {
                    "logloss_non_worse": logloss_non_worse,
                    "brier_non_worse": brier_non_worse,
                    "ece_not_worse": ece_not_worse,
                    "ece_within_threshold": ece_within_threshold,
                    "ece_guardrail": ece_guardrail,
                },
            }
        )

    ranked_rows = sorted(
        rows,
        key=lambda item: (
            not bool(item.get("eligible")),
            float("inf") if (item.get("metrics") or {}).get("logloss") is None else float(item["metrics"]["logloss"]),
            float("inf") if (item.get("metrics") or {}).get("brier") is None else float(item["metrics"]["brier"]),
            float("inf") if (item.get("metrics") or {}).get("ece") is None else float(item["metrics"]["ece"]),
            item.get("method") != NO_CALIBRATION_METHOD,
        ),
    )
    recommended_entry = ranked_rows[0] if ranked_rows else rows[0]
    recommended_method = recommended_entry["method"]
    for row in rows:
        row["selected"] = row["method"] == recommended_method

    reason_codes = []
    if recommended_method == NO_CALIBRATION_METHOD:
        reason_codes.append("retain_raw_probabilities")
        if any(row.get("method") != NO_CALIBRATION_METHOD and not row.get("eligible") for row in rows):
            reason_codes.append("all_posthoc_methods_failed_guardrails")
    else:
        reason_codes.append(f"select_{recommended_method}_for_{variant_name}")

    return {
        "variant": variant_name,
        "source": source,
        "fold_count": int(fold_count),
        "raw_method": NO_CALIBRATION_METHOD,
        "raw_label": get_calibration_method_label(NO_CALIBRATION_METHOD),
        "raw_metrics": raw_summary,
        "recommended_method": recommended_method,
        "recommended_label": get_calibration_method_label(recommended_method),
        "recommended_metrics": recommended_entry.get("metrics") or {},
        "recommended_delta_vs_raw": recommended_entry.get("delta_vs_raw") or build_metric_delta_summary(raw_metrics, raw_metrics),
        "ece_threshold": float(ece_threshold),
        "rule": {
            "selection_policy": "pick the lowest-logloss usable calibration method subject to guardrails; otherwise keep raw probabilities",
            "guardrails": [
                "post-calibration logloss must not worsen vs raw",
                "post-calibration brier must not worsen vs raw",
                "ECE must improve vs raw or stay within ece_threshold",
                "non-identity post-hoc methods must be fitted on the calibration split",
            ],
        },
        "reason_codes": reason_codes,
        "methods": rows,
    }


def rank_model_variants_by_logloss(model_variant_reports):
    rankings = []
    for variant_name, payload in model_variant_reports.items():
        calibration_selection = payload.get("calibration_selection") or {}
        logloss_mean = (calibration_selection.get("recommended_metrics") or {}).get("logloss")
        if logloss_mean is None:
            walk_forward = payload.get("walk_forward")
            if walk_forward is not None:
                logloss_mean = ((walk_forward["aggregate"]["model"]["metric_summary"]["logloss"]).get("mean"))
            else:
                legacy_holdout = payload.get("legacy_holdout")
                logloss_mean = legacy_holdout["model"]["logloss"] if legacy_holdout else None
        rankings.append(
            {
                "variant": variant_name,
                "label": get_model_variant_label(variant_name),
                "logloss_mean": logloss_mean,
                "calibration_method": calibration_selection.get("recommended_method", NO_CALIBRATION_METHOD),
            }
        )
    return sorted(rankings, key=lambda item: float("inf") if item["logloss_mean"] is None else item["logloss_mean"])


def select_comparison_prediction_stream(variant_name, comparison_payloads, calibration_selections):
    payload = comparison_payloads[variant_name]
    selection = calibration_selections.get(variant_name) or {}
    selected_method = resolve_calibration_method(selection.get("recommended_method", NO_CALIBRATION_METHOD))
    if selected_method != NO_CALIBRATION_METHOD:
        calibrated_preds = (payload.get("calibration_method_preds") or {}).get(selected_method)
        if calibrated_preds is not None:
            return calibrated_preds, selected_method
    return payload["model_preds"], NO_CALIBRATION_METHOD


def build_statistical_tests(
    model_variant_reports,
    comparison_payloads,
    calibration_selections,
    alpha,
    bootstrap_samples,
    permutation_samples,
    seed,
):
    if not comparison_payloads:
        return None

    reference_variant = LEGACY_MODEL_VARIANT if LEGACY_MODEL_VARIANT in comparison_payloads else next(iter(comparison_payloads))
    static_summary = model_variant_reports[reference_variant]["walk_forward"]["aggregate"]["static_baselines"]
    best_static_name = min(
        static_summary,
        key=lambda name: static_summary[name]["metric_summary"]["logloss"]["mean"],
    )
    reference_targets = comparison_payloads[reference_variant]["targets"]
    best_static_losses = calculate_per_draw_logloss(
        comparison_payloads[reference_variant]["static_baseline_preds"][best_static_name],
        reference_targets,
    )
    comparisons = {}
    selected_calibration_methods = {}

    if LEGACY_MODEL_VARIANT in comparison_payloads:
        legacy_preds, legacy_method = select_comparison_prediction_stream(
            LEGACY_MODEL_VARIANT,
            comparison_payloads,
            calibration_selections,
        )
        selected_calibration_methods[LEGACY_MODEL_VARIANT] = legacy_method
        legacy_vs_best_static = build_logloss_comparison_summary(
            calculate_per_draw_logloss(legacy_preds, reference_targets),
            best_static_losses,
            candidate_name=LEGACY_MODEL_VARIANT,
            reference_name=f"static_baseline:{best_static_name}",
            confidence=1.0 - alpha,
            n_bootstrap=bootstrap_samples,
            n_permutations=permutation_samples,
            seed=seed,
        )
        legacy_vs_best_static["candidate_calibration_method"] = legacy_method
        legacy_vs_best_static["reference_calibration_method"] = NO_CALIBRATION_METHOD
        comparisons["legacy_vs_best_static"] = legacy_vs_best_static

    if MULTIHOT_MODEL_VARIANT in comparison_payloads:
        multihot_preds, multihot_method = select_comparison_prediction_stream(
            MULTIHOT_MODEL_VARIANT,
            comparison_payloads,
            calibration_selections,
        )
        selected_calibration_methods[MULTIHOT_MODEL_VARIANT] = multihot_method
        multihot_vs_best_static = build_logloss_comparison_summary(
            calculate_per_draw_logloss(multihot_preds, reference_targets),
            best_static_losses,
            candidate_name=MULTIHOT_MODEL_VARIANT,
            reference_name=f"static_baseline:{best_static_name}",
            confidence=1.0 - alpha,
            n_bootstrap=bootstrap_samples,
            n_permutations=permutation_samples,
            seed=seed,
        )
        multihot_vs_best_static["candidate_calibration_method"] = multihot_method
        multihot_vs_best_static["reference_calibration_method"] = NO_CALIBRATION_METHOD
        comparisons["multihot_vs_best_static"] = multihot_vs_best_static

    if LEGACY_MODEL_VARIANT in comparison_payloads and MULTIHOT_MODEL_VARIANT in comparison_payloads:
        legacy_preds, legacy_method = select_comparison_prediction_stream(
            LEGACY_MODEL_VARIANT,
            comparison_payloads,
            calibration_selections,
        )
        multihot_preds, multihot_method = select_comparison_prediction_stream(
            MULTIHOT_MODEL_VARIANT,
            comparison_payloads,
            calibration_selections,
        )
        selected_calibration_methods[LEGACY_MODEL_VARIANT] = legacy_method
        selected_calibration_methods[MULTIHOT_MODEL_VARIANT] = multihot_method
        multihot_vs_legacy = build_logloss_comparison_summary(
            calculate_per_draw_logloss(multihot_preds, reference_targets),
            calculate_per_draw_logloss(legacy_preds, reference_targets),
            candidate_name=MULTIHOT_MODEL_VARIANT,
            reference_name=LEGACY_MODEL_VARIANT,
            confidence=1.0 - alpha,
            n_bootstrap=bootstrap_samples,
            n_permutations=permutation_samples,
            seed=seed,
        )
        multihot_vs_legacy["candidate_calibration_method"] = multihot_method
        multihot_vs_legacy["reference_calibration_method"] = legacy_method
        comparisons["multihot_vs_legacy"] = multihot_vs_legacy

    return {
        "metric": "per_draw_logloss",
        "test_source": "walk_forward_predictions",
        "comparison_mode": "selected_calibration_method_per_variant",
        "best_static_baseline_name": best_static_name,
        "selected_calibration_methods": selected_calibration_methods,
        "alpha": alpha,
        "comparisons": comparisons,
    }


def build_adoption_decision(
    model_variant_reports,
    calibration_selections,
    statistical_tests,
    production_variant,
    production_calibration_method,
    alpha,
    ece_threshold,
):
    rankings = rank_model_variants_by_logloss(model_variant_reports)
    recommended_variant = production_variant
    candidate_selection = calibration_selections.get(MULTIHOT_MODEL_VARIANT) or {}
    candidate_selected_metrics = candidate_selection.get("recommended_metrics") or {}
    candidate_raw_metrics = candidate_selection.get("raw_metrics") or {}
    candidate_selected_method = candidate_selection.get("recommended_method", NO_CALIBRATION_METHOD)
    recommended_calibration_method = (
        (calibration_selections.get(production_variant) or {}).get("recommended_method", production_calibration_method)
    )
    decision_flags = {
        "beats_best_static_with_ci": False,
        "beats_best_static_with_p_value": False,
        "beats_legacy_with_ci": False,
        "beats_legacy_with_p_value": False,
        "calibration_logloss_non_worse": False,
        "calibration_brier_non_worse": False,
        "calibration_ece_guardrail": False,
    }
    reason_codes = []
    reason_summary = []

    comparisons = (statistical_tests or {}).get("comparisons") or {}
    multihot_vs_best_static = comparisons.get("multihot_vs_best_static")
    multihot_vs_legacy = comparisons.get("multihot_vs_legacy")

    if multihot_vs_best_static is None:
        reason_codes.append("multihot_vs_best_static_missing")
    else:
        ci = multihot_vs_best_static.get("bootstrap_ci") or {}
        p_value = ((multihot_vs_best_static.get("permutation_test") or {}).get("p_value"))
        decision_flags["beats_best_static_with_ci"] = ci.get("upper") is not None and ci["upper"] < 0.0
        decision_flags["beats_best_static_with_p_value"] = p_value is not None and p_value < alpha

    if multihot_vs_legacy is None:
        reason_codes.append("multihot_vs_legacy_missing")
    else:
        ci = multihot_vs_legacy.get("bootstrap_ci") or {}
        p_value = ((multihot_vs_legacy.get("permutation_test") or {}).get("p_value"))
        decision_flags["beats_legacy_with_ci"] = ci.get("upper") is not None and ci["upper"] < 0.0
        decision_flags["beats_legacy_with_p_value"] = p_value is not None and p_value < alpha

    if candidate_selected_metrics:
        decision_flags["calibration_logloss_non_worse"] = (
            candidate_selected_metrics.get("logloss") is not None
            and candidate_raw_metrics.get("logloss") is not None
            and candidate_selected_metrics["logloss"] <= candidate_raw_metrics["logloss"] + EPSILON
        )
        decision_flags["calibration_brier_non_worse"] = (
            candidate_selected_metrics.get("brier") is not None
            and candidate_raw_metrics.get("brier") is not None
            and candidate_selected_metrics["brier"] <= candidate_raw_metrics["brier"] + EPSILON
        )
        decision_flags["calibration_ece_guardrail"] = bool(
            candidate_selected_metrics.get("ece") is not None
            and (
                (
                    candidate_raw_metrics.get("ece") is not None
                    and candidate_selected_metrics["ece"] <= candidate_raw_metrics["ece"] + EPSILON
                )
                or candidate_selected_metrics["ece"] <= ece_threshold + EPSILON
            )
        )
    else:
        reason_codes.append("candidate_calibration_metrics_missing")

    should_promote_multihot = all(value is True for value in decision_flags.values())
    if should_promote_multihot:
        recommended_variant = MULTIHOT_MODEL_VARIANT
        recommended_calibration_method = candidate_selected_method
        reason_codes.append("promote_multihot")
        reason_summary.append(
            "multihot が best static / legacy に対する統計条件を満たし、選択された calibration も logloss・Brier・ECE guardrail を満たしました。"
        )
    else:
        reason_codes.append(f"retain_production_variant:{production_variant}")
        if not decision_flags["beats_best_static_with_ci"] or not decision_flags["beats_best_static_with_p_value"]:
            reason_summary.append("best static baseline に対する統計的優位が十分ではありません。")
        if not decision_flags["beats_legacy_with_ci"] or not decision_flags["beats_legacy_with_p_value"]:
            reason_summary.append("legacy に対する統計的優位が十分ではありません。")
        if not decision_flags["calibration_logloss_non_worse"]:
            reason_summary.append("候補 calibration 後の logloss が raw を明確に改善または維持できていません。")
        if not decision_flags["calibration_brier_non_worse"]:
            reason_summary.append("候補 calibration 後の Brier が raw より悪化しています。")
        if not decision_flags["calibration_ece_guardrail"]:
            reason_summary.append("候補 calibration 後の ECE が許容範囲を満たしていません。")

    return {
        "candidate_variant": MULTIHOT_MODEL_VARIANT,
        "candidate_calibration_method": candidate_selected_method,
        "production_variant": production_variant,
        "production_saved_calibration_method": production_calibration_method,
        "recommended_variant": recommended_variant,
        "recommended_calibration_method": recommended_calibration_method,
        "best_variant_by_logloss": rankings[0]["variant"] if rankings else None,
        "rankings": rankings,
        "metric": "logloss",
        "alpha": alpha,
        "rule": {
            "multihot_vs_best_static": "bootstrap_ci.upper < 0 and permutation_test.p_value < alpha",
            "multihot_vs_legacy": "bootstrap_ci.upper < 0 and permutation_test.p_value < alpha",
            "candidate_calibration": (
                "selected calibration must keep post-calibration logloss/brier non-worse vs raw, "
                "and ECE must improve vs raw or stay <= ece_threshold"
            ),
        },
        "flags": decision_flags,
        "should_promote_candidate": should_promote_multihot,
        "reason_codes": reason_codes,
        "reason_summary": reason_summary,
        "ece_threshold": ece_threshold,
    }


def build_compat_report_sections(legacy_holdout, walk_forward):
    if legacy_holdout is not None:
        return legacy_holdout["model"], legacy_holdout["static_baselines"], legacy_holdout["online_baselines"]

    aggregate = walk_forward["aggregate"]
    return (
        summary_entry_to_metrics(aggregate["model"]),
        {name: summary_entry_to_metrics(entry) for name, entry in aggregate["static_baselines"].items()},
        {name: summary_entry_to_metrics(entry) for name, entry in aggregate["online_baselines"].items()},
    )


def select_primary_evaluation(legacy_holdout, walk_forward_report):
    if walk_forward_report is not None:
        return {
            "source": "walk_forward",
            "model_summary": walk_forward_report["aggregate"]["model"]["metric_summary"],
            "static_summary": {
                name: entry["metric_summary"] for name, entry in walk_forward_report["aggregate"]["static_baselines"].items()
            },
            "fold_count": len(walk_forward_report["folds"]),
            "test_window": walk_forward_report["settings"]["test_window"],
        }

    if legacy_holdout is not None:
        return {
            "source": "legacy_holdout",
            "model_summary": metrics_to_summary_entry(legacy_holdout["model"])["metric_summary"],
            "static_summary": {
                name: metrics_to_summary_entry(metrics)["metric_summary"]
                for name, metrics in legacy_holdout["static_baselines"].items()
            },
            "fold_count": 1,
            "test_window": legacy_holdout["test_samples"],
        }

    raise ValueError("manifest 生成には少なくとも1つの評価結果が必要です。")


def build_metric_snapshot(metric_summary):
    if not metric_summary:
        return {}
    return {
        "logloss": metric_summary.get("logloss"),
        "brier": metric_summary.get("brier"),
        "ece": metric_summary.get("ece"),
        "sample_count": metric_summary.get("sample_count"),
    }


def build_manifest(
    loto_type,
    df,
    legacy_holdout,
    walk_forward_report,
    model_variant_reports,
    statistical_tests,
    decision_summary,
    eval_report_path,
    final_artifact_status,
    prediction_history_path,
    prediction_history_rows,
    generated_at,
    bundle_id,
    feature_cols_path,
    model_path,
    scaler_path,
    data_fingerprint,
    training_context,
    runtime_environment,
):
    primary_evaluation = select_primary_evaluation(legacy_holdout, walk_forward_report)
    primary_variant_payload = model_variant_reports.get(training_context.get("model_variant")) or {}
    primary_calibration_selection = primary_variant_payload.get("calibration_selection") or {}
    model_summary = primary_evaluation["model_summary"]
    static_summary = primary_evaluation["static_summary"]
    best_static_name = min(
        static_summary,
        key=lambda name: static_summary[name]["logloss"]["mean"],
    )

    best_static = static_summary[best_static_name]
    primary_model = {
        "logloss_mean": model_summary["logloss"]["mean"],
        "brier_mean": model_summary["brier"]["mean"],
        "ece": ((model_summary.get("ece") or {}).get("mean")),
        "mean_overlap_top_k_mean": model_summary["mean_overlap_top_k"]["mean"],
    }
    pre_calibration_summary = primary_calibration_selection.get("raw_metrics") or {
        "logloss": primary_model.get("logloss_mean"),
        "brier": primary_model.get("brier_mean"),
        "ece": primary_model.get("ece"),
        "sample_count": None,
    }
    selected_calibration_summary = primary_calibration_selection.get("recommended_metrics") or pre_calibration_summary
    saved_calibration_method = training_context.get("saved_calibration_method", NO_CALIBRATION_METHOD)
    saved_calibration_entry = get_calibration_method_entry(primary_calibration_selection, saved_calibration_method)
    saved_calibration_summary = (
        (saved_calibration_entry or {}).get("metrics")
        or (selected_calibration_summary if saved_calibration_method == primary_calibration_selection.get("recommended_method") else pre_calibration_summary)
    )
    calibration_section = {
        "enabled": saved_calibration_method != NO_CALIBRATION_METHOD,
        "requested_method": training_context.get("saved_calibration_method_requested", NO_CALIBRATION_METHOD),
        "saved_method": saved_calibration_method,
        "recommended_method": primary_calibration_selection.get("recommended_method", NO_CALIBRATION_METHOD),
        "evaluation_methods": training_context.get("evaluation_calibration_methods") or [],
        "artifact_path": final_artifact_status.get("calibration_artifact_path"),
        "summary": {
            "source": primary_calibration_selection.get("source"),
            "raw_metrics": pre_calibration_summary,
            "recommended_metrics": selected_calibration_summary,
            "recommended_delta_vs_raw": primary_calibration_selection.get("recommended_delta_vs_raw"),
            "reason_codes": primary_calibration_selection.get("reason_codes") or [],
        },
    }
    metrics_summary = {
        "evaluation_source": primary_evaluation["source"],
        "fold_count": primary_evaluation["fold_count"],
        "test_window": primary_evaluation["test_window"],
        "saved_model_variant": training_context.get("model_variant"),
        "saved_calibration_method": saved_calibration_method,
        "recommended_model_variant": (decision_summary or {}).get("recommended_variant"),
        "recommended_calibration_method": (decision_summary or {}).get(
            "recommended_calibration_method",
            primary_calibration_selection.get("recommended_method", NO_CALIBRATION_METHOD),
        ),
        "primary_model": primary_model,
        "selected_calibration_model": selected_calibration_summary,
        "saved_production_model": saved_calibration_summary,
        "pre_calibration_logloss": pre_calibration_summary.get("logloss"),
        "post_calibration_logloss": selected_calibration_summary.get("logloss"),
        "pre_calibration_brier": pre_calibration_summary.get("brier"),
        "post_calibration_brier": selected_calibration_summary.get("brier"),
        "pre_calibration_ece": pre_calibration_summary.get("ece"),
        "post_calibration_ece": selected_calibration_summary.get("ece"),
        "calibration_summary": calibration_section["summary"],
        "walk_forward_model": primary_model if primary_evaluation["source"] == "walk_forward" else None,
        "best_static_baseline": {
            "name": best_static_name,
            "logloss_mean": best_static["logloss"]["mean"],
            "brier_mean": best_static["brier"]["mean"],
            "mean_overlap_top_k_mean": best_static["mean_overlap_top_k"]["mean"],
            "delta_model_minus_baseline": {
                "logloss": model_summary["logloss"]["mean"] - best_static["logloss"]["mean"],
                "brier": model_summary["brier"]["mean"] - best_static["brier"]["mean"],
                "mean_overlap_top_k": model_summary["mean_overlap_top_k"]["mean"] - best_static["mean_overlap_top_k"]["mean"],
            },
        },
        "decision_summary": decision_summary,
        "final_artifact_status": final_artifact_status,
    }
    artifact_paths = {
        "eval_report": eval_report_path,
        "prediction_history": prediction_history_path,
        "model": model_path,
        "scaler": scaler_path,
        "feature_cols": feature_cols_path,
    }
    if final_artifact_status.get("calibration_artifact_path"):
        artifact_paths["calibrator"] = final_artifact_status["calibration_artifact_path"]

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "bundle_id": bundle_id,
        "generated_at": generated_at,
        "loto_type": loto_type,
        "latest_draw_id": int(df.iloc[-1]["draw_id"]),
        "train_range": build_range_metadata(df, 0, len(df) - 1),
        "data_fingerprint": data_fingerprint,
        "training_context": training_context,
        "runtime_environment": runtime_environment,
        "calibration": calibration_section,
        "metrics_summary": metrics_summary,
        "model_variants": {
            variant_name: {
                "label": payload.get("label"),
                "feature_strategy": payload.get("feature_strategy"),
                "feature_column_count": payload.get("feature_column_count"),
                "selected_evaluation_calibration_method": (
                    (payload.get("calibration_selection") or {}).get("recommended_method", NO_CALIBRATION_METHOD)
                ),
                "selected_evaluation_metrics": (payload.get("calibration_selection") or {}).get("recommended_metrics"),
                "final_artifact_status": payload.get("final_artifact_status"),
            }
            for variant_name, payload in model_variant_reports.items()
        },
        "statistical_tests": statistical_tests,
        "decision_summary": decision_summary,
        "artifacts": artifact_paths,
        "artifact_metadata": collect_file_metadata(artifact_paths),
        "prediction_history_path": prediction_history_path,
        "prediction_history_rows": int(prediction_history_rows),
        "git_commit": get_git_commit(),
    }


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def apply_preset(args):
    preset_values = PRESET_CONFIGS[args.preset]
    for field, value in preset_values.items():
        if getattr(args, field) is None:
            setattr(args, field, value)
    return args


def train_for_type(loto_type, args):
    config = LOTO_CONFIG[loto_type]
    df_path = os.path.join(DATA_DIR, f"{loto_type}_processed.csv")
    raw_path = os.path.join(DATA_DIR, f"{loto_type}_raw.csv")
    if not os.path.exists(df_path):
        print(f"[{loto_type}] データなし。スキップします。")
        return False

    df = pd.read_csv(df_path).sort_values("draw_id").reset_index(drop=True)
    latest_draw_id = int(df.iloc[-1]["draw_id"])
    runtime_environment = collect_runtime_environment()
    generated_at = datetime.now(timezone.utc).isoformat()
    bundle_id = build_bundle_id(loto_type, generated_at)
    prepared_datasets = {}
    model_variant_reports = {}
    walk_forward_comparison_payloads = {}
    prediction_history_records = []
    data_fingerprint = None

    print(f"\n--- {loto_type.upper()} 確率モデル学習 ＆ 信用度評価 ---")
    for variant_name in args.evaluation_model_variants:
        dataset = prepare_dataset(df, loto_type, variant_name)
        prepared_datasets[variant_name] = dataset
        raw_features = dataset["raw_features"]
        targets = dataset["targets"]

        if data_fingerprint is None:
            data_fingerprint = build_data_fingerprint(
                processed_path=df_path,
                raw_path=raw_path if os.path.exists(raw_path) else None,
                row_count=len(df),
                sample_count=len(targets),
                latest_draw_id=latest_draw_id,
                preprocessing_version=PREPROCESSING_VERSION,
            )

        if len(targets) <= args.walk_forward_test_window:
            raise ValueError(f"[{loto_type}] データ量が不足しています。samples={len(targets)}")

        print(f"  [{variant_name}] {get_model_variant_label(variant_name)}")
        variant_legacy_holdout = None
        if args.skip_legacy_holdout:
            print(f"    - [{variant_name}] legacy holdout をスキップします。")
        else:
            holdout_split = max(int(len(targets) * 0.8), args.walk_forward_test_window)
            holdout_split = min(holdout_split, len(targets) - args.walk_forward_test_window)
            (
                variant_legacy_holdout,
                _,
                _,
                _,
                _,
                _,
                legacy_history_records,
                _,
            ) = evaluate_split(
                df=df,
                raw_features=raw_features,
                targets=targets,
                loto_type=loto_type,
                model_variant=variant_name,
                pick_count=config["pick_count"],
                max_num=config["max_num"],
                train_end=holdout_split,
                test_end=len(targets),
                epochs=args.eval_epochs,
                batch_size=args.batch_size,
                patience=args.patience,
                evaluation_calibration_methods=args.evaluation_calibration_methods,
                calibration_fraction=args.calibration_fraction,
                calibration_min_samples=args.calibration_min_samples,
                calibration_max_samples=args.calibration_max_samples,
                calibration_num_bins=args.calibration_num_bins,
                evaluation_mode="legacy_holdout",
            )
            prediction_history_records.extend(legacy_history_records)

        variant_walk_forward = None
        if args.skip_walk_forward:
            print(f"    - [{variant_name}] walk-forward をスキップします。")
        else:
            variant_walk_forward, walk_forward_history_records, comparison_payload = evaluate_walk_forward(
                df=df,
                raw_features=raw_features,
                targets=targets,
                loto_type=loto_type,
                model_variant=variant_name,
                pick_count=config["pick_count"],
                max_num=config["max_num"],
                initial_train_fraction=args.initial_train_fraction,
                test_window=args.walk_forward_test_window,
                max_folds=args.walk_forward_folds,
                epochs=args.eval_epochs,
                batch_size=args.batch_size,
                patience=args.patience,
                evaluation_calibration_methods=args.evaluation_calibration_methods,
                calibration_fraction=args.calibration_fraction,
                calibration_min_samples=args.calibration_min_samples,
                calibration_max_samples=args.calibration_max_samples,
                calibration_num_bins=args.calibration_num_bins,
            )
            walk_forward_comparison_payloads[variant_name] = comparison_payload
            prediction_history_records.extend(walk_forward_history_records)

        model_variant_reports[variant_name] = build_model_variant_payload(
            variant_name=variant_name,
            dataset=dataset,
            legacy_holdout=variant_legacy_holdout,
            walk_forward=variant_walk_forward,
        )

    calibration_selections = {}
    for variant_name, payload in model_variant_reports.items():
        calibration_selection = build_calibration_selection_report(
            variant_name=variant_name,
            legacy_holdout=payload.get("legacy_holdout"),
            walk_forward=payload.get("walk_forward"),
            evaluation_calibration_methods=args.evaluation_calibration_methods,
            ece_threshold=args.ece_threshold,
        )
        calibration_selections[variant_name] = calibration_selection
        payload["calibration_selection"] = calibration_selection
        payload["selected_evaluation_calibration_method"] = calibration_selection.get("recommended_method")
        payload["selected_evaluation_metrics"] = calibration_selection.get("recommended_metrics")

    production_dataset = prepared_datasets[args.model_variant]
    raw_features = production_dataset["raw_features"]
    targets = production_dataset["targets"]
    feature_cols = production_dataset["feature_cols"]
    training_context = build_training_context(
        args=args,
        loto_type=loto_type,
        feature_cols=feature_cols,
        pick_count=config["pick_count"],
        max_num=config["max_num"],
        dataset_metadata=production_dataset["dataset_metadata"],
    )
    primary_variant_payload = model_variant_reports[args.model_variant]
    legacy_holdout = primary_variant_payload["legacy_holdout"]
    walk_forward = primary_variant_payload["walk_forward"]
    compat_model_metrics, compat_static_baselines, compat_online_baselines = build_compat_report_sections(
        legacy_holdout, walk_forward
    )
    statistical_tests = None
    if not args.skip_walk_forward:
        statistical_tests = build_statistical_tests(
            model_variant_reports=model_variant_reports,
            comparison_payloads=walk_forward_comparison_payloads,
            calibration_selections=calibration_selections,
            alpha=args.statistical_alpha,
            bootstrap_samples=args.bootstrap_samples,
            permutation_samples=args.permutation_samples,
            seed=args.seed,
        )
    decision_summary = build_adoption_decision(
        model_variant_reports=model_variant_reports,
        calibration_selections=calibration_selections,
        statistical_tests=statistical_tests,
        production_variant=args.model_variant,
        production_calibration_method=args.saved_calibration_method,
        alpha=args.statistical_alpha,
        ece_threshold=args.ece_threshold,
    )
    eval_report_path = os.path.join(DATA_DIR, f"eval_report_{loto_type}.json")
    prediction_history_path = os.path.join(DATA_DIR, f"prediction_history_{loto_type}.json")
    save_json(
        prediction_history_path,
        build_prediction_history_artifact(
            loto_type=loto_type,
            records=prediction_history_records,
            generated_at=generated_at,
            bundle_id=bundle_id,
        ),
    )
    feature_cols_path = os.path.join(DATA_DIR, f"{loto_type}_feature_cols.json")
    model_path = os.path.join(MODEL_DIR, f"{loto_type}_prob.keras")
    scaler_path = os.path.join(MODEL_DIR, f"{loto_type}_scaler.pkl")

    if args.skip_final_train:
        print("  [3/4] 本番用モデル学習をスキップし、既存成果物を再利用または雛形を保存します...")
    else:
        print("  [3/4] 本番用モデルと scaler を保存中...")
    final_artifact_status = persist_final_artifacts(
        loto_type=loto_type,
        raw_features=raw_features,
        targets=targets,
        feature_cols=feature_cols,
        max_num=config["max_num"],
        epochs=args.final_epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        skip_final_train=args.skip_final_train,
        saved_calibration_method=args.saved_calibration_method,
        calibration_fraction=args.calibration_fraction,
        calibration_min_samples=args.calibration_min_samples,
        calibration_max_samples=args.calibration_max_samples,
    )
    training_context["saved_calibration_method"] = final_artifact_status.get(
        "effective_calibration_method",
        NO_CALIBRATION_METHOD,
    )
    training_context["saved_calibration_method_label"] = get_calibration_method_label(training_context["saved_calibration_method"])
    training_context["calibration_enabled"] = training_context["saved_calibration_method"] != NO_CALIBRATION_METHOD
    training_context["calibration_artifact_path"] = final_artifact_status.get("calibration_artifact_path")
    training_context["selected_evaluation_calibration_method"] = (
        (primary_variant_payload.get("calibration_selection") or {}).get("recommended_method", NO_CALIBRATION_METHOD)
    )
    training_context["selected_evaluation_calibration_method_label"] = get_calibration_method_label(
        training_context["selected_evaluation_calibration_method"]
    )
    model_variant_reports[args.model_variant]["final_artifact_status"] = final_artifact_status

    report = {
        "schema_version": EVAL_REPORT_SCHEMA_VERSION,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "bundle_id": bundle_id,
        "generated_at": generated_at,
        "loto_type": loto_type,
        "lookback_window": LOOKBACK_WINDOW,
        "data_fingerprint": data_fingerprint,
        "training_context": training_context,
        "runtime_environment": runtime_environment,
        "run_options": {
            "preset": args.preset,
            "seed": args.seed,
            "model_variant": args.model_variant,
            "evaluation_model_variants": args.evaluation_model_variants,
            "saved_calibration_method": args.saved_calibration_method,
            "evaluation_calibration_methods": args.evaluation_calibration_methods,
            "skip_final_train": args.skip_final_train,
            "skip_legacy_holdout": args.skip_legacy_holdout,
            "skip_walk_forward": args.skip_walk_forward,
            "walk_forward_folds": args.walk_forward_folds,
            "eval_epochs": args.eval_epochs,
            "final_epochs": args.final_epochs,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "bootstrap_samples": args.bootstrap_samples,
            "permutation_samples": args.permutation_samples,
            "statistical_alpha": args.statistical_alpha,
            "calibration_fraction": args.calibration_fraction,
            "calibration_min_samples": args.calibration_min_samples,
            "calibration_max_samples": args.calibration_max_samples,
            "calibration_num_bins": args.calibration_num_bins,
            "ece_threshold": args.ece_threshold,
        },
        "calibration": {
            "saved_calibration_method_requested": args.saved_calibration_method,
            "saved_calibration_method": training_context["saved_calibration_method"],
            "evaluation_calibration_methods": args.evaluation_calibration_methods,
            "calibration_artifact_path": final_artifact_status.get("calibration_artifact_path"),
        },
        "Model (LSTM)": compat_model_metrics,
        "Baselines": compat_static_baselines,
        "Online Baselines": compat_online_baselines,
        "legacy_holdout": legacy_holdout,
        "walk_forward": walk_forward,
        "model_variants": model_variant_reports,
        "calibration_evaluation": calibration_selections,
        "statistical_tests": statistical_tests,
        "decision_summary": decision_summary,
    }
    save_json(eval_report_path, report)

    print("  [4/4] manifest を生成中...")
    manifest = build_manifest(
        loto_type=loto_type,
        df=df,
        legacy_holdout=legacy_holdout,
        walk_forward_report=walk_forward,
        model_variant_reports=model_variant_reports,
        statistical_tests=statistical_tests,
        decision_summary=decision_summary,
        eval_report_path=eval_report_path,
        final_artifact_status=final_artifact_status,
        prediction_history_path=prediction_history_path,
        prediction_history_rows=len(prediction_history_records),
        generated_at=generated_at,
        bundle_id=bundle_id,
        feature_cols_path=feature_cols_path,
        model_path=model_path,
        scaler_path=scaler_path,
        data_fingerprint=data_fingerprint,
        training_context=training_context,
        runtime_environment=runtime_environment,
    )
    save_json(os.path.join(DATA_DIR, f"manifest_{loto_type}.json"), manifest)

    primary_metrics = compat_model_metrics
    print(
        "✅ 完了 "
        f"(LogLoss: {primary_metrics['logloss']:.4f}, "
        f"Top-k: {primary_metrics['mean_overlap_top_k']:.2f}, "
        f"saved_calibration={training_context['saved_calibration_method']}, "
        f"final_artifact_status: {final_artifact_status['status']})"
    )
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="Leak-free training with legacy holdout and walk-forward evaluation.")
    parser.add_argument("--loto_type", choices=sorted(LOTO_CONFIG.keys()), help="対象の宝くじ種類を1つに絞る")
    parser.add_argument("--preset", choices=sorted(PRESET_CONFIGS.keys()), default="default")
    parser.add_argument("--model_variant", choices=sorted(MODEL_VARIANT_CHOICES), default=DEFAULT_MODEL_VARIANT)
    parser.add_argument(
        "--evaluation_model_variants",
        default="legacy,multihot",
        help="評価対象 variant をカンマ区切りで指定 (例: legacy,multihot)",
    )
    parser.add_argument(
        "--saved_calibration_method",
        choices=sorted(CALIBRATION_METHOD_CHOICES),
        default=DEFAULT_SAVED_CALIBRATION_METHOD,
    )
    parser.add_argument(
        "--evaluation_calibration_methods",
        default=DEFAULT_EVALUATION_CALIBRATION_METHODS,
        help="評価対象 calibration method をカンマ区切りで指定 (例: none,temperature,isotonic)",
    )
    parser.add_argument("--initial_train_fraction", type=float, default=DEFAULT_INITIAL_TRAIN_FRACTION)
    parser.add_argument("--walk_forward_test_window", type=int, default=DEFAULT_WALK_FORWARD_TEST_WINDOW)
    parser.add_argument("--walk_forward_folds", type=int, default=None)
    parser.add_argument("--eval_epochs", type=int, default=None)
    parser.add_argument("--final_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--bootstrap_samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--permutation_samples", type=int, default=DEFAULT_PERMUTATION_SAMPLES)
    parser.add_argument("--statistical_alpha", type=float, default=DEFAULT_STATISTICAL_ALPHA)
    parser.add_argument("--calibration_fraction", type=float, default=DEFAULT_CALIBRATION_FRACTION)
    parser.add_argument("--calibration_min_samples", type=int, default=DEFAULT_CALIBRATION_MIN_SAMPLES)
    parser.add_argument("--calibration_max_samples", type=int, default=DEFAULT_CALIBRATION_MAX_SAMPLES)
    parser.add_argument("--calibration_num_bins", type=int, default=DEFAULT_CALIBRATION_NUM_BINS)
    parser.add_argument("--ece_threshold", type=float, default=DEFAULT_ECE_THRESHOLD)
    parser.add_argument("--skip_final_train", action="store_true")
    parser.add_argument("--skip_legacy_holdout", action="store_true")
    parser.add_argument("--skip_walk_forward", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main():
    args = apply_preset(parse_args())
    args.model_variant = resolve_model_variant(args.model_variant)
    args.evaluation_model_variants = resolve_evaluation_model_variants(args)
    args.saved_calibration_method = resolve_calibration_method(args.saved_calibration_method)
    args.evaluation_calibration_methods = resolve_evaluation_calibration_methods(
        args.saved_calibration_method,
        args.evaluation_calibration_methods,
    )
    if args.skip_legacy_holdout and args.skip_walk_forward:
        raise SystemExit("legacy_holdout と walk_forward を同時に skip することはできません。")
    set_reproducible_seed(args.seed)

    loto_types = [args.loto_type] if args.loto_type else list(LOTO_CONFIG.keys())
    failed_types = []
    processed = 0

    for loto_type in loto_types:
        try:
            if train_for_type(loto_type, args):
                processed += 1
        except Exception as exc:
            failed_types.append(loto_type)
            print(f"❌ [{loto_type}] 学習に失敗しました: {exc}")

    if failed_types:
        raise SystemExit(1)

    if processed == 0:
        print("処理対象がありませんでした。")


if __name__ == "__main__":
    main()
