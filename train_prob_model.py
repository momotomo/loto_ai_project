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

from config import ARTIFACT_SCHEMA_VERSION, LOOKBACK_WINDOW, LOTO_CONFIG

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
        "eval_epochs": 1,
        "final_epochs": 1,
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
    vectors = []
    for nums in targets:
        vec = np.zeros(max_num, dtype=np.float32)
        for n in nums:
            if 1 <= int(n) <= max_num:
                vec[int(n) - 1] = 1.0
        vectors.append(vec)
    return np.array(vectors, dtype=np.float32)


def sanitize_probabilities(preds):
    return np.clip(np.asarray(preds, dtype=np.float32), EPSILON, 1.0 - EPSILON)


def calculate_calibration(preds, targets, num_bins=10):
    preds_flat = sanitize_probabilities(preds).reshape(-1)
    targets_flat = np.asarray(targets, dtype=np.float32).reshape(-1)
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    calibration = []

    for idx in range(num_bins):
        lower = bins[idx]
        upper = bins[idx + 1]
        if idx == num_bins - 1:
            mask = (preds_flat >= lower) & (preds_flat <= upper)
        else:
            mask = (preds_flat >= lower) & (preds_flat < upper)

        count = int(mask.sum())
        calibration.append(
            {
                "bin_index": idx,
                "bin_range": f"{lower:.1f}-{upper:.1f}",
                "count": count,
                "pred_prob": float(preds_flat[mask].mean()) if count else None,
                "true_prob": float(targets_flat[mask].mean()) if count else None,
            }
        )

    return calibration


def calculate_overlap_distribution(preds, targets, k):
    top_k_preds = np.argsort(preds, axis=1)[:, -k:]
    overlaps = []

    for pred_indices, target_vector in zip(top_k_preds, targets):
        true_indices = set(np.flatnonzero(target_vector > 0.5).tolist())
        overlaps.append(len(true_indices & set(pred_indices.tolist())))

    overlap_dist = {str(i): int(overlaps.count(i)) for i in range(k + 1)}
    return overlaps, overlap_dist


def calculate_metrics(preds, targets, k):
    preds = sanitize_probabilities(preds)
    targets = np.asarray(targets, dtype=np.float32)
    logloss = -np.mean(targets * np.log(preds) + (1.0 - targets) * np.log(1.0 - preds))
    brier = np.mean((preds - targets) ** 2)
    overlaps, overlap_dist = calculate_overlap_distribution(preds, targets, k)

    return {
        "logloss": float(logloss),
        "brier": float(brier),
        "mean_overlap_top_k": float(np.mean(overlaps)) if overlaps else 0.0,
        "overlap_dist": overlap_dist,
        "calibration": calculate_calibration(preds, targets),
    }


def target_vector_to_numbers(target_vector):
    indices = np.flatnonzero(np.asarray(target_vector, dtype=np.float32) > 0.5)
    return [int(index) + 1 for index in indices.tolist()]


def build_prediction_history_records(
    df,
    preds,
    targets,
    loto_type,
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
        "schema_version": 1,
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
            "mean_overlap_top_k": {"mean": metrics["mean_overlap_top_k"], "variance": 0.0},
        },
        "overlap_dist_total": metrics["overlap_dist"],
        "calibration": metrics["calibration"],
    }


def summary_entry_to_metrics(summary_entry):
    metric_summary = summary_entry["metric_summary"]
    return {
        "logloss": metric_summary["logloss"]["mean"],
        "brier": metric_summary["brier"]["mean"],
        "mean_overlap_top_k": metric_summary["mean_overlap_top_k"]["mean"],
        "overlap_dist": summary_entry["overlap_dist_total"],
        "calibration": summary_entry["calibration"],
    }


def summarize_metric_series(values):
    array = np.asarray(values, dtype=np.float32)
    return {
        "mean": float(array.mean()) if array.size else None,
        "variance": float(array.var()) if array.size else None,
    }


def build_prob_model(input_shape, max_num):
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
    model.compile(optimizer="adam", loss="binary_crossentropy")
    return model


def fit_prob_model(X_train, y_train, max_num, epochs, batch_size, patience):
    tf.keras.backend.clear_session()
    model = build_prob_model((X_train.shape[1], X_train.shape[2]), max_num)
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
        overlap_dist = calculate_overlap_distribution(merged_preds, merged_targets, pick_count)[1]
        aggregate[name] = {
            "fold_count": len(fold_metrics),
            "test_samples": int(len(merged_targets)),
            "metric_summary": {
                "logloss": summarize_metric_series([item["logloss"] for item in fold_metrics]),
                "brier": summarize_metric_series([item["brier"] for item in fold_metrics]),
                "mean_overlap_top_k": summarize_metric_series([item["mean_overlap_top_k"] for item in fold_metrics]),
            },
            "overlap_dist_total": overlap_dist,
            "calibration": calculate_calibration(merged_preds, merged_targets),
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


def prepare_dataset(df, pick_count, max_num):
    target_cols = [f"num{i + 1}" for i in range(pick_count)]
    features_df = df.drop(["draw_id", "date"], axis=1)
    other_cols = [column for column in features_df.columns if column not in target_cols]
    feature_cols = target_cols + other_cols
    features_df = features_df[feature_cols]

    raw_features = features_df.to_numpy(dtype=np.float32)
    target_numbers = df[target_cols].to_numpy(dtype=np.int32)
    targets = create_multi_hot(target_numbers[LOOKBACK_WINDOW:], max_num)

    return {
        "feature_cols": feature_cols,
        "raw_features": raw_features,
        "targets": targets,
    }


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
    pick_count,
    max_num,
    train_end,
    test_end,
    epochs,
    batch_size,
    patience,
    evaluation_mode,
    fold_index=None,
):
    scaler = fit_scaler_for_split(raw_features, train_end)
    required_rows = LOOKBACK_WINDOW + test_end
    scaled_features = scaler.transform(raw_features[:required_rows])

    X_train, y_train = build_samples_from_scaled_features(scaled_features, targets, 0, train_end)
    X_test, y_test = build_samples_from_scaled_features(scaled_features, targets, train_end, test_end)

    model = fit_prob_model(X_train, y_train, max_num, epochs, batch_size, patience)
    preds_test = sanitize_probabilities(model(tf.convert_to_tensor(X_test), training=False).numpy())
    model_metrics = calculate_metrics(preds_test, y_test, pick_count)
    baseline_outputs = evaluate_baselines(y_train, y_test, max_num, pick_count)
    history_records = build_prediction_history_records(
        df=df,
        preds=preds_test,
        targets=y_test,
        loto_type=loto_type,
        pick_count=pick_count,
        max_num=max_num,
        sample_start=train_end,
        evaluation_mode=evaluation_mode,
        fold_index=fold_index,
    )

    split_report = {
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "train_range": build_sample_range_metadata(df, 0, train_end),
        "test_range": build_sample_range_metadata(df, train_end, test_end),
        "model": model_metrics,
        "static_baselines": {name: payload["metrics"] for name, payload in baseline_outputs["static_baselines"].items()},
        "online_baselines": {name: payload["metrics"] for name, payload in baseline_outputs["online_baselines"].items()},
    }

    return split_report, scaler, model, preds_test, y_test, baseline_outputs, history_records


def evaluate_walk_forward(
    df,
    raw_features,
    targets,
    loto_type,
    pick_count,
    max_num,
    initial_train_fraction,
    test_window,
    max_folds,
    epochs,
    batch_size,
    patience,
):
    fold_starts = select_walk_forward_starts(len(targets), initial_train_fraction, test_window, max_folds)
    if not fold_starts:
        raise ValueError("walk-forward に必要な fold を作成できませんでした。")
    folds = []
    history_records = []

    metrics_by_name = {"model": []}
    preds_by_name = {"model": []}
    targets_by_name = {"model": []}

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
        split_report, _, model, preds_test, y_test, baseline_outputs, split_history_records = evaluate_split(
            df=df,
            raw_features=raw_features,
            targets=targets,
            loto_type=loto_type,
            pick_count=pick_count,
            max_num=max_num,
            train_end=train_end,
            test_end=test_end,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            evaluation_mode="walk_forward",
            fold_index=fold_index,
        )
        split_report["fold"] = fold_index
        folds.append(split_report)
        history_records.extend(split_history_records)

        metrics_by_name["model"].append(split_report["model"])
        preds_by_name["model"].append(preds_test)
        targets_by_name["model"].append(y_test)

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
        "static_baselines": {},
        "online_baselines": {},
    }

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
    )


def train_final_model(raw_features, targets, max_num, epochs, batch_size, patience):
    scaler = MinMaxScaler()
    scaler.fit(raw_features)
    scaled_features = scaler.transform(raw_features)
    X_all, y_all = build_samples_from_scaled_features(scaled_features, targets, 0, len(targets))
    model = fit_prob_model(X_all, y_all, max_num, epochs, batch_size, patience)
    return scaler, model


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


def persist_final_artifacts(loto_type, raw_features, targets, feature_cols, max_num, epochs, batch_size, patience, skip_final_train):
    model_path = os.path.join(MODEL_DIR, f"{loto_type}_prob.keras")
    scaler_path = os.path.join(MODEL_DIR, f"{loto_type}_scaler.pkl")
    existing_feature_cols = load_existing_feature_cols(loto_type)
    feature_cols_match = existing_feature_cols == feature_cols if existing_feature_cols is not None else False
    artifacts_exist = os.path.exists(model_path) and os.path.exists(scaler_path)

    save_feature_cols(loto_type, feature_cols)

    if skip_final_train and artifacts_exist and feature_cols_match:
        return "reused_existing_artifacts"

    effective_epochs = 0 if skip_final_train else epochs
    final_scaler, final_model = train_final_model(
        raw_features=raw_features,
        targets=targets,
        max_num=max_num,
        epochs=effective_epochs,
        batch_size=batch_size,
        patience=patience,
    )
    final_model.save(model_path)
    with open(scaler_path, "wb") as handle:
        pickle.dump(final_scaler, handle)
    del final_model
    tf.keras.backend.clear_session()

    if skip_final_train:
        return "initialized_without_training"
    return "trained_on_full_data"


def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


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


def build_manifest(
    loto_type,
    df,
    legacy_holdout,
    walk_forward_report,
    eval_report_path,
    final_artifact_status,
    prediction_history_path,
    prediction_history_rows,
    generated_at,
    bundle_id,
):
    primary_evaluation = select_primary_evaluation(legacy_holdout, walk_forward_report)
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
        "mean_overlap_top_k_mean": model_summary["mean_overlap_top_k"]["mean"],
    }
    metrics_summary = {
        "evaluation_source": primary_evaluation["source"],
        "fold_count": primary_evaluation["fold_count"],
        "test_window": primary_evaluation["test_window"],
        "primary_model": primary_model,
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
        "final_artifact_status": final_artifact_status,
    }

    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "bundle_id": bundle_id,
        "generated_at": generated_at,
        "loto_type": loto_type,
        "latest_draw_id": int(df.iloc[-1]["draw_id"]),
        "train_range": build_range_metadata(df, 0, len(df) - 1),
        "metrics_summary": metrics_summary,
        "artifacts": {
            "eval_report": eval_report_path,
            "prediction_history": prediction_history_path,
            "model": os.path.join(MODEL_DIR, f"{loto_type}_prob.keras"),
            "scaler": os.path.join(MODEL_DIR, f"{loto_type}_scaler.pkl"),
            "feature_cols": os.path.join(DATA_DIR, f"{loto_type}_feature_cols.json"),
        },
        "prediction_history_path": prediction_history_path,
        "prediction_history_rows": int(prediction_history_rows),
        "git_commit": get_git_commit(),
    }


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def apply_preset(args):
    preset_values = PRESET_CONFIGS[args.preset]
    for field, value in preset_values.items():
        if getattr(args, field) is None:
            setattr(args, field, value)
    return args


def train_for_type(loto_type, args):
    config = LOTO_CONFIG[loto_type]
    df_path = os.path.join(DATA_DIR, f"{loto_type}_processed.csv")
    if not os.path.exists(df_path):
        print(f"[{loto_type}] データなし。スキップします。")
        return False

    df = pd.read_csv(df_path).sort_values("draw_id").reset_index(drop=True)
    dataset = prepare_dataset(df, config["pick_count"], config["max_num"])
    raw_features = dataset["raw_features"]
    targets = dataset["targets"]
    feature_cols = dataset["feature_cols"]

    if len(targets) <= args.walk_forward_test_window:
        raise ValueError(f"[{loto_type}] データ量が不足しています。samples={len(targets)}")

    generated_at = datetime.now(timezone.utc).isoformat()
    bundle_id = build_bundle_id(loto_type, generated_at)

    print(f"\n--- {loto_type.upper()} 確率モデル学習 ＆ 信用度評価 ---")
    legacy_holdout = None
    prediction_history_records = []
    if args.skip_legacy_holdout:
        print("  [1/4] legacy holdout をスキップします。")
    else:
        print("  [1/4] legacy holdout を計算中...")
        holdout_split = max(int(len(targets) * 0.8), args.walk_forward_test_window)
        holdout_split = min(holdout_split, len(targets) - args.walk_forward_test_window)
        legacy_holdout, _, _, _, _, _, legacy_history_records = evaluate_split(
            df=df,
            raw_features=raw_features,
            targets=targets,
            loto_type=loto_type,
            pick_count=config["pick_count"],
            max_num=config["max_num"],
            train_end=holdout_split,
            test_end=len(targets),
            epochs=args.eval_epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            evaluation_mode="legacy_holdout",
        )
        prediction_history_records.extend(legacy_history_records)

    walk_forward = None
    if args.skip_walk_forward:
        print("  [2/4] walk-forward をスキップします。")
    else:
        print("  [2/4] walk-forward を計算中...")
        walk_forward, walk_forward_history_records = evaluate_walk_forward(
            df=df,
            raw_features=raw_features,
            targets=targets,
            loto_type=loto_type,
            pick_count=config["pick_count"],
            max_num=config["max_num"],
            initial_train_fraction=args.initial_train_fraction,
            test_window=args.walk_forward_test_window,
            max_folds=args.walk_forward_folds,
            epochs=args.eval_epochs,
            batch_size=args.batch_size,
            patience=args.patience,
        )
        prediction_history_records.extend(walk_forward_history_records)

    compat_model_metrics, compat_static_baselines, compat_online_baselines = build_compat_report_sections(
        legacy_holdout, walk_forward
    )

    report = {
        "schema_version": 2,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "bundle_id": bundle_id,
        "generated_at": generated_at,
        "loto_type": loto_type,
        "lookback_window": LOOKBACK_WINDOW,
        "run_options": {
            "preset": args.preset,
            "skip_final_train": args.skip_final_train,
            "skip_legacy_holdout": args.skip_legacy_holdout,
            "skip_walk_forward": args.skip_walk_forward,
            "walk_forward_folds": args.walk_forward_folds,
            "eval_epochs": args.eval_epochs,
            "final_epochs": args.final_epochs,
            "batch_size": args.batch_size,
            "patience": args.patience,
        },
        "Model (LSTM)": compat_model_metrics,
        "Baselines": compat_static_baselines,
        "Online Baselines": compat_online_baselines,
        "legacy_holdout": legacy_holdout,
        "walk_forward": walk_forward,
    }
    eval_report_path = os.path.join(DATA_DIR, f"eval_report_{loto_type}.json")
    save_json(eval_report_path, report)
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
    )

    print("  [4/4] manifest を生成中...")
    manifest = build_manifest(
        loto_type=loto_type,
        df=df,
        legacy_holdout=legacy_holdout,
        walk_forward_report=walk_forward,
        eval_report_path=eval_report_path,
        final_artifact_status=final_artifact_status,
        prediction_history_path=prediction_history_path,
        prediction_history_rows=len(prediction_history_records),
        generated_at=generated_at,
        bundle_id=bundle_id,
    )
    save_json(os.path.join(DATA_DIR, f"manifest_{loto_type}.json"), manifest)

    primary_metrics = compat_model_metrics
    print(
        "✅ 完了 "
        f"(LogLoss: {primary_metrics['logloss']:.4f}, "
        f"Top-k: {primary_metrics['mean_overlap_top_k']:.2f}, "
        f"final_artifact_status: {final_artifact_status})"
    )
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="Leak-free training with legacy holdout and walk-forward evaluation.")
    parser.add_argument("--loto_type", choices=sorted(LOTO_CONFIG.keys()), help="対象の宝くじ種類を1つに絞る")
    parser.add_argument("--preset", choices=sorted(PRESET_CONFIGS.keys()), default="default")
    parser.add_argument("--initial_train_fraction", type=float, default=DEFAULT_INITIAL_TRAIN_FRACTION)
    parser.add_argument("--walk_forward_test_window", type=int, default=DEFAULT_WALK_FORWARD_TEST_WINDOW)
    parser.add_argument("--walk_forward_folds", type=int, default=None)
    parser.add_argument("--eval_epochs", type=int, default=None)
    parser.add_argument("--final_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--skip_final_train", action="store_true")
    parser.add_argument("--skip_legacy_holdout", action="store_true")
    parser.add_argument("--skip_walk_forward", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main():
    args = apply_preset(parse_args())
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
