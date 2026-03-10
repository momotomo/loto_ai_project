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

from config import LOOKBACK_WINDOW, LOTO_CONFIG

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

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


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


def evaluate_split(df, raw_features, targets, pick_count, max_num, train_end, test_end, epochs, batch_size, patience):
    scaler = fit_scaler_for_split(raw_features, train_end)
    required_rows = LOOKBACK_WINDOW + test_end
    scaled_features = scaler.transform(raw_features[:required_rows])

    X_train, y_train = build_samples_from_scaled_features(scaled_features, targets, 0, train_end)
    X_test, y_test = build_samples_from_scaled_features(scaled_features, targets, train_end, test_end)

    model = fit_prob_model(X_train, y_train, max_num, epochs, batch_size, patience)
    preds_test = sanitize_probabilities(model(tf.convert_to_tensor(X_test), training=False).numpy())
    model_metrics = calculate_metrics(preds_test, y_test, pick_count)
    baseline_outputs = evaluate_baselines(y_train, y_test, max_num, pick_count)

    split_report = {
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "train_range": build_sample_range_metadata(df, 0, train_end),
        "test_range": build_sample_range_metadata(df, train_end, test_end),
        "model": model_metrics,
        "static_baselines": {name: payload["metrics"] for name, payload in baseline_outputs["static_baselines"].items()},
        "online_baselines": {name: payload["metrics"] for name, payload in baseline_outputs["online_baselines"].items()},
    }

    return split_report, scaler, model, preds_test, y_test, baseline_outputs


def evaluate_walk_forward(df, raw_features, targets, pick_count, max_num, initial_train_fraction, test_window, max_folds, epochs, batch_size, patience):
    fold_starts = select_walk_forward_starts(len(targets), initial_train_fraction, test_window, max_folds)
    if not fold_starts:
        raise ValueError("walk-forward に必要な fold を作成できませんでした。")
    folds = []

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
        split_report, _, model, preds_test, y_test, baseline_outputs = evaluate_split(
            df=df,
            raw_features=raw_features,
            targets=targets,
            pick_count=pick_count,
            max_num=max_num,
            train_end=train_end,
            test_end=test_end,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
        )
        split_report["fold"] = fold_index
        folds.append(split_report)

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

    return {
        "settings": {
            "initial_train_fraction": initial_train_fraction,
            "test_window": test_window,
            "max_folds": max_folds,
            "fold_selection_policy": "expanding-window rolling-origin, evenly sampled after the initial train period",
        },
        "folds": folds,
        "aggregate": aggregate,
    }


def train_final_model(raw_features, targets, max_num, epochs, batch_size, patience):
    scaler = MinMaxScaler()
    scaler.fit(raw_features)
    scaled_features = scaler.transform(raw_features)
    X_all, y_all = build_samples_from_scaled_features(scaled_features, targets, 0, len(targets))
    model = fit_prob_model(X_all, y_all, max_num, epochs, batch_size, patience)
    return scaler, model


def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def build_manifest(loto_type, df, walk_forward_report, eval_report_path):
    model_summary = walk_forward_report["aggregate"]["model"]["metric_summary"]
    static_summary = walk_forward_report["aggregate"]["static_baselines"]
    best_static_name = min(
        static_summary,
        key=lambda name: static_summary[name]["metric_summary"]["logloss"]["mean"],
    )

    best_static = static_summary[best_static_name]["metric_summary"]
    metrics_summary = {
        "fold_count": len(walk_forward_report["folds"]),
        "test_window": walk_forward_report["settings"]["test_window"],
        "walk_forward_model": {
            "logloss_mean": model_summary["logloss"]["mean"],
            "brier_mean": model_summary["brier"]["mean"],
            "mean_overlap_top_k_mean": model_summary["mean_overlap_top_k"]["mean"],
        },
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
    }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "loto_type": loto_type,
        "latest_draw_id": int(df.iloc[-1]["draw_id"]),
        "train_range": build_range_metadata(df, 0, len(df) - 1),
        "metrics_summary": metrics_summary,
        "artifacts": {
            "eval_report": eval_report_path,
            "model": os.path.join(MODEL_DIR, f"{loto_type}_prob.keras"),
            "scaler": os.path.join(MODEL_DIR, f"{loto_type}_scaler.pkl"),
            "feature_cols": os.path.join(DATA_DIR, f"{loto_type}_feature_cols.json"),
        },
        "git_commit": get_git_commit(),
    }


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


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

    print(f"\n--- {loto_type.upper()} 確率モデル学習 ＆ 信用度評価 ---")
    print("  [1/4] legacy holdout を計算中...")
    holdout_split = max(int(len(targets) * 0.8), args.walk_forward_test_window)
    holdout_split = min(holdout_split, len(targets) - args.walk_forward_test_window)
    legacy_holdout, _, _, _, _, _ = evaluate_split(
        df=df,
        raw_features=raw_features,
        targets=targets,
        pick_count=config["pick_count"],
        max_num=config["max_num"],
        train_end=holdout_split,
        test_end=len(targets),
        epochs=args.eval_epochs,
        batch_size=args.batch_size,
        patience=args.patience,
    )

    print("  [2/4] walk-forward を計算中...")
    walk_forward = evaluate_walk_forward(
        df=df,
        raw_features=raw_features,
        targets=targets,
        pick_count=config["pick_count"],
        max_num=config["max_num"],
        initial_train_fraction=args.initial_train_fraction,
        test_window=args.walk_forward_test_window,
        max_folds=args.walk_forward_folds,
        epochs=args.eval_epochs,
        batch_size=args.batch_size,
        patience=args.patience,
    )

    report = {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "loto_type": loto_type,
        "lookback_window": LOOKBACK_WINDOW,
        "Model (LSTM)": legacy_holdout["model"],
        "Baselines": legacy_holdout["static_baselines"],
        "Online Baselines": legacy_holdout["online_baselines"],
        "legacy_holdout": legacy_holdout,
        "walk_forward": walk_forward,
    }
    eval_report_path = os.path.join(DATA_DIR, f"eval_report_{loto_type}.json")
    save_json(eval_report_path, report)

    print("  [3/4] 本番用モデルと scaler を保存中...")
    final_scaler, final_model = train_final_model(
        raw_features=raw_features,
        targets=targets,
        max_num=config["max_num"],
        epochs=args.final_epochs,
        batch_size=args.batch_size,
        patience=args.patience,
    )
    final_model.save(os.path.join(MODEL_DIR, f"{loto_type}_prob.keras"))
    with open(os.path.join(MODEL_DIR, f"{loto_type}_scaler.pkl"), "wb") as handle:
        pickle.dump(final_scaler, handle)

    save_json(os.path.join(DATA_DIR, f"{loto_type}_feature_cols.json"), feature_cols)
    save_json(os.path.join(MODEL_DIR, f"{loto_type}_feature_cols.json"), feature_cols)

    print("  [4/4] manifest を生成中...")
    manifest = build_manifest(loto_type, df, walk_forward, eval_report_path)
    save_json(os.path.join(DATA_DIR, f"manifest_{loto_type}.json"), manifest)

    wf_model = walk_forward["aggregate"]["model"]["metric_summary"]
    print(
        "✅ 完了 "
        f"(WF LogLoss mean: {wf_model['logloss']['mean']:.4f}, "
        f"WF Top-k mean: {wf_model['mean_overlap_top_k']['mean']:.2f})"
    )
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="Leak-free training with legacy holdout and walk-forward evaluation.")
    parser.add_argument("--loto_type", choices=sorted(LOTO_CONFIG.keys()), help="対象の宝くじ種類を1つに絞る")
    parser.add_argument("--initial_train_fraction", type=float, default=DEFAULT_INITIAL_TRAIN_FRACTION)
    parser.add_argument("--walk_forward_test_window", type=int, default=DEFAULT_WALK_FORWARD_TEST_WINDOW)
    parser.add_argument("--walk_forward_folds", type=int, default=DEFAULT_MAX_WALK_FORWARD_FOLDS)
    parser.add_argument("--eval_epochs", type=int, default=DEFAULT_EVAL_EPOCHS)
    parser.add_argument("--final_epochs", type=int, default=DEFAULT_FINAL_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main():
    args = parse_args()
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
