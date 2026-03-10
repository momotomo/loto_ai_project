import ast
import json
import os
import pickle
import re
import shutil
import tempfile
import warnings
from datetime import datetime

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model

from config import ARTIFACT_SCHEMA_VERSION, LOOKBACK_WINDOW, LOTO_CONFIG, generate_valid_sample

# --- Mac環境安定化設定 ---
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

st.set_page_config(page_title="宝くじAI 確率予測ダッシュボード", page_icon="🎲", layout="wide")

BASELINE_LABELS = {
    "uniform": "Uniform",
    "frequency": "Frequency (static)",
    "gap": "Gap (static)",
    "frequency_online": "Frequency (online)",
    "gap_online": "Gap (online)",
}
HISTORY_LIST_COLUMNS = [
    "actual_numbers",
    "predicted_top_k",
    "predicted_top_k_hit_numbers",
    "top_probability_numbers",
    "top_probability_scores",
]
KERNEL_REF_EXAMPLE = "username/my-loto-kernel"
SYNC_NOTICE_STATE_KEY = "last_kaggle_sync_notice"
LOCAL_CLEAN_NOTICE_STATE_KEY = "last_local_bundle_cleanup_notice"
SYNC_REQUIRED_BUNDLE_FILES = [
    "{loto_type}_processed.csv",
    "{loto_type}_feature_cols.json",
    "{loto_type}_prob.keras",
    "{loto_type}_scaler.pkl",
    "manifest_{loto_type}.json",
]
SYNC_OPTIONAL_BUNDLE_FILES = [
    "eval_report_{loto_type}.json",
    "prediction_history_{loto_type}.json",
]
LOCAL_BUNDLE_DELETE_PATTERNS = [
    os.path.join("data", "{loto_type}_processed.csv"),
    os.path.join("data", "{loto_type}_feature_cols.json"),
    os.path.join("data", "manifest_{loto_type}.json"),
    os.path.join("data", "eval_report_{loto_type}.json"),
    os.path.join("data", "prediction_history_{loto_type}.json"),
    os.path.join("data", "prediction_history_{loto_type}.csv"),
    os.path.join("models", "{loto_type}_feature_cols.json"),
    os.path.join("models", "{loto_type}_scaler.pkl"),
    os.path.join("models", "{loto_type}_prob.keras"),
]


def normalize_kernel_ref(value):
    return (value or "").strip()


def validate_kernel_ref(value):
    kernel_ref = normalize_kernel_ref(value)
    if kernel_ref.count("/") != 1:
        return False, kernel_ref, f"Kernel Ref は `owner/kernel-slug` 形式で入力してください。例: `{KERNEL_REF_EXAMPLE}`"

    owner, slug = [part.strip() for part in kernel_ref.split("/", 1)]
    if not owner or not slug or any(" " in part for part in (owner, slug)):
        return False, kernel_ref, f"Kernel Ref は `owner/kernel-slug` 形式で入力してください。例: `{KERNEL_REF_EXAMPLE}`"

    return True, f"{owner}/{slug}", None


def extract_http_status(exc):
    for attr in ("status", "status_code"):
        value = getattr(exc, attr, None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass

    match = re.search(r"\b(403|404|401)\b", str(exc))
    if match:
        return int(match.group(1))
    return None


def build_kaggle_sync_error_message(kernel_ref, exc):
    status = extract_http_status(exc)
    example_line = f"例: {KERNEL_REF_EXAMPLE}"

    if status == 403:
        return (
            "❌ Kaggle 同期に失敗しました (403 Forbidden)\n"
            f"- Kernel Ref: {kernel_ref}\n"
            "- `owner/kernel-slug` 形式か確認してください\n"
            "- `KAGGLE_USERNAME` / `KAGGLE_KEY` がその kernel の所有者、または閲覧権限のあるアカウントか確認してください\n"
            "- private kernel の場合は権限が必要です\n"
            f"- {example_line}"
        )

    if status == 404:
        return (
            "❌ Kaggle 同期に失敗しました (404 Not Found)\n"
            f"- Kernel Ref: {kernel_ref}\n"
            "- Kernel Ref の owner / slug が正しいか確認してください\n"
            "- slug 単体ではなく `owner/kernel-slug` を入力してください\n"
            f"- {example_line}"
        )

    return (
        "❌ Kaggle 同期エラー\n"
        f"- Kernel Ref: {kernel_ref}\n"
        f"- 詳細: {str(exc)}\n"
        "- Kernel Ref、認証情報、Kaggle 側の Output 公開状態を確認してください"
    )


def classify_sync_destination(file_name):
    if file_name.endswith(".csv"):
        return [os.path.join("data", file_name)]
    if file_name.endswith((".keras", ".pkl")):
        return [os.path.join("models", file_name)]
    if file_name.endswith(".json"):
        destinations = [os.path.join("data", file_name)]
        if file_name.endswith("_feature_cols.json"):
            destinations.append(os.path.join("models", file_name))
        return destinations
    return []


def source_preference(file_name, relative_path):
    normalized = relative_path.replace("\\", "/")
    if file_name.endswith(".csv") and normalized.startswith("data/"):
        return 0
    if file_name.endswith((".keras", ".pkl")) and normalized.startswith("models/"):
        return 0
    if file_name.endswith(".json") and normalized.startswith("data/"):
        return 0
    return 1


def build_sync_plan(download_dir):
    selected_files = {}

    for root, _, files in os.walk(download_dir):
        for file_name in files:
            destinations = classify_sync_destination(file_name)
            if not destinations:
                continue

            source_path = os.path.join(root, file_name)
            relative_path = os.path.relpath(source_path, download_dir)
            candidate = {
                "file_name": file_name,
                "source_path": source_path,
                "relative_path": relative_path,
                "destinations": destinations,
            }

            existing = selected_files.get(file_name)
            if existing is None or source_preference(file_name, relative_path) < source_preference(
                file_name, existing["relative_path"]
            ):
                selected_files[file_name] = candidate

    plan = [selected_files[name] for name in sorted(selected_files.keys())]
    return plan, selected_files


def load_selected_json(selected_files, file_name):
    entry = selected_files.get(file_name)
    if not entry:
        return None
    try:
        with open(entry["source_path"], "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def build_bundle_source_names(loto_type, include_optional=False):
    templates = list(SYNC_REQUIRED_BUNDLE_FILES)
    if include_optional:
        templates.extend(SYNC_OPTIONAL_BUNDLE_FILES)
    return {template.format(loto_type=loto_type) for template in templates}


def get_local_bundle_paths(loto_type):
    return [pattern.format(loto_type=loto_type) for pattern in LOCAL_BUNDLE_DELETE_PATTERNS]


def remove_local_artifacts_for_loto(loto_type):
    removed = []
    missing = []

    for path in get_local_bundle_paths(loto_type):
        if os.path.exists(path):
            os.remove(path)
            removed.append(path)
        else:
            missing.append(path)

    return {"removed": removed, "missing": missing}


def format_cleanup_notice(loto_type, cleanup_result):
    return (
        f"{loto_type} のローカル artifact を削除しました。"
        f" removed={len(cleanup_result.get('removed', []))}"
        f", missing={len(cleanup_result.get('missing', []))}"
    )


def validate_staged_manifest(manifest, loto_type):
    if not isinstance(manifest, dict):
        return "manifest 読み込み失敗"
    if manifest.get("loto_type") != loto_type:
        return "manifest の loto_type 不一致"
    if manifest.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        return "artifact_schema_version 不一致"
    if not manifest.get("bundle_id"):
        return "bundle_id 欠落"
    return None


def normalize_loto_targets(raw_targets):
    if isinstance(raw_targets, dict):
        candidates = raw_targets.keys()
    elif isinstance(raw_targets, (list, tuple, set)):
        candidates = raw_targets
    else:
        return []
    return sorted(target for target in candidates if target in LOTO_CONFIG)


def infer_sync_target_loto_types(selected_files):
    summary_payload = load_selected_json(selected_files, "kaggle_run_summary.json")
    if isinstance(summary_payload, dict):
        run_config = summary_payload.get("run_config") if isinstance(summary_payload.get("run_config"), dict) else {}
        targets = normalize_loto_targets(run_config.get("targets"))
        if targets:
            return targets, "kaggle_run_summary.json"
        targets = normalize_loto_targets(summary_payload.get("targets"))
        if targets:
            return targets, "kaggle_run_summary.json"

    run_config_payload = load_selected_json(selected_files, "run_config.json")
    if isinstance(run_config_payload, dict):
        targets = normalize_loto_targets(run_config_payload.get("targets"))
        if targets:
            return targets, "run_config.json"

    manifest_targets = [
        loto_type for loto_type in sorted(LOTO_CONFIG.keys()) if f"manifest_{loto_type}.json" in selected_files
    ]
    if manifest_targets:
        return manifest_targets, "manifest_*.json"

    return None, None


def file_belongs_to_loto(file_name, loto_type):
    exact_matches = {
        f"{loto_type}_raw.csv",
        f"{loto_type}_processed.csv",
        f"{loto_type}_feature_cols.json",
        f"{loto_type}_prob.keras",
        f"{loto_type}_scaler.pkl",
        f"eval_report_{loto_type}.json",
        f"manifest_{loto_type}.json",
        f"prediction_history_{loto_type}.json",
        f"prediction_history_{loto_type}.csv",
    }
    return file_name in exact_matches


def summarize_manifest_sources(selected_files, updated_loto_types):
    summaries = []

    for loto_type in updated_loto_types:
        manifest = load_selected_json(selected_files, f"manifest_{loto_type}.json")
        if not isinstance(manifest, dict):
            continue
        summaries.append(
            f"{loto_type}: generated_at={manifest.get('generated_at', '-')}, "
            f"latest_draw_id={manifest.get('latest_draw_id', '-')}"
        )

    return summaries


def evaluate_sync_plan(plan, selected_files):
    downloaded_names = {item["file_name"] for item in plan}
    inferred_targets, inference_source = infer_sync_target_loto_types(selected_files)
    updated_loto_types = []
    skipped = []
    bundle_details = {}

    for loto_type in sorted(LOTO_CONFIG.keys()):
        expected = build_bundle_source_names(loto_type)
        present_core = sorted(name for name in expected if name in downloaded_names)
        missing_core = sorted(expected - set(present_core))
        related_files = sorted(name for name in downloaded_names if file_belongs_to_loto(name, loto_type))
        manifest = load_selected_json(selected_files, f"manifest_{loto_type}.json")
        manifest_error = validate_staged_manifest(manifest, loto_type) if manifest is not None else "manifest 欠落"
        manifest_bundle_id = manifest.get("bundle_id") if isinstance(manifest, dict) else None
        manifest_generated_at = manifest.get("generated_at") if isinstance(manifest, dict) else None

        bundle_details[loto_type] = {
            "bundle_id": manifest_bundle_id,
            "generated_at": manifest_generated_at,
            "manifest_error": manifest_error,
        }

        if inferred_targets is not None:
            if loto_type in inferred_targets:
                if not related_files:
                    skipped.append((loto_type, "artifact 欠落", []))
                elif manifest_error is None and not missing_core:
                    updated_loto_types.append(loto_type)
                else:
                    reasons = [manifest_error] if manifest_error else []
                    reasons.extend(missing_core)
                    skipped.append((loto_type, "bundle 不完全", reasons))
            elif related_files:
                skipped.append((loto_type, "今回の実行対象外。現行ローカル bundle を維持", []))
            continue

        if not present_core:
            continue
        if manifest_error is None and not missing_core:
            updated_loto_types.append(loto_type)
        else:
            reasons = [manifest_error] if manifest_error else []
            reasons.extend(missing_core)
            skipped.append((loto_type, "bundle 不完全", reasons))

    filtered_plan = []
    for item in plan:
        owning_loto_types = [loto_type for loto_type in LOTO_CONFIG if file_belongs_to_loto(item["file_name"], loto_type)]
        if not owning_loto_types or owning_loto_types[0] in updated_loto_types:
            filtered_plan.append(item)

    summary_lines = [f"更新: {', '.join(updated_loto_types)}"] if updated_loto_types else []
    for loto_type, reason, missing in skipped:
        if missing:
            summary_lines.append(f"スキップ: {loto_type}（{reason}: {', '.join(missing)}）")
        else:
            summary_lines.append(f"スキップ: {loto_type}（{reason}）")

    if updated_loto_types:
        return {
            "ok": True,
            "plan": filtered_plan,
            "updated_loto_types": updated_loto_types,
            "skipped": skipped,
            "summary_lines": summary_lines,
            "target_inference_source": inference_source,
            "bundle_details": bundle_details,
        }

    lines = ["❌ 更新可能な loto_type が見つかりませんでした。"]
    if inference_source:
        lines.append(f"- target inference source: {inference_source}")
    lines.extend(summary_lines)
    lines.append("- Kaggle 側の今回対象と artifact bundle を確認してください。")
    return {
        "ok": False,
        "error_message": "\n".join(lines),
        "summary_lines": summary_lines,
        "target_inference_source": inference_source,
        "bundle_details": bundle_details,
    }


def apply_sync_plan(plan, updated_loto_types):
    staging_dir = tempfile.mkdtemp(prefix="kaggle_sync_stage_")
    staged_files = []
    cleanup_summary = {}

    try:
        for item in plan:
            for destination in item["destinations"]:
                staged_path = os.path.join(staging_dir, destination)
                os.makedirs(os.path.dirname(staged_path), exist_ok=True)
                shutil.copy2(item["source_path"], staged_path)
                staged_files.append((staged_path, destination))

        for loto_type in updated_loto_types:
            cleanup_summary[loto_type] = remove_local_artifacts_for_loto(loto_type)

        for staged_path, destination in staged_files:
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            os.replace(staged_path, destination)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

    return cleanup_summary


def sync_from_kaggle(kernel_ref):
    download_dir = tempfile.mkdtemp(prefix="kaggle_sync_download_")
    try:
        if not os.getenv("KAGGLE_USERNAME") or not os.getenv("KAGGLE_KEY"):
            if "KAGGLE_USERNAME" in st.secrets and "KAGGLE_KEY" in st.secrets:
                os.environ["KAGGLE_USERNAME"] = st.secrets["KAGGLE_USERNAME"]
                os.environ["KAGGLE_KEY"] = st.secrets["KAGGLE_KEY"]
            else:
                return False, "❌ Kaggleの認証情報が見つかりません。", None

        is_valid, normalized_ref, validation_message = validate_kernel_ref(kernel_ref)
        if not is_valid:
            return False, validation_message, None

        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        api.kernels_output(normalized_ref, path=download_dir)

        plan, selected_files = build_sync_plan(download_dir)
        if not plan:
            return False, "❌ Kaggle Output から同期対象 artifact を見つけられませんでした。", None

        sync_evaluation = evaluate_sync_plan(plan, selected_files)
        if not sync_evaluation["ok"]:
            return False, sync_evaluation["error_message"], sync_evaluation

        cleanup_summary = apply_sync_plan(sync_evaluation["plan"], sync_evaluation["updated_loto_types"])
        summary = {
            "kernel_ref": normalized_ref,
            "file_count": len(sync_evaluation["plan"]),
            "updated_loto_types": sync_evaluation["updated_loto_types"],
            "skipped": sync_evaluation["skipped"],
            "summary_lines": sync_evaluation["summary_lines"],
            "target_inference_source": sync_evaluation["target_inference_source"],
            "bundle_details": sync_evaluation.get("bundle_details", {}),
            "cleanup_summary": cleanup_summary,
            "manifest_lines": summarize_manifest_sources(selected_files, sync_evaluation["updated_loto_types"]),
        }
        return True, "✅ Kaggleからの同期が完了しました。", summary
    except Exception as exc:
        return False, build_kaggle_sync_error_message(kernel_ref, exc), None
    finally:
        shutil.rmtree(download_dir, ignore_errors=True)


@st.cache_data(ttl=3600)
def load_tabular_assets(ltype):
    data_path = f"data/{ltype}_processed.csv"
    cols_paths = [f"data/{ltype}_feature_cols.json", f"models/{ltype}_feature_cols.json"]

    if not os.path.exists(data_path):
        return None, None

    df = pd.read_csv(data_path)

    feature_cols = None
    for cols_path in cols_paths:
        if os.path.exists(cols_path):
            with open(cols_path, "r", encoding="utf-8") as handle:
                feature_cols = json.load(handle)
            break

    return df, feature_cols


@st.cache_resource(ttl=3600)
def load_model_assets(ltype):
    model_path = f"models/{ltype}_prob.keras"
    scaler_path = f"models/{ltype}_scaler.pkl"
    if not all(os.path.exists(path) for path in [model_path, scaler_path]):
        return None, None

    model = load_model(model_path, compile=False)
    with open(scaler_path, "rb") as handle:
        scaler = pickle.load(handle)
    return model, scaler


def load_assets(ltype):
    df, feature_cols = load_tabular_assets(ltype)
    model, scaler = load_model_assets(ltype)
    return df, model, scaler, feature_cols


@st.cache_data(ttl=3600)
def load_json_candidates(file_name):
    for path in [os.path.join("data", file_name), os.path.join("models", file_name)]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
    return None


def load_eval_report(ltype):
    return load_json_candidates(f"eval_report_{ltype}.json")


def load_manifest(ltype):
    return load_json_candidates(f"manifest_{ltype}.json")


def get_missing_prediction_artifacts(loto_type, df, feature_cols, model, scaler):
    missing = []
    if df is None:
        missing.append(f"data/{loto_type}_processed.csv")
    if feature_cols is None:
        missing.append(f"data/{loto_type}_feature_cols.json or models/{loto_type}_feature_cols.json")
    if model is None:
        missing.append(f"models/{loto_type}_prob.keras")
    if scaler is None:
        missing.append(f"models/{loto_type}_scaler.pkl")
    return missing


def normalize_model_input_shape(model):
    if model is None:
        return None

    input_shape = getattr(model, "input_shape", None)
    if isinstance(input_shape, list):
        input_shape = input_shape[0] if input_shape else None
    if isinstance(input_shape, tuple):
        return input_shape
    return None


def inspect_prediction_artifact_integrity(loto_type, df, feature_cols, model, scaler):
    issues = []

    if df is None or feature_cols is None:
        return issues

    missing_cols = [column for column in feature_cols if column not in df.columns]
    if missing_cols:
        issues.append(
            {
                "kind": "missing_columns",
                "message": "特徴量定義と processed.csv が不整合です。",
                "missing_cols": missing_cols,
                "csv_column_count": len(df.columns),
                "feature_col_count": len(feature_cols),
            }
        )

    scaler_feature_count = getattr(scaler, "n_features_in_", None) if scaler is not None else None
    if scaler_feature_count is not None and scaler_feature_count != len(feature_cols):
        issues.append(
            {
                "kind": "scaler_dimension_mismatch",
                "message": "scaler の入力次元と feature_cols の長さが一致しません。",
                "scaler_feature_count": int(scaler_feature_count),
                "feature_col_count": len(feature_cols),
            }
        )

    model_input_shape = normalize_model_input_shape(model)
    if model_input_shape and len(model_input_shape) >= 3:
        model_lookback = model_input_shape[1]
        model_feature_count = model_input_shape[2]

        if model_lookback is not None and int(model_lookback) != int(LOOKBACK_WINDOW):
            issues.append(
                {
                    "kind": "model_lookback_mismatch",
                    "message": "モデルの入力 lookback と現在の設定が一致しません。",
                    "model_lookback": int(model_lookback),
                    "expected_lookback": int(LOOKBACK_WINDOW),
                }
            )

        if model_feature_count is not None and int(model_feature_count) != len(feature_cols):
            issues.append(
                {
                    "kind": "model_feature_mismatch",
                    "message": "モデル入力次元と feature_cols の長さが一致しません。",
                    "model_feature_count": int(model_feature_count),
                    "feature_col_count": len(feature_cols),
                }
            )

    if len(df) < LOOKBACK_WINDOW:
        issues.append(
            {
                "kind": "insufficient_rows",
                "message": "processed.csv の行数が LOOKBACK_WINDOW より少ないため予測できません。",
                "row_count": int(len(df)),
                "expected_min_rows": int(LOOKBACK_WINDOW),
            }
        )

    return issues


def render_prediction_integrity_issues(loto_type, issues, manifest=None):
    st.error(f"⚠️ {loto_type} の予測用 artifact に整合性エラーがあります。")
    st.write("Kaggle 同期をやり直すか、古い artifact を削除して世代を揃えてください。")
    if isinstance(manifest, dict):
        st.write(
            f"- bundle_id: {manifest.get('bundle_id', '-')}, "
            f"generated_at: {manifest.get('generated_at', '-')}, "
            f"artifact_schema_version: {manifest.get('artifact_schema_version', '-')}, "
            f"loto_type: {manifest.get('loto_type', loto_type)}"
        )

    for issue in issues:
        st.write(f"- {issue['message']}")
        if issue["kind"] == "missing_columns":
            missing_cols = issue["missing_cols"]
            st.write(f"  - 不足カラム数: {len(missing_cols)}")
            st.write(f"  - 先頭不足カラム: {missing_cols[:10]}")
            st.write(f"  - CSV 側カラム数: {issue['csv_column_count']}")
            st.write(f"  - feature_cols 数: {issue['feature_col_count']}")
        elif issue["kind"] == "scaler_dimension_mismatch":
            st.write(f"  - scaler 入力次元: {issue['scaler_feature_count']}")
            st.write(f"  - feature_cols 数: {issue['feature_col_count']}")
        elif issue["kind"] == "model_lookback_mismatch":
            st.write(f"  - model lookback: {issue['model_lookback']}")
            st.write(f"  - expected lookback: {issue['expected_lookback']}")
        elif issue["kind"] == "model_feature_mismatch":
            st.write(f"  - model input feature 次元: {issue['model_feature_count']}")
            st.write(f"  - feature_cols 数: {issue['feature_col_count']}")
        elif issue["kind"] == "insufficient_rows":
            st.write(f"  - CSV 行数: {issue['row_count']}")
            st.write(f"  - 必要最小行数: {issue['expected_min_rows']}")

    st.info(
        "対処案: Kaggle 同期の再実行、`feature_cols.json` / `processed.csv` / `scaler.pkl` / `model.keras` の再取得、"
        "この loto_type の古い artifact を削除して再同期、Kaggle 側で学習完了後に再同期、"
        "bundle_id が一致する成果物セットで揃え直しを実施してください。"
    )


def parse_history_list_cell(value):
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
            except Exception:
                continue
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
        return [text]
    return [value]


def parse_history_bool_series(series):
    if series is None:
        return pd.Series(dtype=bool)
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna(False).astype(str).str.lower().isin(["1", "true", "yes", "y"])


def normalize_prediction_history_df(df):
    if df is None or df.empty:
        return pd.DataFrame()

    normalized = df.copy()
    for column in HISTORY_LIST_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = [[] for _ in range(len(normalized))]
        else:
            normalized[column] = normalized[column].apply(parse_history_list_cell)

    for column in ["draw_id", "pick_count", "max_num", "predicted_top_k_hit_count"]:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    if "fold_index" in normalized.columns:
        normalized["fold_index"] = pd.to_numeric(normalized["fold_index"], errors="coerce")
    else:
        normalized["fold_index"] = np.nan

    if "evaluation_mode" not in normalized.columns:
        normalized["evaluation_mode"] = "unknown"
    normalized["evaluation_mode"] = normalized["evaluation_mode"].fillna("unknown").astype(str)

    if "hit_rate_any" not in normalized.columns:
        normalized["hit_rate_any"] = normalized["predicted_top_k_hit_count"].fillna(0) >= 1
    else:
        normalized["hit_rate_any"] = parse_history_bool_series(normalized["hit_rate_any"])

    if "hit_rate_two_plus" not in normalized.columns:
        normalized["hit_rate_two_plus"] = normalized["predicted_top_k_hit_count"].fillna(0) >= 2
    else:
        normalized["hit_rate_two_plus"] = parse_history_bool_series(normalized["hit_rate_two_plus"])

    if "date" in normalized.columns:
        normalized["date"] = normalized["date"].astype(str)

    normalized = normalized.sort_values(
        ["draw_id", "evaluation_mode", "fold_index"],
        ascending=[True, True, True],
        na_position="first",
    ).reset_index(drop=True)
    return normalized


@st.cache_data(ttl=3600)
def load_prediction_history(ltype):
    json_path = os.path.join("data", f"prediction_history_{ltype}.json")
    csv_path = os.path.join("data", f"prediction_history_{ltype}.csv")

    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        records = payload.get("records", payload) if isinstance(payload, dict) else payload
        return normalize_prediction_history_df(pd.DataFrame(records)), {"format": "json", "path": json_path}

    if os.path.exists(csv_path):
        return normalize_prediction_history_df(pd.read_csv(csv_path)), {"format": "csv", "path": csv_path}

    return None, None


def calculate_next_draw_date(loto_type, last_date_str):
    last_date = pd.to_datetime(last_date_str.replace("/", "-"))
    today = pd.to_datetime(datetime.now().date())
    start_date = last_date + pd.Timedelta(days=1) if last_date >= today else today

    if loto_type == "miniloto":
        draw_weekdays = [1]
    elif loto_type == "loto6":
        draw_weekdays = [0, 3]
    else:
        draw_weekdays = [4]

    next_date = start_date
    while next_date.weekday() not in draw_weekdays:
        next_date += pd.Timedelta(days=1)

    weekdays_ja = ["月", "火", "水", "木", "金", "土", "日"]
    return next_date.strftime("%Y/%m/%d"), weekdays_ja[next_date.weekday()]


def format_metric(value, digits=4):
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def format_number_list(values):
    if not values:
        return "-"
    return ", ".join(f"{int(value):02d}" for value in values)


def format_score_list(values):
    if not values:
        return "-"
    return ", ".join(f"{float(value):.3f}" for value in values)


def summary_entry_to_row(label, summary_entry):
    metric_summary = summary_entry["metric_summary"]
    return {
        "モデル": label,
        "LogLoss mean↓": metric_summary["logloss"]["mean"],
        "LogLoss var": metric_summary["logloss"]["variance"],
        "Brier mean↓": metric_summary["brier"]["mean"],
        "Brier var": metric_summary["brier"]["variance"],
        "Top-K mean↑": metric_summary["mean_overlap_top_k"]["mean"],
        "Top-K var": metric_summary["mean_overlap_top_k"]["variance"],
    }


def legacy_metrics_to_row(label, metrics):
    return {
        "モデル": label,
        "LogLoss (BCE)↓": metrics["logloss"],
        "Brier Score↓": metrics["brier"],
        "Top-K 正解重なり↑": metrics["mean_overlap_top_k"],
    }


def render_calibration_chart(calibration_rows, color):
    calibration_df = pd.DataFrame(calibration_rows)
    calibration_df = calibration_df.dropna(subset=["pred_prob", "true_prob"], how="all")
    if calibration_df.empty:
        st.info("Calibration の表示に十分なデータがありません。")
        return

    bars = alt.Chart(calibration_df).mark_bar(opacity=0.75, color=color).encode(
        x=alt.X("bin_range:O", title="予測確率 bin"),
        y=alt.Y("pred_prob:Q", title="平均予測確率"),
        tooltip=["bin_range", "count", alt.Tooltip("pred_prob:Q", format=".4f"), alt.Tooltip("true_prob:Q", format=".4f")],
    )
    line = alt.Chart(calibration_df).mark_line(color="red", point=True).encode(
        x="bin_range:O",
        y=alt.Y("true_prob:Q", title="実測率"),
    )
    st.altair_chart(bars + line, use_container_width=True)


def render_manifest_section(manifest):
    if not manifest:
        return

    st.subheader("🧾 Artifact Manifest")
    metrics_summary = manifest.get("metrics_summary", {})
    primary_model = metrics_summary.get("primary_model") or metrics_summary.get("walk_forward_model", {})
    best_static = metrics_summary.get("best_static_baseline", {})
    evaluation_source = metrics_summary.get("evaluation_source", "walk_forward")
    final_artifact_status = metrics_summary.get("final_artifact_status", "-")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("latest_draw_id", manifest.get("latest_draw_id"))
    col2.metric("Primary LogLoss", format_metric(primary_model.get("logloss_mean")))
    col3.metric("Primary Brier", format_metric(primary_model.get("brier_mean")))
    col4.metric("Best static", BASELINE_LABELS.get(best_static.get("name"), best_static.get("name", "-")))

    train_range = manifest.get("train_range", {})
    delta = best_static.get("delta_model_minus_baseline", {})
    st.caption(
        f"学習レンジ: 第{train_range.get('start_draw_id', '-')}回 ({train_range.get('start_date', '-')}) "
        f"〜 第{train_range.get('end_draw_id', '-')}回 ({train_range.get('end_date', '-')})"
    )
    if delta:
        st.caption(
            "model - best static: "
            f"logloss={format_metric(delta.get('logloss'))}, "
            f"brier={format_metric(delta.get('brier'))}, "
            f"top-k={format_metric(delta.get('mean_overlap_top_k'))}"
        )
    if manifest.get("prediction_history_rows") is not None:
        st.caption(
            "prediction_history: "
            f"{manifest.get('prediction_history_rows')} rows "
            f"({manifest.get('prediction_history_path', '-')})"
        )
    if manifest.get("bundle_id"):
        st.caption(
            f"bundle_id={manifest.get('bundle_id')}, "
            f"artifact_schema_version={manifest.get('artifact_schema_version', '-')}"
        )
    if manifest.get("generated_at"):
        st.caption(f"generated_at={manifest.get('generated_at')}")
    st.caption(f"evaluation_source={evaluation_source}, final_artifact_status={final_artifact_status}")

    with st.expander("Manifest 詳細", expanded=False):
        st.json(manifest)


def render_legacy_holdout_section(report):
    legacy = report.get("legacy_holdout")
    if not legacy:
        return

    rows = [legacy_metrics_to_row("★ AI Model (holdout)", legacy["model"])]
    rows.extend(
        legacy_metrics_to_row(BASELINE_LABELS.get(name, name), metrics)
        for name, metrics in legacy.get("static_baselines", {}).items()
    )
    rows.extend(
        legacy_metrics_to_row(BASELINE_LABELS.get(name, name), metrics)
        for name, metrics in legacy.get("online_baselines", {}).items()
    )

    st.dataframe(pd.DataFrame(rows).set_index("モデル"), use_container_width=True)
    st.write("##### Holdout Calibration")
    render_calibration_chart(legacy["model"]["calibration"], "#2563eb")


def render_walk_forward_section(report):
    walk_forward = report.get("walk_forward")
    if not walk_forward:
        st.subheader("🕵️ 評価レポート")
        rows = [legacy_metrics_to_row("★ AI Model (LSTM)", report["Model (LSTM)"])]
        rows.extend(
            legacy_metrics_to_row(BASELINE_LABELS.get(name, name), metrics)
            for name, metrics in report.get("Baselines", {}).items()
        )
        st.dataframe(pd.DataFrame(rows).set_index("モデル"), use_container_width=True)
        st.write("##### Calibration")
        render_calibration_chart(report["Model (LSTM)"]["calibration"], "#2563eb")
        return

    aggregate = walk_forward["aggregate"]
    settings = walk_forward.get("settings", {})

    st.subheader("🕵️ Walk-Forward 評価レポート")
    st.caption(
        "主比較対象は static baselines です。online baselines はテスト中に状態更新する参考値として別枠で表示します。"
    )
    st.caption(
        f"initial_train_fraction={settings.get('initial_train_fraction', '-')}, "
        f"test_window={settings.get('test_window', '-')}, "
        f"folds={len(walk_forward.get('folds', []))}"
    )

    main_rows = [summary_entry_to_row("★ AI Model", aggregate["model"])]
    main_rows.extend(
        summary_entry_to_row(BASELINE_LABELS.get(name, name), summary)
        for name, summary in aggregate.get("static_baselines", {}).items()
    )
    st.write("##### Walk-Forward Summary")
    st.dataframe(pd.DataFrame(main_rows).set_index("モデル"), use_container_width=True)

    online_rows = [
        summary_entry_to_row(BASELINE_LABELS.get(name, name), summary)
        for name, summary in aggregate.get("online_baselines", {}).items()
    ]
    if online_rows:
        st.write("##### Online Baselines (参考)")
        st.dataframe(pd.DataFrame(online_rows).set_index("モデル"), use_container_width=True)

    fold_rows = []
    for fold in walk_forward.get("folds", []):
        fold_rows.append(
            {
                "fold": fold["fold"],
                "train_end_draw_id": fold["train_range"]["end_draw_id"],
                "test_start_draw_id": fold["test_range"]["start_draw_id"],
                "test_end_draw_id": fold["test_range"]["end_draw_id"],
                "LogLoss": fold["model"]["logloss"],
                "Brier": fold["model"]["brier"],
                "Top-K": fold["model"]["mean_overlap_top_k"],
            }
        )

    if fold_rows:
        st.write("##### Fold 別モデル成績")
        st.dataframe(pd.DataFrame(fold_rows), use_container_width=True)

    st.write("##### Walk-Forward Calibration")
    render_calibration_chart(aggregate["model"]["calibration"], "#1d4ed8")

    with st.expander("Legacy Holdout", expanded=False):
        render_legacy_holdout_section(report)


def render_prediction_history_section(loto_type):
    history_df, history_meta = load_prediction_history(loto_type)
    st.subheader("✅ 実績との照合")

    if history_df is None or history_df.empty:
        st.info("prediction history が未生成です")
        st.write("train を再実行するか Kaggle 同期を実施してください")
        return

    st.caption(f"source={history_meta.get('format', '-')}, path={history_meta.get('path', '-')}")

    available_modes = ["all"] + sorted(history_df["evaluation_mode"].dropna().unique().tolist())
    default_min_draw = int(history_df["draw_id"].min())
    default_max_draw = int(history_df["draw_id"].max())

    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
    with filter_col1:
        selected_mode = st.selectbox("evaluation_mode", available_modes, index=0)
    with filter_col2:
        draw_range = st.slider("draw_id 範囲", default_min_draw, default_max_draw, (default_min_draw, default_max_draw))
    with filter_col3:
        recent_option = st.selectbox("直近N件表示", ["10", "20", "50", "全件"], index=3)
    with filter_col4:
        one_plus_only = st.checkbox("1個以上一致のみ表示", value=False)

    filtered = history_df.copy()
    if selected_mode != "all":
        filtered = filtered[filtered["evaluation_mode"] == selected_mode]
    filtered = filtered[filtered["draw_id"].between(draw_range[0], draw_range[1])]
    if one_plus_only:
        filtered = filtered[filtered["hit_rate_any"]]

    filtered = filtered.sort_values(["draw_id", "evaluation_mode", "fold_index"], ascending=[False, True, True])
    if recent_option != "全件":
        filtered = filtered.head(int(recent_option))

    if filtered.empty:
        st.warning("条件に一致する prediction history がありません。")
        return

    distribution = (
        filtered["predicted_top_k_hit_count"]
        .fillna(0)
        .astype(int)
        .value_counts()
        .sort_index()
        .rename_axis("一致数")
        .reset_index(name="件数")
    )

    total_rows = len(filtered)
    avg_hit = float(filtered["predicted_top_k_hit_count"].fillna(0).mean()) if total_rows else 0.0
    any_hits = int(filtered["hit_rate_any"].sum())
    two_plus_hits = int(filtered["hit_rate_two_plus"].sum())

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("対象件数", total_rows)
    metric_col2.metric("平均一致数", f"{avg_hit:.2f}")
    metric_col3.metric("1個以上一致", f"{any_hits} / {total_rows}", f"{(any_hits / total_rows):.1%}")
    metric_col4.metric("2個以上一致", f"{two_plus_hits} / {total_rows}", f"{(two_plus_hits / total_rows):.1%}")

    st.write("##### 一致数の分布")
    st.bar_chart(distribution.set_index("一致数"))

    display_df = filtered.copy()
    display_df["actual_numbers"] = display_df["actual_numbers"].apply(format_number_list)
    display_df["predicted_top_k"] = display_df["predicted_top_k"].apply(format_number_list)
    display_df["predicted_top_k_hit_numbers"] = display_df["predicted_top_k_hit_numbers"].apply(format_number_list)
    display_df["fold_index"] = display_df["fold_index"].apply(lambda value: "-" if pd.isna(value) else int(value))
    display_df = display_df[
        [
            "draw_id",
            "date",
            "actual_numbers",
            "predicted_top_k",
            "predicted_top_k_hit_numbers",
            "predicted_top_k_hit_count",
            "evaluation_mode",
            "fold_index",
        ]
    ].rename(
        columns={
            "draw_id": "draw_id",
            "date": "date",
            "actual_numbers": "actual_numbers",
            "predicted_top_k": "predicted_top_k",
            "predicted_top_k_hit_numbers": "predicted_top_k_hit_numbers",
            "predicted_top_k_hit_count": "hit_count",
            "evaluation_mode": "evaluation_mode",
            "fold_index": "fold_index",
        }
    )

    st.write("##### 回別一覧")
    st.dataframe(display_df, use_container_width=True)

    detail_source = filtered.reset_index(drop=True)
    selected_index = st.selectbox(
        "詳細表示",
        options=list(range(len(detail_source))),
        format_func=lambda index: (
            f"第{int(detail_source.iloc[index]['draw_id'])}回 "
            f"{detail_source.iloc[index]['date']} "
            f"{detail_source.iloc[index]['evaluation_mode']} "
            f"fold={('-' if pd.isna(detail_source.iloc[index]['fold_index']) else int(detail_source.iloc[index]['fold_index']))}"
        ),
    )
    detail_row = detail_source.iloc[selected_index]

    st.write("##### 詳細")
    detail_col1, detail_col2, detail_col3 = st.columns(3)
    detail_col1.metric("actual_numbers", format_number_list(detail_row["actual_numbers"]))
    detail_col2.metric("predicted_top_k", format_number_list(detail_row["predicted_top_k"]))
    detail_col3.metric("hit_numbers", format_number_list(detail_row["predicted_top_k_hit_numbers"]))

    probability_df = pd.DataFrame(
        {
            "number": detail_row["top_probability_numbers"],
            "probability": detail_row["top_probability_scores"],
        }
    )
    probability_df["actual_overlap"] = probability_df["number"].isin(detail_row["actual_numbers"])
    probability_df["probability"] = probability_df["probability"].map(lambda value: f"{float(value):.3f}")
    st.dataframe(probability_df, use_container_width=True)
    st.caption(f"top_probability_scores: {format_score_list(detail_row['top_probability_scores'])}")


def render_prediction_tab(loto_type, config, df, model, scaler, feature_cols, manifest):
    st.subheader("✨ 次回向けの確率予測と買い目生成")

    missing_artifacts = get_missing_prediction_artifacts(loto_type, df, feature_cols, model, scaler)
    if missing_artifacts:
        st.error(f"⚠️ {loto_type} の予測用 artifact が不足しています。")
        st.write("不足しているファイル:")
        for path in missing_artifacts:
            st.write(f"- {path}")
        st.info("Kaggle 同期をやり直すか、学習を再実行して artifact 世代を揃えてください。")
        st.stop()

    integrity_issues = inspect_prediction_artifact_integrity(loto_type, df, feature_cols, model, scaler)
    if integrity_issues:
        render_prediction_integrity_issues(loto_type, integrity_issues, manifest)
        st.stop()

    last_draw_id = int(df.iloc[-1]["draw_id"])
    last_draw_date = df.iloc[-1]["date"]
    next_date, next_weekday = calculate_next_draw_date(loto_type, last_draw_date)

    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.info(f"💾 **最新データ:** 第{last_draw_id}回 ({last_draw_date})")
    with col_info2:
        st.success(f"🗓️ **次回抽選:** {next_date} ({next_weekday})")

    st.markdown("---")
    st.write(f"対象 loto_type: `{loto_type}`")
    features_df = df[feature_cols]
    scaled_data = scaler.transform(features_df)
    recent_input = np.array([scaled_data[-LOOKBACK_WINDOW:]], dtype=np.float32)
    probs = model(tf.convert_to_tensor(recent_input), training=False).numpy()[0]

    prob_df = pd.DataFrame({"Number": np.arange(1, config["max_num"] + 1), "Probability": probs})
    chart = alt.Chart(prob_df).mark_bar(color=config["color"]).encode(
        x=alt.X("Number:O", title="数字"),
        y=alt.Y("Probability:Q", title="出現確率", axis=alt.Axis(format="%")),
        tooltip=["Number", alt.Tooltip("Probability:Q", format=".2%")],
    ).properties(height=250)
    st.altair_chart(chart, use_container_width=True)

    st.markdown("##### ⚙️ 買い目生成オプション")
    col1, col2, col3 = st.columns(3)
    with col1:
        sampling_mode = st.selectbox("抽出方式", ["weighted (確率重み付き抽出)", "top-k (最上位固定)"])
    with col2:
        use_psych = st.checkbox("期待値最大化フィルタ", value=True)
    with col3:
        use_stat = st.checkbox("基本統計フィルタ", value=True)

    num_tickets = st.slider("生成口数", 1, 10, 5)

    if st.button("買い目を生成する", type="primary", use_container_width=True):
        mode = "top-k" if "top-k" in sampling_mode else "weighted"
        cols = st.columns(min(num_tickets, 5))
        for index in range(num_tickets):
            candidate = generate_valid_sample(probs, config, use_psych, use_stat, sampling_mode=mode)
            candidate_str = ", ".join(f"{number:02d}" for number in candidate)
            with cols[index % 5]:
                st.markdown(
                    (
                        f'<div style="background-color:{config["color"]}; padding:10px; border-radius:8px; '
                        f'text-align:center; color:white; margin-bottom:10px;"><b>{candidate_str}</b></div>'
                    ),
                    unsafe_allow_html=True,
                )


with st.sidebar:
    st.header("☁️ Kaggle 同期設定")
    sync_notice = st.session_state.pop(SYNC_NOTICE_STATE_KEY, None)
    if sync_notice:
        st.success(sync_notice["message"])
        if sync_notice.get("target_inference_source"):
            st.caption(f"target inference: {sync_notice['target_inference_source']}")
        for line in sync_notice.get("summary_lines", []):
            st.caption(line)
        for loto_type, bundle_info in (sync_notice.get("bundle_details") or {}).items():
            if bundle_info.get("bundle_id"):
                st.caption(
                    f"{loto_type}: bundle_id={bundle_info['bundle_id']} / "
                    f"generated_at={bundle_info.get('generated_at', '-')}"
                )
        for line in sync_notice.get("manifest_lines", []):
            st.caption(line)

    cleanup_notice = st.session_state.pop(LOCAL_CLEAN_NOTICE_STATE_KEY, None)
    if cleanup_notice:
        st.success(cleanup_notice["message"])
        for line in cleanup_notice.get("details", []):
            st.caption(line)

    default_kernel_ref = os.getenv("KAGGLE_SLUG", "")
    if not default_kernel_ref and "KAGGLE_SLUG" in st.secrets:
        default_kernel_ref = st.secrets["KAGGLE_SLUG"]

    kernel_ref_input = st.text_input("Kernel Ref (owner/kernel-slug)", value=default_kernel_ref, help=f"例: {KERNEL_REF_EXAMPLE}")
    st.caption(f"例: `{KERNEL_REF_EXAMPLE}`")
    st.caption("slug 単体ではなく owner を含む `owner/kernel-slug` 形式が必要です。")

    if st.button("🔄 最新AIモデルを同期", use_container_width=True):
        normalized_input = normalize_kernel_ref(kernel_ref_input)
        if not normalized_input:
            st.warning("Kernel Ref を入力してください。")
        else:
            is_valid_ref, normalized_ref, validation_message = validate_kernel_ref(normalized_input)
            if not is_valid_ref:
                st.error(validation_message)
            else:
                with st.spinner("同期中..."):
                    success, message, sync_summary = sync_from_kaggle(normalized_ref)
                    if success:
                        st.session_state[SYNC_NOTICE_STATE_KEY] = {
                            "message": message
                            + f" Kernel Ref: {normalized_ref} / files={((sync_summary or {}).get('file_count', '-'))}",
                            "target_inference_source": (sync_summary or {}).get("target_inference_source"),
                            "summary_lines": (sync_summary or {}).get("summary_lines", []),
                            "bundle_details": (sync_summary or {}).get("bundle_details", {}),
                            "manifest_lines": (sync_summary or {}).get("manifest_lines", []),
                        }
                        st.cache_data.clear()
                        st.cache_resource.clear()
                        st.rerun()
                    else:
                        st.error(message)

    with st.expander("🧹 ローカル artifact 管理", expanded=False):
        st.caption("対象 loto_type の processed / feature_cols / scaler / model / manifest / eval_report / prediction_history を削除します。")
        for loto_type in sorted(LOTO_CONFIG.keys()):
            if st.button(f"🧹 {loto_type} のローカル artifact を削除", use_container_width=True, key=f"cleanup-{loto_type}"):
                cleanup_result = remove_local_artifacts_for_loto(loto_type)
                st.session_state[LOCAL_CLEAN_NOTICE_STATE_KEY] = {
                    "message": format_cleanup_notice(loto_type, cleanup_result),
                    "details": [
                        f"removed: {', '.join(cleanup_result['removed'])}" if cleanup_result["removed"] else "removed: なし",
                        f"missing: {', '.join(cleanup_result['missing'])}" if cleanup_result["missing"] else "missing: なし",
                    ],
                }
                st.cache_data.clear()
                st.cache_resource.clear()
                st.rerun()


st.title("🎯 宝くじ AI確率予測システム")
st.markdown("LSTMから出力された**出現確率ベクトル**に基づき、重み付きサンプリングで買い目を生成します。")
st.markdown("---")

selected_loto = st.radio(
    "宝くじの種類",
    options=list(LOTO_CONFIG.keys()),
    format_func=lambda x: LOTO_CONFIG[x]["name"],
    horizontal=True,
)
config = LOTO_CONFIG[selected_loto]

df, model, scaler, feature_cols = load_assets(selected_loto)
manifest = load_manifest(selected_loto)

tab1, tab2, tab3 = st.tabs(["🎲 確率サンプリング予測", "📊 モデル評価レポート (Walk-Forward)", "✅ 実績との照合"])

with tab2:
    report = load_eval_report(selected_loto)
    render_manifest_section(manifest)

    if report:
        render_walk_forward_section(report)
    else:
        st.info("評価レポートが見つかりません。")
        st.write("デバッグ情報: data/ フォルダ内のファイル一覧")
        st.write(os.listdir("data") if os.path.exists("data") else "data フォルダが存在しません")

with tab3:
    render_prediction_history_section(selected_loto)

with tab1:
    render_prediction_tab(selected_loto, config, df, model, scaler, feature_cols, manifest)
