import hashlib
import os
import platform
from importlib import metadata


TRACKED_DEPENDENCIES = (
    "altair",
    "kaggle",
    "numpy",
    "pandas",
    "pytest",
    "requests",
    "scikit-learn",
    "streamlit",
    "tensorflow",
)


def get_package_version(distribution_name):
    try:
        return metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        return None


def collect_runtime_environment():
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "dependencies": {
            dependency_name: get_package_version(dependency_name) for dependency_name in TRACKED_DEPENDENCIES
        },
    }


def compute_file_sha256(path, chunk_size=65536):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_file_metadata(path_map):
    metadata_map = {}
    for name, path in path_map.items():
        if not path or not os.path.exists(path):
            continue
        metadata_map[name] = {
            "path": path,
            "sha256": compute_file_sha256(path),
            "size_bytes": os.path.getsize(path),
        }
    return metadata_map


def build_data_fingerprint(
    processed_path,
    raw_path=None,
    row_count=None,
    sample_count=None,
    latest_draw_id=None,
    preprocessing_version=None,
):
    file_metadata = collect_file_metadata(
        {
            "processed_csv": processed_path,
            "raw_csv": raw_path,
        }
    )
    processed_metadata = file_metadata.get("processed_csv", {})
    return {
        "data_hash": processed_metadata.get("sha256"),
        "preprocessing_version": preprocessing_version,
        "row_count": int(row_count) if row_count is not None else None,
        "sample_count": int(sample_count) if sample_count is not None else None,
        "latest_draw_id": int(latest_draw_id) if latest_draw_id is not None else None,
        "files": file_metadata,
    }


def normalize_model_input_shape(model):
    if model is None:
        return None

    input_shape = getattr(model, "input_shape", None)
    if isinstance(input_shape, list):
        input_shape = input_shape[0] if input_shape else None
    if isinstance(input_shape, tuple):
        return input_shape
    return None


def inspect_prediction_artifact_integrity(
    df,
    feature_cols,
    model,
    scaler,
    lookback_window,
    feature_strategy="tabular",
    prepared_feature_count=None,
):
    issues = []

    if df is None or feature_cols is None:
        return issues

    if feature_strategy == "tabular":
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
    elif prepared_feature_count is not None and prepared_feature_count != len(feature_cols):
        issues.append(
            {
                "kind": "prepared_feature_mismatch",
                "message": "派生特徴の生成結果と feature_cols の長さが一致しません。",
                "prepared_feature_count": int(prepared_feature_count),
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

        if model_lookback is not None and int(model_lookback) != int(lookback_window):
            issues.append(
                {
                    "kind": "model_lookback_mismatch",
                    "message": "モデルの入力 lookback と現在の設定が一致しません。",
                    "model_lookback": int(model_lookback),
                    "expected_lookback": int(lookback_window),
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

    if len(df) < lookback_window:
        issues.append(
            {
                "kind": "insufficient_rows",
                "message": "processed.csv の行数が LOOKBACK_WINDOW より少ないため予測できません。",
                "row_count": int(len(df)),
                "expected_min_rows": int(lookback_window),
            }
        )

    return issues
