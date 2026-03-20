import json
import math
import os

import numpy as np
from sklearn.isotonic import IsotonicRegression


NO_CALIBRATION_METHOD = "none"
TEMPERATURE_CALIBRATION_METHOD = "temperature"
ISOTONIC_CALIBRATION_METHOD = "isotonic"
DEFAULT_SAVED_CALIBRATION_METHOD = NO_CALIBRATION_METHOD
DEFAULT_EVALUATION_CALIBRATION_METHODS = (
    f"{NO_CALIBRATION_METHOD},{TEMPERATURE_CALIBRATION_METHOD},{ISOTONIC_CALIBRATION_METHOD}"
)
CALIBRATION_METHOD_CHOICES = (
    NO_CALIBRATION_METHOD,
    TEMPERATURE_CALIBRATION_METHOD,
    ISOTONIC_CALIBRATION_METHOD,
)
CALIBRATION_METHOD_LABELS = {
    NO_CALIBRATION_METHOD: "No Calibration",
    TEMPERATURE_CALIBRATION_METHOD: "Temperature Scaling",
    ISOTONIC_CALIBRATION_METHOD: "Isotonic Regression",
}
CALIBRATOR_ARTIFACT_SCHEMA_VERSION = 1
EPSILON = 1e-6


def resolve_calibration_method(value):
    normalized = (value or DEFAULT_SAVED_CALIBRATION_METHOD).strip().lower()
    if normalized not in CALIBRATION_METHOD_CHOICES:
        raise ValueError(f"unknown calibration_method: {value}")
    return normalized


def get_calibration_method_label(value):
    resolved = resolve_calibration_method(value)
    return CALIBRATION_METHOD_LABELS.get(resolved, resolved)


def parse_calibration_methods_csv(value):
    methods = []
    for chunk in str(value or "").split(","):
        normalized = chunk.strip()
        if not normalized:
            continue
        resolved = resolve_calibration_method(normalized)
        if resolved not in methods:
            methods.append(resolved)
    return methods


def resolve_evaluation_calibration_methods(saved_method, csv_value=None):
    methods = parse_calibration_methods_csv(csv_value)
    if not methods:
        methods = parse_calibration_methods_csv(DEFAULT_EVALUATION_CALIBRATION_METHODS)
    if NO_CALIBRATION_METHOD not in methods:
        methods.insert(0, NO_CALIBRATION_METHOD)
    if saved_method not in methods:
        methods.append(saved_method)
    return methods


def sanitize_probabilities(preds):
    return np.clip(np.asarray(preds, dtype=np.float32), EPSILON, 1.0 - EPSILON)


def sigmoid(values):
    array = np.asarray(values, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-array))


def logit(values):
    probs = sanitize_probabilities(values)
    return np.log(probs / (1.0 - probs))


def calculate_binary_logloss(preds, targets):
    probabilities = sanitize_probabilities(preds).reshape(-1)
    truth = np.asarray(targets, dtype=np.float32).reshape(-1)
    return float(-np.mean(truth * np.log(probabilities) + (1.0 - truth) * np.log(1.0 - probabilities)))


def calculate_reliability_bins(preds, targets, num_bins=10):
    probabilities = sanitize_probabilities(preds).reshape(-1)
    truth = np.asarray(targets, dtype=np.float32).reshape(-1)
    bins = np.linspace(0.0, 1.0, int(num_bins) + 1)
    rows = []

    for index in range(int(num_bins)):
        lower = float(bins[index])
        upper = float(bins[index + 1])
        if index == int(num_bins) - 1:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
        count = int(mask.sum())
        rows.append(
            {
                "bin_index": index,
                "bin_range": f"{lower:.1f}-{upper:.1f}",
                "count": count,
                "pred_prob": float(probabilities[mask].mean()) if count else None,
                "true_prob": float(truth[mask].mean()) if count else None,
            }
        )

    return rows


def summarize_reliability_bins(reliability_bins):
    total_count = sum(int(row.get("count", 0)) for row in reliability_bins)
    if total_count <= 0:
        return {
            "num_bins": len(reliability_bins),
            "sample_count": 0,
            "ece": None,
            "max_calibration_error": None,
            "avg_pred_prob": None,
            "avg_true_prob": None,
        }

    ece = 0.0
    max_error = 0.0
    weighted_pred = 0.0
    weighted_true = 0.0
    for row in reliability_bins:
        count = int(row.get("count", 0))
        pred_prob = row.get("pred_prob")
        true_prob = row.get("true_prob")
        if count <= 0 or pred_prob is None or true_prob is None:
            continue
        error = abs(float(pred_prob) - float(true_prob))
        weight = count / total_count
        ece += error * weight
        max_error = max(max_error, error)
        weighted_pred += float(pred_prob) * weight
        weighted_true += float(true_prob) * weight

    return {
        "num_bins": len(reliability_bins),
        "sample_count": int(total_count),
        "ece": float(ece),
        "max_calibration_error": float(max_error),
        "avg_pred_prob": float(weighted_pred),
        "avg_true_prob": float(weighted_true),
    }


def build_calibration_quality_summary(preds, targets, num_bins=10):
    reliability_bins = calculate_reliability_bins(preds, targets, num_bins=num_bins)
    summary = summarize_reliability_bins(reliability_bins)
    summary["reliability_bins"] = reliability_bins
    return summary


def build_identity_calibrator_artifact(status="disabled", reason=None):
    artifact = {
        "schema_version": CALIBRATOR_ARTIFACT_SCHEMA_VERSION,
        "method": NO_CALIBRATION_METHOD,
        "status": status,
        "enabled": False,
    }
    if reason:
        artifact["reason"] = reason
    return artifact


def summarize_calibrator_artifact(artifact):
    if not isinstance(artifact, dict):
        return {"status": "missing", "method": NO_CALIBRATION_METHOD}

    summary = {
        "method": artifact.get("method", NO_CALIBRATION_METHOD),
        "status": artifact.get("status", "unknown"),
        "enabled": bool(artifact.get("enabled", False)),
    }
    if "temperature" in artifact:
        summary["temperature"] = artifact["temperature"]
    if "x_thresholds" in artifact:
        summary["threshold_count"] = len(artifact.get("x_thresholds") or [])
    if artifact.get("reason"):
        summary["reason"] = artifact["reason"]
    return summary


def fit_temperature_calibrator(preds, targets, grid_size=61):
    probabilities = sanitize_probabilities(preds).reshape(-1)
    truth = np.asarray(targets, dtype=np.float32).reshape(-1)
    if probabilities.size == 0:
        return build_identity_calibrator_artifact(status="skipped", reason="empty_calibration_split")

    logits = logit(probabilities)
    temperature_grid = np.exp(np.linspace(math.log(0.25), math.log(4.0), int(grid_size)))
    losses = []
    for temperature in temperature_grid:
        candidate = sigmoid(logits / float(temperature))
        losses.append(calculate_binary_logloss(candidate, truth))

    best_index = int(np.argmin(losses))
    best_temperature = float(temperature_grid[best_index])
    return {
        "schema_version": CALIBRATOR_ARTIFACT_SCHEMA_VERSION,
        "method": TEMPERATURE_CALIBRATION_METHOD,
        "status": "fitted",
        "enabled": True,
        "temperature": best_temperature,
        "grid_size": int(grid_size),
        "calibration_sample_count": int(probabilities.size),
        "best_calibration_logloss": float(losses[best_index]),
    }


def fit_isotonic_calibrator(preds, targets):
    probabilities = sanitize_probabilities(preds).reshape(-1)
    truth = np.asarray(targets, dtype=np.float32).reshape(-1)
    unique_targets = np.unique(truth)
    unique_probabilities = np.unique(probabilities)
    if probabilities.size == 0:
        return build_identity_calibrator_artifact(status="skipped", reason="empty_calibration_split")
    if unique_targets.size < 2:
        return build_identity_calibrator_artifact(status="skipped", reason="single_class_targets")
    if unique_probabilities.size < 2:
        return build_identity_calibrator_artifact(status="skipped", reason="constant_probabilities")

    regressor = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    regressor.fit(probabilities, truth)
    return {
        "schema_version": CALIBRATOR_ARTIFACT_SCHEMA_VERSION,
        "method": ISOTONIC_CALIBRATION_METHOD,
        "status": "fitted",
        "enabled": True,
        "x_thresholds": [float(value) for value in regressor.X_thresholds_.tolist()],
        "y_thresholds": [float(value) for value in regressor.y_thresholds_.tolist()],
        "calibration_sample_count": int(probabilities.size),
    }


def fit_calibrator(method, preds, targets):
    resolved_method = resolve_calibration_method(method)
    if resolved_method == NO_CALIBRATION_METHOD:
        return build_identity_calibrator_artifact()
    if resolved_method == TEMPERATURE_CALIBRATION_METHOD:
        return fit_temperature_calibrator(preds, targets)
    if resolved_method == ISOTONIC_CALIBRATION_METHOD:
        return fit_isotonic_calibrator(preds, targets)
    raise ValueError(f"unsupported calibration method: {method}")


def apply_calibration_artifact(artifact, probs):
    probabilities = sanitize_probabilities(probs)
    if not isinstance(artifact, dict):
        return probabilities

    method = resolve_calibration_method(artifact.get("method", NO_CALIBRATION_METHOD))
    status = artifact.get("status")
    if method == NO_CALIBRATION_METHOD or status != "fitted":
        return probabilities

    if method == TEMPERATURE_CALIBRATION_METHOD:
        temperature = float(artifact.get("temperature", 1.0))
        if temperature <= 0.0:
            return probabilities
        return sanitize_probabilities(sigmoid(logit(probabilities) / temperature))

    if method == ISOTONIC_CALIBRATION_METHOD:
        x_thresholds = np.asarray(artifact.get("x_thresholds") or [], dtype=np.float32)
        y_thresholds = np.asarray(artifact.get("y_thresholds") or [], dtype=np.float32)
        if x_thresholds.size < 2 or y_thresholds.size != x_thresholds.size:
            return probabilities
        calibrated = np.interp(
            probabilities.reshape(-1),
            x_thresholds,
            y_thresholds,
            left=float(y_thresholds[0]),
            right=float(y_thresholds[-1]),
        )
        return sanitize_probabilities(calibrated.reshape(probabilities.shape))

    return probabilities


def get_calibrator_file_name(loto_type):
    return f"{loto_type}_calibrator.json"


def get_calibrator_candidate_paths(loto_type, data_dir="data", model_dir="models"):
    file_name = get_calibrator_file_name(loto_type)
    return [
        os.path.join(model_dir, file_name),
        os.path.join(data_dir, file_name),
    ]


def load_calibrator_artifact(loto_type, data_dir="data", model_dir="models"):
    for path in get_calibrator_candidate_paths(loto_type, data_dir=data_dir, model_dir=model_dir):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            return payload, path
    return None, None
