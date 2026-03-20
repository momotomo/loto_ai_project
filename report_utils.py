from calibration_utils import DEFAULT_SAVED_CALIBRATION_METHOD, get_calibration_method_label, resolve_calibration_method
from model_variants import DEFAULT_MODEL_VARIANT, get_model_variant_label, resolve_model_variant


def safe_variant_label(value):
    if not value:
        return "-"
    try:
        return get_model_variant_label(value)
    except Exception:
        return str(value)


def get_training_context(manifest=None, report=None):
    if isinstance(manifest, dict) and isinstance(manifest.get("training_context"), dict):
        return manifest["training_context"]
    if isinstance(report, dict) and isinstance(report.get("training_context"), dict):
        return report["training_context"]
    return {}


def get_saved_model_variant(manifest=None, report=None):
    training_context = get_training_context(manifest=manifest, report=report)
    try:
        return resolve_model_variant(training_context.get("model_variant", DEFAULT_MODEL_VARIANT))
    except Exception:
        return DEFAULT_MODEL_VARIANT


def safe_calibration_label(value):
    if not value:
        return get_calibration_method_label(DEFAULT_SAVED_CALIBRATION_METHOD)
    try:
        return get_calibration_method_label(value)
    except Exception:
        return str(value)


def get_saved_calibration_method(manifest=None, report=None):
    training_context = get_training_context(manifest=manifest, report=report)
    calibration_payload = {}
    if isinstance(manifest, dict):
        calibration_payload = manifest.get("calibration") or {}
    elif isinstance(report, dict):
        calibration_payload = report.get("calibration") or {}

    for candidate in [
        training_context.get("saved_calibration_method"),
        calibration_payload.get("saved_method"),
        calibration_payload.get("saved_calibration_method"),
        training_context.get("saved_calibration_method_requested"),
    ]:
        try:
            return resolve_calibration_method(candidate)
        except Exception:
            continue
    return DEFAULT_SAVED_CALIBRATION_METHOD


def build_variant_summary_rows(report):
    if not isinstance(report, dict):
        return []

    model_variants = report.get("model_variants") or {}
    rows = []
    for variant_name, payload in sorted(model_variants.items()):
        walk_forward = payload.get("walk_forward") or {}
        aggregate = walk_forward.get("aggregate") or {}
        model_summary = (aggregate.get("model") or {}).get("metric_summary") or {}
        calibration_selection = payload.get("calibration_selection") or {}
        selected_metrics = calibration_selection.get("recommended_metrics") or {}
        raw_metrics = calibration_selection.get("raw_metrics") or {}
        rows.append(
            {
                "variant": variant_name,
                "label": safe_variant_label(variant_name),
                "dataset_variant": payload.get("dataset_variant", variant_name),
                "feature_strategy": payload.get("feature_strategy", "-"),
                "selected_calibration_method": calibration_selection.get(
                    "recommended_method", DEFAULT_SAVED_CALIBRATION_METHOD
                ),
                "selected_calibration_label": safe_calibration_label(
                    calibration_selection.get("recommended_method", DEFAULT_SAVED_CALIBRATION_METHOD)
                ),
                "raw_logloss_mean": raw_metrics.get("logloss", ((model_summary.get("logloss") or {}).get("mean"))),
                "raw_brier_mean": raw_metrics.get("brier", ((model_summary.get("brier") or {}).get("mean"))),
                "raw_ece": raw_metrics.get("ece", ((aggregate.get("model") or {}).get("calibration_summary") or {}).get("ece")),
                "logloss_mean": selected_metrics.get("logloss", ((model_summary.get("logloss") or {}).get("mean"))),
                "brier_mean": selected_metrics.get("brier", ((model_summary.get("brier") or {}).get("mean"))),
                "ece": selected_metrics.get("ece"),
                "mean_overlap_top_k_mean": ((model_summary.get("mean_overlap_top_k") or {}).get("mean")),
            }
        )
    return rows


def build_statistical_test_rows(report):
    if not isinstance(report, dict):
        return []

    statistical_tests = report.get("statistical_tests") or {}
    comparisons = statistical_tests.get("comparisons") or {}
    rows = []
    for comparison_name, payload in sorted(comparisons.items()):
        rows.append(
            {
                "comparison": comparison_name,
                "candidate": payload.get("candidate_name", "-"),
                "reference": payload.get("reference_name", "-"),
                "candidate_calibration_method": payload.get("candidate_calibration_method", DEFAULT_SAVED_CALIBRATION_METHOD),
                "reference_calibration_method": payload.get("reference_calibration_method", DEFAULT_SAVED_CALIBRATION_METHOD),
                "mean_delta": payload.get("mean_delta"),
                "ci_lower": ((payload.get("bootstrap_ci") or {}).get("lower")),
                "ci_upper": ((payload.get("bootstrap_ci") or {}).get("upper")),
                "p_value": ((payload.get("permutation_test") or {}).get("p_value")),
                "sample_count": payload.get("sample_count"),
            }
        )
    return rows


def build_calibration_summary_rows(report):
    if not isinstance(report, dict):
        return []

    rows = []
    for variant_name, payload in sorted((report.get("model_variants") or {}).items()):
        calibration_selection = payload.get("calibration_selection") or {}
        for method_payload in calibration_selection.get("methods", []):
            metrics = method_payload.get("metrics") or {}
            delta_vs_raw = method_payload.get("delta_vs_raw") or {}
            rows.append(
                {
                    "variant": variant_name,
                    "variant_label": safe_variant_label(variant_name),
                    "method": method_payload.get("method", DEFAULT_SAVED_CALIBRATION_METHOD),
                    "method_label": safe_calibration_label(method_payload.get("method", DEFAULT_SAVED_CALIBRATION_METHOD)),
                    "status": method_payload.get("status", "-"),
                    "selected": bool(method_payload.get("selected", False)),
                    "eligible": bool(method_payload.get("eligible", False)),
                    "logloss": metrics.get("logloss"),
                    "brier": metrics.get("brier"),
                    "ece": metrics.get("ece"),
                    "delta_logloss_vs_raw": delta_vs_raw.get("logloss"),
                    "delta_brier_vs_raw": delta_vs_raw.get("brier"),
                    "delta_ece_vs_raw": delta_vs_raw.get("ece"),
                    "sample_count": metrics.get("sample_count"),
                }
            )
    return rows
