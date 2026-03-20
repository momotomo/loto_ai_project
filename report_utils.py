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


def build_variant_summary_rows(report):
    if not isinstance(report, dict):
        return []

    model_variants = report.get("model_variants") or {}
    rows = []
    for variant_name, payload in sorted(model_variants.items()):
        walk_forward = payload.get("walk_forward") or {}
        aggregate = walk_forward.get("aggregate") or {}
        model_summary = (aggregate.get("model") or {}).get("metric_summary") or {}
        rows.append(
            {
                "variant": variant_name,
                "label": safe_variant_label(variant_name),
                "dataset_variant": payload.get("dataset_variant", variant_name),
                "feature_strategy": payload.get("feature_strategy", "-"),
                "logloss_mean": ((model_summary.get("logloss") or {}).get("mean")),
                "brier_mean": ((model_summary.get("brier") or {}).get("mean")),
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
                "mean_delta": payload.get("mean_delta"),
                "ci_lower": ((payload.get("bootstrap_ci") or {}).get("lower")),
                "ci_upper": ((payload.get("bootstrap_ci") or {}).get("upper")),
                "p_value": ((payload.get("permutation_test") or {}).get("p_value")),
                "sample_count": payload.get("sample_count"),
            }
        )
    return rows
