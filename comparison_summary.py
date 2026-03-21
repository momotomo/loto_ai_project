"""comparison_summary.py

Aggregates eval_report results from multiple single-seed runs into a
cross-seed comparison summary.  Designed to work alongside
scripts/run_multi_seed.py but can also be used standalone.

Output artifact schema (saved as data/comparison_summary_{loto_type}.json):
{
  "schema_version": 1,
  "generated_at": "...",
  "loto_type": "loto6",
  "preset": "archcomp",
  "seeds": [42, 123, 456],
  "run_count": 3,
  "variants": {
    "deepsets": {
      "run_count": 3,
      "logloss": {"mean": 0.123, "std": 0.005, "values": [...]},
      "brier":   {"mean": 0.045, "std": 0.002, "values": [...]},
      "ece":     {"mean": 0.012, "std": 0.003, "values": [...]},
      "calibration_recommendations": {"none": 2, "temperature": 1},
      "promote_count": 0,
      "hold_count": 3,
    },
    ...
  },
  "pairwise_comparisons": {
    "settransformer_vs_deepsets": {
      "run_count": 3,
      "ci_wins": 1,          # runs where bootstrap_ci.upper < 0
      "permutation_wins": 2, # runs where permutation_test.p_value < alpha
      "both_pass_count": 0,  # runs where both pass simultaneously
    },
    ...
  },
}
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

COMPARISON_SUMMARY_SCHEMA_VERSION = 1

# Pairwise comparison keys that are surfaced in the summary
PAIRWISE_KEYS_OF_INTEREST = [
    "settransformer_vs_deepsets",
    "settransformer_vs_multihot",
    "deepsets_vs_multihot",
    "settransformer_vs_legacy",
    "deepsets_vs_legacy",
    "multihot_vs_legacy",
]


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _aggregate_metric_values(values: list[float | None]) -> dict[str, Any]:
    """Compute mean, std, and raw values for a list of metric readings."""
    clean = [v for v in values if v is not None]
    if not clean:
        return {"mean": None, "std": None, "values": values}
    arr = np.asarray(clean, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "values": [float(v) if v is not None else None for v in values],
    }


def _extract_variant_metrics(eval_report: dict, variant_name: str) -> dict[str, float | None]:
    """Extract walk-forward aggregate logloss/brier/ece for a single variant."""
    model_variants = eval_report.get("model_variants") or {}
    variant_payload = model_variants.get(variant_name) or {}

    # Prefer walk_forward aggregate over legacy_holdout
    walk_forward = variant_payload.get("walk_forward")
    if walk_forward is not None:
        agg_model = (walk_forward.get("aggregate") or {}).get("model") or {}
        metric_summary = agg_model.get("metric_summary") or {}
        logloss = _float_or_none((metric_summary.get("logloss") or {}).get("mean"))
        brier = _float_or_none((metric_summary.get("brier") or {}).get("mean"))
        ece_entry = metric_summary.get("ece") or {}
        ece = _float_or_none(ece_entry.get("mean") if isinstance(ece_entry, dict) else ece_entry)
        return {"logloss": logloss, "brier": brier, "ece": ece}

    legacy_holdout = variant_payload.get("legacy_holdout")
    if legacy_holdout is not None:
        model_metrics = legacy_holdout.get("model") or {}
        return {
            "logloss": _float_or_none(model_metrics.get("logloss")),
            "brier": _float_or_none(model_metrics.get("brier")),
            "ece": _float_or_none((model_metrics.get("calibration_summary") or {}).get("ece")),
        }

    return {"logloss": None, "brier": None, "ece": None}


def _extract_calibration_recommendation(eval_report: dict, variant_name: str) -> str:
    """Return the recommended calibration method for a variant in this run."""
    calibration_eval = eval_report.get("calibration_evaluation") or {}
    selection = calibration_eval.get(variant_name) or {}
    return str(selection.get("recommended_method") or "none")


def _extract_promote_flag(eval_report: dict, variant_name: str) -> bool | None:
    """Return should_promote for a variant in the decision summary."""
    decision = eval_report.get("decision_summary") or {}
    challenger_decisions = decision.get("challenger_decisions") or {}
    payload = challenger_decisions.get(variant_name)
    if payload is None:
        return None
    return bool(payload.get("should_promote", False))


def _extract_pairwise_flags(eval_report: dict, comparison_key: str, alpha: float = 0.05) -> dict[str, bool]:
    """Return ci_win and permutation_win flags for a given comparison key."""
    statistical_tests = eval_report.get("statistical_tests") or {}
    comparisons = statistical_tests.get("comparisons") or {}
    comp = comparisons.get(comparison_key)
    if comp is None:
        return {"ci_win": False, "permutation_win": False, "both_pass": False}

    ci = comp.get("bootstrap_ci") or {}
    ci_upper = _float_or_none(ci.get("upper"))
    ci_win = ci_upper is not None and ci_upper < 0.0

    perm = comp.get("permutation_test") or {}
    p_value = _float_or_none(perm.get("p_value"))
    perm_win = p_value is not None and p_value < alpha

    return {"ci_win": ci_win, "permutation_win": perm_win, "both_pass": ci_win and perm_win}


def aggregate_variant_metrics(eval_reports: list[dict], variant_names: list[str]) -> dict[str, Any]:
    """Aggregate per-variant metrics across multiple eval_report dicts."""
    result: dict[str, Any] = {}

    for variant_name in variant_names:
        logloss_values: list[float | None] = []
        brier_values: list[float | None] = []
        ece_values: list[float | None] = []
        calibration_recs: dict[str, int] = {}
        promote_count = 0
        hold_count = 0
        run_count = 0

        for report in eval_reports:
            if variant_name not in (report.get("model_variants") or {}):
                continue
            run_count += 1
            metrics = _extract_variant_metrics(report, variant_name)
            logloss_values.append(metrics["logloss"])
            brier_values.append(metrics["brier"])
            ece_values.append(metrics["ece"])

            cal_rec = _extract_calibration_recommendation(report, variant_name)
            calibration_recs[cal_rec] = calibration_recs.get(cal_rec, 0) + 1

            promote_flag = _extract_promote_flag(report, variant_name)
            if promote_flag is True:
                promote_count += 1
            elif promote_flag is False:
                hold_count += 1

        result[variant_name] = {
            "run_count": run_count,
            "logloss": _aggregate_metric_values(logloss_values),
            "brier": _aggregate_metric_values(brier_values),
            "ece": _aggregate_metric_values(ece_values),
            "calibration_recommendations": calibration_recs,
            "promote_count": promote_count,
            "hold_count": hold_count,
        }

    return result


def aggregate_pairwise_comparisons(
    eval_reports: list[dict],
    comparison_keys: list[str] | None = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Aggregate pairwise comparison flags across runs."""
    if comparison_keys is None:
        # Collect all keys present in any report
        all_keys: set[str] = set()
        for report in eval_reports:
            comps = (report.get("statistical_tests") or {}).get("comparisons") or {}
            all_keys.update(comps.keys())
        # Prioritise the keys of interest, then add any additional ones
        ordered: list[str] = [k for k in PAIRWISE_KEYS_OF_INTEREST if k in all_keys]
        for k in sorted(all_keys):
            if k not in ordered:
                ordered.append(k)
        comparison_keys = ordered

    result: dict[str, Any] = {}
    for key in comparison_keys:
        ci_wins = 0
        perm_wins = 0
        both_pass = 0
        run_count = 0

        for report in eval_reports:
            comps = (report.get("statistical_tests") or {}).get("comparisons") or {}
            if key not in comps:
                continue
            run_count += 1
            flags = _extract_pairwise_flags(report, key, alpha=alpha)
            if flags["ci_win"]:
                ci_wins += 1
            if flags["permutation_win"]:
                perm_wins += 1
            if flags["both_pass"]:
                both_pass += 1

        result[key] = {
            "run_count": run_count,
            "ci_wins": ci_wins,
            "permutation_wins": perm_wins,
            "both_pass_count": both_pass,
        }

    return result


def build_comparison_summary(
    eval_reports: list[dict],
    loto_type: str,
    preset: str,
    seeds: list[int],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Build the full comparison summary from a list of eval_reports."""
    # Collect all variant names present across runs
    variant_names_seen: set[str] = set()
    for report in eval_reports:
        variant_names_seen.update((report.get("model_variants") or {}).keys())
    # Sort for determinism
    variant_names = sorted(variant_names_seen)

    return {
        "schema_version": COMPARISON_SUMMARY_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "loto_type": loto_type,
        "preset": preset,
        "seeds": list(seeds),
        "run_count": len(eval_reports),
        "alpha": alpha,
        "variants": aggregate_variant_metrics(eval_reports, variant_names),
        "pairwise_comparisons": aggregate_pairwise_comparisons(eval_reports, alpha=alpha),
    }


def load_eval_report(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_comparison_summary(summary: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def get_comparison_summary_path(loto_type: str, data_dir: str = "data") -> Path:
    return Path(data_dir) / f"comparison_summary_{loto_type}.json"
