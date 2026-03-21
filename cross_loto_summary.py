"""cross_loto_summary.py

Aggregates per-loto comparison summaries (comparison_summary_{loto_type}.json)
into a cross-loto summary artifact and a human/machine-readable recommendation
artifact.

Output artifacts:
  data/cross_loto_summary.json    — variant ranking, pairwise comparisons,
                                    and promotion trends across loto_types.
  data/recommendation.json        — next-action recommendation derived from
                                    the cross-loto summary.

Schema overview (cross_loto_summary.json):
{
  "schema_version": 1,
  "generated_at": "...",
  "loto_types": [...],
  "preset": "archcomp",
  "seeds": [...],
  "alpha": 0.05,
  "run_count_per_loto": {"loto6": 3, ...},
  "overall_summary": {
    "variants": {
      "<name>": {
        "loto_types_evaluated": [...],
        "logloss": {"mean": ..., "std": ..., "per_loto": {...}},
        "brier":   {"mean": ..., "std": ..., "per_loto": {...}},
        "ece":     {"mean": ..., "std": ..., "per_loto": {...}},
        "promote_count_total": 0,
        "hold_count_total": 6,
      }
    }
  },
  "per_loto_summary": {
    "<loto_type>": { ...comparison_summary... }
  },
  "variant_ranking": {
    "by_logloss": [{"rank": 1, "variant": "...", "mean": ...}, ...],
    "by_brier":   [...],
    "by_ece":     [...],
    "promote_counts": {"<name>": {"promote_count": 0, "hold_count": 6, "promote_rate": 0.0}},
    "calibration_recommendations": {"<name>": {"none": 4, ...}},
  },
  "pairwise_comparison_summary": {
    "<key>": {
      "per_loto": {"<loto_type>": {"run_count": 3, "ci_wins": 1, ...}},
      "overall":  {"run_count": 9, "ci_wins": 2, ...},
    }
  },
  "promotion_recommendation_summary": {
    "<name>": {
      "promoted_in": [...],
      "held_in": [...],
      "consistent_promote": false,
      "consistent_hold": true,
    }
  },
}

Schema overview (recommendation.json):
{
  "schema_version": 1,
  "generated_at": "...",
  "based_on": "cross_loto_summary",
  "recommended_next_action": "hold",
  "recommended_challenger": null,
  "keep_production_as_is": true,
  "evidence_summary": {...},
  "blockers_to_promotion": [...],
  "whether_to_try_pma_or_isab_next": false,
  "next_experiment_recommendations": [...],
}
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

CROSS_LOTO_SUMMARY_SCHEMA_VERSION = 1
RECOMMENDATION_SCHEMA_VERSION = 1

# Pairwise keys of interest (same set used in comparison_summary.py)
PAIRWISE_KEYS_OF_INTEREST = [
    "settransformer_vs_deepsets",
    "settransformer_vs_multihot",
    "deepsets_vs_multihot",
    "settransformer_vs_legacy",
    "deepsets_vs_legacy",
    "multihot_vs_legacy",
]

# Promotion threshold: a variant is "consistently promoted" if it promoted in
# at least this fraction of loto_types (rounded down).
CONSISTENT_PROMOTE_THRESHOLD = 0.5

# Pairwise "signal" threshold: overall both_pass_count / run_count >= this
# value signals a meaningful advantage for whether_to_try_pma_or_isab_next.
PAIRWISE_SIGNAL_THRESHOLD = 0.5


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _safe_mean_std(values: list[float]) -> tuple[float | None, float | None]:
    """Return (mean, std) of a non-empty list, or (None, None)."""
    if not values:
        return None, None
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def build_overall_variant_summary(
    per_loto_summaries: dict[str, dict],
) -> dict[str, Any]:
    """Aggregate per-variant logloss/brier/ece/promote/hold across loto_types."""
    # Collect all variant names
    all_variants: set[str] = set()
    for summary in per_loto_summaries.values():
        all_variants.update((summary.get("variants") or {}).keys())

    result: dict[str, Any] = {}
    for variant in sorted(all_variants):
        loto_types_eval: list[str] = []
        logloss_per_loto: dict[str, float | None] = {}
        brier_per_loto: dict[str, float | None] = {}
        ece_per_loto: dict[str, float | None] = {}
        promote_total = 0
        hold_total = 0

        for loto_type, summary in per_loto_summaries.items():
            variant_data = (summary.get("variants") or {}).get(variant)
            if variant_data is None:
                continue
            loto_types_eval.append(loto_type)

            logloss_per_loto[loto_type] = _float_or_none(
                (variant_data.get("logloss") or {}).get("mean")
            )
            brier_per_loto[loto_type] = _float_or_none(
                (variant_data.get("brier") or {}).get("mean")
            )
            ece_per_loto[loto_type] = _float_or_none(
                (variant_data.get("ece") or {}).get("mean")
            )
            promote_total += int(variant_data.get("promote_count") or 0)
            hold_total += int(variant_data.get("hold_count") or 0)

        logloss_vals = [v for v in logloss_per_loto.values() if v is not None]
        brier_vals = [v for v in brier_per_loto.values() if v is not None]
        ece_vals = [v for v in ece_per_loto.values() if v is not None]

        logloss_mean, logloss_std = _safe_mean_std(logloss_vals)
        brier_mean, brier_std = _safe_mean_std(brier_vals)
        ece_mean, ece_std = _safe_mean_std(ece_vals)

        result[variant] = {
            "loto_types_evaluated": sorted(loto_types_eval),
            "logloss": {"mean": logloss_mean, "std": logloss_std, "per_loto": logloss_per_loto},
            "brier": {"mean": brier_mean, "std": brier_std, "per_loto": brier_per_loto},
            "ece": {"mean": ece_mean, "std": ece_std, "per_loto": ece_per_loto},
            "promote_count_total": promote_total,
            "hold_count_total": hold_total,
        }

    return result


def _rank_variants_by_metric(
    overall_variants: dict[str, Any], metric_key: str
) -> list[dict[str, Any]]:
    """Return a list sorted by metric mean (ascending, lower=better), None last."""
    items = []
    for variant, data in overall_variants.items():
        mean_val = _float_or_none((data.get(metric_key) or {}).get("mean"))
        items.append({"variant": variant, "mean": mean_val})

    # Sort: None values go to end, otherwise ascending
    items.sort(key=lambda x: (x["mean"] is None, x["mean"] if x["mean"] is not None else 0.0))
    return [{"rank": i + 1, **item} for i, item in enumerate(items)]


def build_variant_ranking(
    overall_variants: dict[str, Any],
    per_loto_summaries: dict[str, dict],
) -> dict[str, Any]:
    """Build ranking summary by logloss/brier/ece plus promote_counts and calibration."""
    promote_counts: dict[str, Any] = {}
    calibration_recs: dict[str, dict[str, int]] = {}

    for variant, data in overall_variants.items():
        promote_total = data.get("promote_count_total", 0)
        hold_total = data.get("hold_count_total", 0)
        total = promote_total + hold_total
        rate = promote_total / total if total > 0 else 0.0
        promote_counts[variant] = {
            "promote_count": promote_total,
            "hold_count": hold_total,
            "promote_rate": round(rate, 4),
        }

        # Aggregate calibration recommendations across loto_types
        merged_recs: dict[str, int] = {}
        for summary in per_loto_summaries.values():
            variant_data = (summary.get("variants") or {}).get(variant) or {}
            for method, count in (variant_data.get("calibration_recommendations") or {}).items():
                merged_recs[method] = merged_recs.get(method, 0) + count
        calibration_recs[variant] = merged_recs

    return {
        "by_logloss": _rank_variants_by_metric(overall_variants, "logloss"),
        "by_brier": _rank_variants_by_metric(overall_variants, "brier"),
        "by_ece": _rank_variants_by_metric(overall_variants, "ece"),
        "promote_counts": promote_counts,
        "calibration_recommendations": calibration_recs,
    }


def build_pairwise_comparison_summary(
    per_loto_summaries: dict[str, dict],
) -> dict[str, Any]:
    """Aggregate pairwise comparisons per_loto and as an overall total."""
    # Collect all comparison keys present in any loto_type
    all_keys: set[str] = set()
    for summary in per_loto_summaries.values():
        pairwise = summary.get("pairwise_comparisons") or {}
        all_keys.update(pairwise.keys())

    # Order: keys of interest first, then alphabetical
    ordered_keys: list[str] = [k for k in PAIRWISE_KEYS_OF_INTEREST if k in all_keys]
    for k in sorted(all_keys):
        if k not in ordered_keys:
            ordered_keys.append(k)

    result: dict[str, Any] = {}
    for key in ordered_keys:
        per_loto: dict[str, Any] = {}
        overall_run_count = 0
        overall_ci_wins = 0
        overall_perm_wins = 0
        overall_both_pass = 0

        for loto_type, summary in per_loto_summaries.items():
            pairwise = summary.get("pairwise_comparisons") or {}
            comp = pairwise.get(key)
            if comp is None:
                continue
            run_count = int(comp.get("run_count") or 0)
            ci_wins = int(comp.get("ci_wins") or 0)
            perm_wins = int(comp.get("permutation_wins") or 0)
            both_pass = int(comp.get("both_pass_count") or 0)
            per_loto[loto_type] = {
                "run_count": run_count,
                "ci_wins": ci_wins,
                "permutation_wins": perm_wins,
                "both_pass_count": both_pass,
            }
            overall_run_count += run_count
            overall_ci_wins += ci_wins
            overall_perm_wins += perm_wins
            overall_both_pass += both_pass

        result[key] = {
            "per_loto": per_loto,
            "overall": {
                "run_count": overall_run_count,
                "ci_wins": overall_ci_wins,
                "permutation_wins": overall_perm_wins,
                "both_pass_count": overall_both_pass,
            },
        }

    return result


def build_promotion_recommendation_summary(
    overall_variants: dict[str, Any],
    per_loto_summaries: dict[str, dict],
) -> dict[str, Any]:
    """Summarise which loto_types promoted/held each variant."""
    result: dict[str, Any] = {}

    for variant in sorted(overall_variants.keys()):
        promoted_in: list[str] = []
        held_in: list[str] = []

        for loto_type, summary in per_loto_summaries.items():
            variant_data = (summary.get("variants") or {}).get(variant)
            if variant_data is None:
                continue
            promote_count = int(variant_data.get("promote_count") or 0)
            hold_count = int(variant_data.get("hold_count") or 0)
            if promote_count > 0:
                promoted_in.append(loto_type)
            if hold_count > 0:
                held_in.append(loto_type)

        num_evaluated = len((overall_variants.get(variant) or {}).get("loto_types_evaluated", []))
        consistent_promote = (
            len(promoted_in) >= max(1, round(num_evaluated * CONSISTENT_PROMOTE_THRESHOLD + 0.5))
            if num_evaluated > 0
            else False
        )
        consistent_hold = len(held_in) == num_evaluated and num_evaluated > 0

        result[variant] = {
            "promoted_in": sorted(promoted_in),
            "held_in": sorted(held_in),
            "consistent_promote": consistent_promote,
            "consistent_hold": consistent_hold,
        }

    return result


def build_cross_loto_summary(
    per_loto_summaries: dict[str, dict],
    loto_types: list[str],
    preset: str,
    seeds: list[int],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Build a full cross-loto summary from a mapping of loto_type → comparison_summary."""
    run_count_per_loto = {
        lt: int((per_loto_summaries.get(lt) or {}).get("run_count") or 0)
        for lt in loto_types
    }
    available = {lt: s for lt, s in per_loto_summaries.items() if lt in loto_types}

    overall_variants = build_overall_variant_summary(available)
    variant_ranking = build_variant_ranking(overall_variants, available)
    pairwise_summary = build_pairwise_comparison_summary(available)
    promotion_summary = build_promotion_recommendation_summary(overall_variants, available)

    return {
        "schema_version": CROSS_LOTO_SUMMARY_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "loto_types": sorted(loto_types),
        "preset": preset,
        "seeds": list(seeds),
        "alpha": alpha,
        "run_count_per_loto": run_count_per_loto,
        "overall_summary": {"variants": overall_variants},
        "per_loto_summary": {lt: per_loto_summaries.get(lt) or {} for lt in sorted(loto_types)},
        "variant_ranking": variant_ranking,
        "pairwise_comparison_summary": pairwise_summary,
        "promotion_recommendation_summary": promotion_summary,
    }


# ---------------------------------------------------------------------------
# Recommendation artifact
# ---------------------------------------------------------------------------


def _determine_next_action(
    variant_ranking: dict[str, Any],
    pairwise_summary: dict[str, Any],
    promotion_summary: dict[str, Any],
) -> str:
    """Decide the recommended next action string."""
    # Check if any variant consistently promotes across loto_types
    for variant, promo in promotion_summary.items():
        if variant == "legacy":
            continue
        if promo.get("consistent_promote"):
            return "consider_promotion"

    # Check if pairwise signals are mixed (some ci_wins but not consistent)
    has_any_signal = False
    for key, comp in pairwise_summary.items():
        overall = comp.get("overall") or {}
        run_count = int(overall.get("run_count") or 0)
        both_pass = int(overall.get("both_pass_count") or 0)
        if run_count > 0 and both_pass / run_count >= PAIRWISE_SIGNAL_THRESHOLD:
            has_any_signal = True
            break

    if has_any_signal:
        return "run_more_seeds"

    return "hold"


def _find_best_challenger(
    variant_ranking: dict[str, Any],
    promotion_summary: dict[str, Any],
) -> str | None:
    """Return the best non-legacy challenger by logloss rank."""
    for entry in variant_ranking.get("by_logloss") or []:
        variant = entry.get("variant")
        if variant and variant != "legacy":
            return variant
    return None


def _check_pma_isab_signal(pairwise_summary: dict[str, Any]) -> bool:
    """Return True if settransformer consistently beats deepsets (attention helps)."""
    comp = (pairwise_summary.get("settransformer_vs_deepsets") or {}).get("overall") or {}
    run_count = int(comp.get("run_count") or 0)
    both_pass = int(comp.get("both_pass_count") or 0)
    if run_count == 0:
        return False
    return both_pass / run_count >= PAIRWISE_SIGNAL_THRESHOLD


def build_recommendation(cross_loto_summary: dict[str, Any]) -> dict[str, Any]:
    """Build a recommendation artifact from the cross-loto summary."""
    variant_ranking = cross_loto_summary.get("variant_ranking") or {}
    pairwise_summary = cross_loto_summary.get("pairwise_comparison_summary") or {}
    promotion_summary = cross_loto_summary.get("promotion_recommendation_summary") or {}
    overall_variants = (cross_loto_summary.get("overall_summary") or {}).get("variants") or {}

    next_action = _determine_next_action(variant_ranking, pairwise_summary, promotion_summary)
    best_challenger = _find_best_challenger(variant_ranking, promotion_summary)
    pma_signal = _check_pma_isab_signal(pairwise_summary)

    keep_production = next_action not in ("consider_promotion",)

    # Blockers
    blockers: list[str] = []
    consistently_promoting = [
        v for v, p in promotion_summary.items()
        if p.get("consistent_promote") and v != "legacy"
    ]
    if not consistently_promoting:
        blockers.append(
            "No variant consistently passed promotion guardrails across loto_types."
        )
    st_vs_ds = (pairwise_summary.get("settransformer_vs_deepsets") or {}).get("overall") or {}
    if int(st_vs_ds.get("both_pass_count") or 0) == 0:
        blockers.append(
            "settransformer_vs_deepsets: no run showed both CI and permutation significance."
        )

    # Next experiment recommendations
    next_experiments: list[str] = []
    loto_types = cross_loto_summary.get("loto_types") or []
    seeds = cross_loto_summary.get("seeds") or []
    preset = cross_loto_summary.get("preset") or "archcomp"

    if next_action == "hold":
        if len(seeds) < 5:
            next_experiments.append(
                f"Increase seed count (currently {len(seeds)}) to improve confidence intervals."
            )
        if preset != "default":
            next_experiments.append(
                "Run with --preset default (more epochs) to allow better architecture differentiation."
            )
        if len(loto_types) < 3:
            next_experiments.append(
                "Run cross-loto comparison across all 3 loto_types (miniloto, loto6, loto7)."
            )
    elif next_action == "run_more_seeds":
        next_experiments.append(
            "Increase seed count to ≥5 and re-run cross-loto comparison."
        )
    elif next_action == "consider_promotion":
        promotable = [v for v, p in promotion_summary.items() if p.get("consistent_promote")]
        next_experiments.append(
            f"Run production training for candidate variant(s): {', '.join(promotable)}."
        )

    if pma_signal:
        next_experiments.append(
            "settransformer shows advantage over deepsets — PMA / ISAB extension is worth exploring."
        )
    else:
        next_experiments.append(
            "settransformer vs deepsets advantage is not yet clear. "
            "PMA / ISAB exploration should wait until settransformer shows consistent benefit."
        )

    # Evidence summary
    best_by_logloss = (variant_ranking.get("by_logloss") or [{}])[0].get("variant")
    evidence = {
        "best_variant_by_logloss": best_by_logloss,
        "consistent_promote_variants": consistently_promoting,
        "pairwise_clear_winner": (
            "settransformer" if _check_pma_isab_signal(pairwise_summary) else None
        ),
        "total_runs_across_loto_types": sum(
            cross_loto_summary.get("run_count_per_loto", {}).values()
        ),
    }

    return {
        "schema_version": RECOMMENDATION_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "based_on": "cross_loto_summary",
        "recommended_next_action": next_action,
        "recommended_challenger": best_challenger,
        "keep_production_as_is": keep_production,
        "evidence_summary": evidence,
        "blockers_to_promotion": blockers,
        "whether_to_try_pma_or_isab_next": pma_signal,
        "next_experiment_recommendations": next_experiments,
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_comparison_summary(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(payload: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def get_cross_loto_summary_path(data_dir: str = "data") -> Path:
    return Path(data_dir) / "cross_loto_summary.json"


def get_recommendation_path(data_dir: str = "data") -> Path:
    return Path(data_dir) / "recommendation.json"
