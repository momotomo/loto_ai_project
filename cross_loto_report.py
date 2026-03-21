"""cross_loto_report.py

Generates human-readable Markdown reports and CSV artifacts from cross-loto
summary and recommendation artifacts.

Output artifacts (all written under --data_dir):
  data/cross_loto_report.md       — full Markdown evidence pack
  data/variant_metrics.csv        — per-variant logloss/brier/ece across loto_types
  data/pairwise_comparisons.csv   — aggregated pairwise test results
  data/recommendation_summary.csv — promote/hold counts and next-action summary

Decision rules documented here and in the Markdown report:
  hold             : No variant passed promotion guardrails in ≥50% of loto_types
                     AND no pairwise signal (both_pass_count/run_count < 0.5).
  run_more_seeds   : No consistent promotion, but at least one pairwise pair shows
                     both_pass_count/run_count ≥ 0.5 — signal is present but weak.
  consider_promotion: At least one non-legacy variant promoted in ≥50% of loto_types
                      across all evaluated loto_types.
  PMA/ISAB next    : settransformer_vs_deepsets overall both_pass_count/run_count ≥ 0.5
                     (attention benefit confirmed; exploring deeper pooling is warranted).

Thresholds (kept in sync with cross_loto_summary.py):
  CONSISTENT_PROMOTE_THRESHOLD = 0.5
  PAIRWISE_SIGNAL_THRESHOLD    = 0.5
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CROSS_LOTO_REPORT_SCHEMA_VERSION = 1

# Human-readable threshold descriptions (mirrored from cross_loto_summary.py)
CONSISTENT_PROMOTE_THRESHOLD = 0.5
PAIRWISE_SIGNAL_THRESHOLD = 0.5

_DECISION_RULES_MD = """\
## Decision Rules

| Condition | Next Action |
|-----------|-------------|
| At least one non-legacy variant promoted in ≥50% of loto_types | `consider_promotion` |
| No consistent promotion, but some pairwise pair has both_pass_count/run_count ≥ 0.5 | `run_more_seeds` |
| Neither condition above is met | `hold` |

**Promotion to production requires:**
- `consider_promotion` verdict from cross-loto summary
- Manual review of per-loto details and calibration recommendation

**PMA / ISAB exploration requires:**
- `settransformer_vs_deepsets` overall `both_pass_count / run_count ≥ 0.5`
  (confirms attention benefit before investing in deeper pooling architectures)

**Thresholds:**
- `CONSISTENT_PROMOTE_THRESHOLD = 0.5` — variant promoted in ≥ ceil(N × 0.5) loto_types
- `PAIRWISE_SIGNAL_THRESHOLD = 0.5` — both_pass_count / run_count for PMA/ISAB signal
"""


# ---------------------------------------------------------------------------
# Markdown report builder
# ---------------------------------------------------------------------------


def _fmt(value: float | None, decimals: int = 4) -> str:
    return f"{value:.{decimals}f}" if value is not None else "N/A"


def _pct(value: float | None) -> str:
    return f"{value * 100:.1f}%" if value is not None else "N/A"


def build_markdown_report(
    cross_loto_summary: dict[str, Any],
    recommendation: dict[str, Any],
    run_metadata: dict[str, Any] | None = None,
) -> str:
    """Build a human-readable Markdown evidence pack."""
    lines: list[str] = []

    generated_at = cross_loto_summary.get("generated_at") or datetime.now(timezone.utc).isoformat()
    loto_types = cross_loto_summary.get("loto_types") or []
    preset = cross_loto_summary.get("preset") or "?"
    seeds = cross_loto_summary.get("seeds") or []
    alpha = cross_loto_summary.get("alpha") or 0.05
    run_count_per_loto = cross_loto_summary.get("run_count_per_loto") or {}

    # --- Header ---
    lines.append("# Cross-Loto Architecture Comparison Report")
    lines.append("")
    lines.append(f"Generated: {generated_at}")
    lines.append("")

    # --- Execution Conditions ---
    lines.append("## Execution Conditions")
    lines.append("")
    lines.append(f"- **loto_types**: {', '.join(loto_types) if loto_types else 'N/A'}")
    lines.append(f"- **preset**: {preset}")
    lines.append(f"- **seeds**: {seeds}")
    lines.append(f"- **alpha**: {alpha}")

    if run_metadata:
        ev = run_metadata.get("evaluation_model_variants") or []
        if isinstance(ev, list):
            ev_str = ", ".join(ev)
        else:
            ev_str = str(ev)
        lines.append(f"- **evaluation_model_variants**: {ev_str}")
        run_id = run_metadata.get("run_id")
        if run_id:
            lines.append(f"- **run_id**: {run_id}")
        source_paths = run_metadata.get("source_summary_paths") or {}
        if source_paths:
            lines.append("- **source_summary_paths**:")
            for lt, path in sorted(source_paths.items()):
                lines.append(f"  - `{lt}`: `{path}`")

    lines.append("")
    lines.append("**Runs per loto_type:**")
    lines.append("")
    lines.append("| loto_type | run_count |")
    lines.append("|-----------|-----------|")
    for lt in sorted(loto_types):
        lines.append(f"| {lt} | {run_count_per_loto.get(lt, 0)} |")
    lines.append("")

    # --- Variant Metrics Summary ---
    lines.append("## Variant Metrics Summary")
    lines.append("")
    lines.append(
        "Metrics are averaged across all loto_types evaluated. Lower is better for logloss/brier/ece."
    )
    lines.append("")
    lines.append("| rank | variant | logloss (mean±std) | brier (mean±std) | ece (mean±std) | loto_types |")
    lines.append("|------|---------|-------------------|-----------------|---------------|------------|")

    overall_variants = (cross_loto_summary.get("overall_summary") or {}).get("variants") or {}
    variant_ranking = cross_loto_summary.get("variant_ranking") or {}
    by_logloss = variant_ranking.get("by_logloss") or []

    for entry in by_logloss:
        rank = entry.get("rank", "?")
        v = entry.get("variant", "?")
        vdata = overall_variants.get(v) or {}
        ll = vdata.get("logloss") or {}
        br = vdata.get("brier") or {}
        ec = vdata.get("ece") or {}
        lts = vdata.get("loto_types_evaluated") or []
        ll_str = f"{_fmt(ll.get('mean'))}±{_fmt(ll.get('std'))}"
        br_str = f"{_fmt(br.get('mean'))}±{_fmt(br.get('std'))}"
        ec_str = f"{_fmt(ec.get('mean'))}±{_fmt(ec.get('std'))}"
        lines.append(f"| {rank} | **{v}** | {ll_str} | {br_str} | {ec_str} | {', '.join(lts)} |")
    lines.append("")

    # --- Promote / Hold Summary ---
    lines.append("## Promote / Hold Summary")
    lines.append("")
    lines.append("| variant | promote_count | hold_count | promote_rate | consistent_promote | consistent_hold |")
    lines.append("|---------|--------------|-----------|-------------|-------------------|----------------|")

    promotion_summary = cross_loto_summary.get("promotion_recommendation_summary") or {}
    promote_counts = variant_ranking.get("promote_counts") or {}

    for entry in by_logloss:
        v = entry.get("variant", "?")
        pc = promote_counts.get(v) or {}
        prom_count = pc.get("promote_count", 0)
        hold_count = pc.get("hold_count", 0)
        rate = pc.get("promote_rate", 0.0)
        promo_summary = promotion_summary.get(v) or {}
        cons_prom = "✓" if promo_summary.get("consistent_promote") else "✗"
        cons_hold = "✓" if promo_summary.get("consistent_hold") else "✗"
        promoted_in = ", ".join(promo_summary.get("promoted_in") or []) or "—"
        lines.append(
            f"| **{v}** | {prom_count} | {hold_count} | {_pct(rate)} | {cons_prom} | {cons_hold} |"
        )
    lines.append("")

    # Per-variant promoted_in details
    for v in sorted(promotion_summary.keys()):
        ps = promotion_summary[v]
        promoted_in = ps.get("promoted_in") or []
        held_in = ps.get("held_in") or []
        if promoted_in:
            lines.append(f"- **{v}** promoted in: {', '.join(promoted_in)}")
        if held_in:
            lines.append(f"- **{v}** held in: {', '.join(held_in)}")
    lines.append("")

    # --- Pairwise Comparison Summary ---
    lines.append("## Pairwise Comparison Summary")
    lines.append("")
    lines.append(
        "**ci_wins**: runs where bootstrap CI upper bound < 0 (challenger better than baseline).  \n"
        "**perm_wins**: runs where permutation test p-value < alpha.  \n"
        "**both_pass**: runs where both CI and permutation test pass simultaneously."
    )
    lines.append("")
    lines.append("| comparison | ci_wins / runs | perm_wins / runs | both_pass / runs | both_pass rate |")
    lines.append("|------------|---------------|-----------------|-----------------|---------------|")

    pairwise = cross_loto_summary.get("pairwise_comparison_summary") or {}
    for key, comp in pairwise.items():
        overall = comp.get("overall") or {}
        run_count = int(overall.get("run_count") or 0)
        ci_wins = int(overall.get("ci_wins") or 0)
        perm_wins = int(overall.get("permutation_wins") or 0)
        both_pass = int(overall.get("both_pass_count") or 0)
        rate = both_pass / run_count if run_count > 0 else 0.0
        signal_marker = " ⭐" if rate >= PAIRWISE_SIGNAL_THRESHOLD else ""
        lines.append(
            f"| {key} | {ci_wins}/{run_count} | {perm_wins}/{run_count} | "
            f"{both_pass}/{run_count} | {_pct(rate)}{signal_marker} |"
        )
    lines.append("")
    lines.append("⭐ = both_pass_count / run_count ≥ 0.5 (meaningful signal)")
    lines.append("")

    # Per-loto pairwise breakdown
    lines.append("### Per-Loto Pairwise Breakdown")
    lines.append("")
    for key, comp in pairwise.items():
        per_loto = comp.get("per_loto") or {}
        if not per_loto:
            continue
        lines.append(f"**{key}**")
        lines.append("")
        lines.append("| loto_type | ci_wins / runs | perm_wins / runs | both_pass / runs |")
        lines.append("|-----------|---------------|-----------------|-----------------|")
        for lt in sorted(per_loto.keys()):
            pl = per_loto[lt]
            rc = int(pl.get("run_count") or 0)
            cw = int(pl.get("ci_wins") or 0)
            pw = int(pl.get("permutation_wins") or 0)
            bp = int(pl.get("both_pass_count") or 0)
            lines.append(f"| {lt} | {cw}/{rc} | {pw}/{rc} | {bp}/{rc} |")
        lines.append("")

    # --- Recommendation ---
    lines.append("## Recommendation")
    lines.append("")
    next_action = recommendation.get("recommended_next_action") or "N/A"
    challenger = recommendation.get("recommended_challenger") or "N/A"
    keep_prod = recommendation.get("keep_production_as_is")
    pma_next = recommendation.get("whether_to_try_pma_or_isab_next")

    lines.append(f"**Recommended next action**: `{next_action}`")
    lines.append(f"**Recommended challenger**: {challenger}")
    lines.append(f"**Keep production as-is**: {'Yes' if keep_prod else 'No — promotion candidate exists'}")
    lines.append(f"**Whether to try PMA / ISAB next**: {'Yes ✓' if pma_next else 'No ✗'}")
    lines.append("")

    evidence = recommendation.get("evidence_summary") or {}
    lines.append("### Evidence Summary")
    lines.append("")
    lines.append(f"- Best variant by logloss: **{evidence.get('best_variant_by_logloss') or 'N/A'}**")
    lines.append(f"- Consistently promoting variants: {evidence.get('consistent_promote_variants') or []}")
    lines.append(f"- Clear pairwise winner: {evidence.get('pairwise_clear_winner') or 'None'}")
    lines.append(f"- Total runs across loto_types: {evidence.get('total_runs_across_loto_types') or 0}")
    lines.append("")

    blockers = recommendation.get("blockers_to_promotion") or []
    if blockers:
        lines.append("### Blockers to Promotion")
        lines.append("")
        for b in blockers:
            lines.append(f"- {b}")
        lines.append("")

    next_exps = recommendation.get("next_experiment_recommendations") or []
    if next_exps:
        lines.append("### Next Experiment Recommendations")
        lines.append("")
        for r in next_exps:
            lines.append(f"- {r}")
        lines.append("")

    # --- Production Change Rationale ---
    lines.append("## Production Change Rationale")
    lines.append("")
    if keep_prod:
        lines.append(
            "**Production model is NOT changed** by this comparison run.  \n"
            "Final training (`--no_skip_final_train`) was either skipped (default) or no variant "
            "met the consistent-promotion threshold across loto_types."
        )
        lines.append("")
        lines.append("To change production, you would need:")
        lines.append("1. A `consider_promotion` verdict from this report")
        lines.append("2. Manual confirmation of the candidate variant")
        lines.append("3. Re-run with `--no_skip_final_train` for the chosen `--model_variant`")
    else:
        lines.append(
            "**A production change is indicated.**  \n"
            "At least one variant has consistently passed promotion guardrails across loto_types.  \n"
            "Review the promotion candidate and run with `--no_skip_final_train` to update artifacts."
        )
    lines.append("")

    # --- Decision Rules ---
    lines.append(_DECISION_RULES_MD)

    # --- Calibration Recommendations ---
    lines.append("## Calibration Recommendations")
    lines.append("")
    calib_recs = variant_ranking.get("calibration_recommendations") or {}
    lines.append("| variant | none | temperature | isotonic |")
    lines.append("|---------|------|-------------|---------|")
    for v in sorted(calib_recs.keys()):
        cr = calib_recs[v]
        none_cnt = cr.get("none", 0)
        temp_cnt = cr.get("temperature", 0)
        iso_cnt = cr.get("isotonic", 0)
        lines.append(f"| **{v}** | {none_cnt} | {temp_cnt} | {iso_cnt} |")
    lines.append("")

    # --- Footer ---
    lines.append("---")
    lines.append(f"*Report generated by cross_loto_report.py v{CROSS_LOTO_REPORT_SCHEMA_VERSION}*")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSV builders
# ---------------------------------------------------------------------------


def build_variant_metrics_csv(cross_loto_summary: dict[str, Any]) -> str:
    """Build CSV with per-variant metrics across loto_types.

    Required columns: variant, logloss_mean, logloss_std, brier_mean, brier_std,
                      ece_mean, ece_std, promote_count_total, hold_count_total,
                      promote_rate, consistent_promote, consistent_hold, loto_types_evaluated
    """
    overall_variants = (cross_loto_summary.get("overall_summary") or {}).get("variants") or {}
    variant_ranking = cross_loto_summary.get("variant_ranking") or {}
    promote_counts = variant_ranking.get("promote_counts") or {}
    promotion_summary = cross_loto_summary.get("promotion_recommendation_summary") or {}

    by_logloss = variant_ranking.get("by_logloss") or []
    ordered_variants = [e["variant"] for e in by_logloss if "variant" in e]
    # Add any not in ranking
    for v in sorted(overall_variants.keys()):
        if v not in ordered_variants:
            ordered_variants.append(v)

    fieldnames = [
        "rank",
        "variant",
        "logloss_mean",
        "logloss_std",
        "brier_mean",
        "brier_std",
        "ece_mean",
        "ece_std",
        "promote_count_total",
        "hold_count_total",
        "promote_rate",
        "consistent_promote",
        "consistent_hold",
        "loto_types_evaluated",
    ]

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()

    rank_map = {e["variant"]: e["rank"] for e in by_logloss}
    for v in ordered_variants:
        vdata = overall_variants.get(v) or {}
        ll = vdata.get("logloss") or {}
        br = vdata.get("brier") or {}
        ec = vdata.get("ece") or {}
        pc = promote_counts.get(v) or {}
        ps = promotion_summary.get(v) or {}

        writer.writerow({
            "rank": rank_map.get(v, ""),
            "variant": v,
            "logloss_mean": ll.get("mean") if ll.get("mean") is not None else "",
            "logloss_std": ll.get("std") if ll.get("std") is not None else "",
            "brier_mean": br.get("mean") if br.get("mean") is not None else "",
            "brier_std": br.get("std") if br.get("std") is not None else "",
            "ece_mean": ec.get("mean") if ec.get("mean") is not None else "",
            "ece_std": ec.get("std") if ec.get("std") is not None else "",
            "promote_count_total": pc.get("promote_count", 0),
            "hold_count_total": pc.get("hold_count", 0),
            "promote_rate": pc.get("promote_rate", 0.0),
            "consistent_promote": "true" if ps.get("consistent_promote") else "false",
            "consistent_hold": "true" if ps.get("consistent_hold") else "false",
            "loto_types_evaluated": "|".join(vdata.get("loto_types_evaluated") or []),
        })

    return buf.getvalue()


def build_pairwise_comparisons_csv(cross_loto_summary: dict[str, Any]) -> str:
    """Build CSV with pairwise comparison results.

    Required columns: comparison_key, scope, loto_type, run_count,
                      ci_wins, permutation_wins, both_pass_count, both_pass_rate
    """
    pairwise = cross_loto_summary.get("pairwise_comparison_summary") or {}

    fieldnames = [
        "comparison_key",
        "scope",
        "loto_type",
        "run_count",
        "ci_wins",
        "permutation_wins",
        "both_pass_count",
        "both_pass_rate",
    ]

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()

    for key, comp in pairwise.items():
        # Per-loto rows
        per_loto = comp.get("per_loto") or {}
        for lt in sorted(per_loto.keys()):
            pl = per_loto[lt]
            rc = int(pl.get("run_count") or 0)
            cw = int(pl.get("ci_wins") or 0)
            pw = int(pl.get("permutation_wins") or 0)
            bp = int(pl.get("both_pass_count") or 0)
            rate = bp / rc if rc > 0 else 0.0
            writer.writerow({
                "comparison_key": key,
                "scope": "per_loto",
                "loto_type": lt,
                "run_count": rc,
                "ci_wins": cw,
                "permutation_wins": pw,
                "both_pass_count": bp,
                "both_pass_rate": round(rate, 4),
            })
        # Overall row
        overall = comp.get("overall") or {}
        rc = int(overall.get("run_count") or 0)
        cw = int(overall.get("ci_wins") or 0)
        pw = int(overall.get("permutation_wins") or 0)
        bp = int(overall.get("both_pass_count") or 0)
        rate = bp / rc if rc > 0 else 0.0
        writer.writerow({
            "comparison_key": key,
            "scope": "overall",
            "loto_type": "ALL",
            "run_count": rc,
            "ci_wins": cw,
            "permutation_wins": pw,
            "both_pass_count": bp,
            "both_pass_rate": round(rate, 4),
        })

    return buf.getvalue()


def build_recommendation_summary_csv(
    cross_loto_summary: dict[str, Any],
    recommendation: dict[str, Any],
) -> str:
    """Build a single-row CSV summarising the recommendation.

    Required columns: generated_at, loto_types, preset, seeds, run_count_total,
                      recommended_next_action, recommended_challenger,
                      keep_production_as_is, whether_to_try_pma_or_isab_next,
                      best_variant_by_logloss, consistent_promote_variants,
                      blockers_count, next_experiment_count
    """
    loto_types = cross_loto_summary.get("loto_types") or []
    seeds = cross_loto_summary.get("seeds") or []
    run_count_per_loto = cross_loto_summary.get("run_count_per_loto") or {}
    run_count_total = sum(run_count_per_loto.values())

    evidence = recommendation.get("evidence_summary") or {}
    consistent_promote = evidence.get("consistent_promote_variants") or []

    fieldnames = [
        "generated_at",
        "loto_types",
        "preset",
        "seeds",
        "run_count_total",
        "recommended_next_action",
        "recommended_challenger",
        "keep_production_as_is",
        "whether_to_try_pma_or_isab_next",
        "best_variant_by_logloss",
        "consistent_promote_variants",
        "blockers_count",
        "next_experiment_count",
    ]

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({
        "generated_at": cross_loto_summary.get("generated_at") or "",
        "loto_types": "|".join(loto_types),
        "preset": cross_loto_summary.get("preset") or "",
        "seeds": "|".join(str(s) for s in seeds),
        "run_count_total": run_count_total,
        "recommended_next_action": recommendation.get("recommended_next_action") or "",
        "recommended_challenger": recommendation.get("recommended_challenger") or "",
        "keep_production_as_is": "true" if recommendation.get("keep_production_as_is") else "false",
        "whether_to_try_pma_or_isab_next": "true" if recommendation.get("whether_to_try_pma_or_isab_next") else "false",
        "best_variant_by_logloss": evidence.get("best_variant_by_logloss") or "",
        "consistent_promote_variants": "|".join(consistent_promote),
        "blockers_count": len(recommendation.get("blockers_to_promotion") or []),
        "next_experiment_count": len(recommendation.get("next_experiment_recommendations") or []),
    })

    return buf.getvalue()


# ---------------------------------------------------------------------------
# Run metadata
# ---------------------------------------------------------------------------


def build_run_metadata(
    *,
    loto_types: list[str],
    preset: str,
    seeds: list[int],
    evaluation_model_variants: str | list[str],
    alpha: float = 0.05,
    source_summary_paths: dict[str, str] | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Build a run_metadata dict for tracking what went into this recommendation."""
    if isinstance(evaluation_model_variants, str):
        ev_list = [v.strip() for v in evaluation_model_variants.split(",") if v.strip()]
    else:
        ev_list = list(evaluation_model_variants)

    return {
        "schema_version": CROSS_LOTO_REPORT_SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "preset": preset,
        "seeds": list(seeds),
        "loto_types": sorted(loto_types),
        "evaluation_model_variants": ev_list,
        "alpha": alpha,
        "source_summary_paths": source_summary_paths or {},
    }


# ---------------------------------------------------------------------------
# Artifact I/O
# ---------------------------------------------------------------------------


def save_report_artifacts(
    cross_loto_summary: dict[str, Any],
    recommendation: dict[str, Any],
    data_dir: str | Path = "data",
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Generate and save Markdown + CSV artifacts.  Returns dict of {name: path}."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}

    # Markdown report
    md_content = build_markdown_report(cross_loto_summary, recommendation, run_metadata)
    md_path = data_dir / "cross_loto_report.md"
    md_path.write_text(md_content, encoding="utf-8")
    paths["cross_loto_report.md"] = str(md_path)

    # CSV: variant metrics
    vm_csv = build_variant_metrics_csv(cross_loto_summary)
    vm_path = data_dir / "variant_metrics.csv"
    vm_path.write_text(vm_csv, encoding="utf-8")
    paths["variant_metrics.csv"] = str(vm_path)

    # CSV: pairwise comparisons
    pw_csv = build_pairwise_comparisons_csv(cross_loto_summary)
    pw_path = data_dir / "pairwise_comparisons.csv"
    pw_path.write_text(pw_csv, encoding="utf-8")
    paths["pairwise_comparisons.csv"] = str(pw_path)

    # CSV: recommendation summary
    rec_csv = build_recommendation_summary_csv(cross_loto_summary, recommendation)
    rec_path = data_dir / "recommendation_summary.csv"
    rec_path.write_text(rec_csv, encoding="utf-8")
    paths["recommendation_summary.csv"] = str(rec_path)

    return paths


def load_cross_loto_artifacts(
    data_dir: str | Path = "data",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load cross_loto_summary.json and recommendation.json from data_dir."""
    data_dir = Path(data_dir)
    with open(data_dir / "cross_loto_summary.json", encoding="utf-8") as f:
        cross_summary = json.load(f)
    with open(data_dir / "recommendation.json", encoding="utf-8") as f:
        recommendation = json.load(f)
    return cross_summary, recommendation
