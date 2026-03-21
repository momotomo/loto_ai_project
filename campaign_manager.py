"""campaign_manager.py

Manages campaign history artifact, diff report generation, and recommendation
stability tracking for cross-loto comparison campaigns.

Output artifacts (all written under data_dir):
  data/campaign_history.json   — ordered list of all campaign entries + stability
  data/campaign_history.csv    — tabular summary for spreadsheet analysis
  data/campaign_diff_report.md — diff between the two most recent campaigns

Reading order
-------------
1. data/campaign_diff_report.md   — "what changed since last time?"
2. data/campaign_history.json     — stability trend across all campaigns
3. data/campaign_history.csv      — spreadsheet-friendly version of history
4. <campaign_dir>/cross_loto_report.md — detailed evidence pack for each campaign

Schema overview (campaign_history.json):
{
  "schema_version": 1,
  "generated_at": "...",
  "total_campaigns": 3,
  "recommendation_stability": {
    "total_campaigns": 3,
    "latest_action": "hold",
    "consecutive_same_action": 2,
    "consecutive_same_challenger": 3,
    "consecutive_keep_production": 3,
    "consecutive_run_more_seeds": 0,
    "consecutive_positive_signal_for_settransformer": 0,
    "consecutive_positive_signal_for_deepsets": 1,
  },
  "campaigns": [
    {
      "campaign_name": "2026-03-01_archcomp",
      "profile_name": "archcomp",
      "generated_at": "...",
      "started_at": "...",
      "finished_at": "...",
      "campaign_dir": "campaigns/2026-03-01_archcomp",
      "loto_types": ["loto6", "loto7", "miniloto"],
      "preset": "archcomp",
      "seeds": [42, 123, 456],
      "recommended_next_action": "hold",
      "recommended_challenger": "multihot",
      "keep_production_as_is": true,
      "whether_to_try_pma_or_isab_next": false,
      "best_variant_by_logloss": "multihot",
      "consistent_promote_variants": [],
      "blockers_count": 2,
      "variant_ranking_summary": [
        {"rank": 1, "variant": "multihot", "logloss_mean": 0.312},
        ...
      ],
      "key_pairwise_signals": {
        "settransformer_vs_deepsets": {
          "run_count": 9,
          "both_pass_count": 2,
          "both_pass_rate": 0.2222,
        },
        ...
      },
    },
    ...
  ],
}
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CAMPAIGN_HISTORY_SCHEMA_VERSION = 1
CAMPAIGN_DIFF_REPORT_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Campaign entry builder
# ---------------------------------------------------------------------------


def build_campaign_entry(
    campaign_name: str,
    profile_name: str,
    cross_loto_summary: dict[str, Any],
    recommendation: dict[str, Any],
    *,
    started_at: str | None = None,
    finished_at: str | None = None,
    campaign_dir: str | None = None,
) -> dict[str, Any]:
    """Build a campaign history entry from cross_loto summary and recommendation.

    This entry is designed to be self-contained enough to generate a diff report
    and stability summary without loading the full cross_loto_summary.json.
    """
    generated_at = cross_loto_summary.get("generated_at") or datetime.now(timezone.utc).isoformat()

    # Variant ranking summary (logloss-ranked)
    variant_ranking = cross_loto_summary.get("variant_ranking") or {}
    by_logloss = variant_ranking.get("by_logloss") or []
    ranking_summary = [
        {
            "rank": e.get("rank"),
            "variant": e.get("variant"),
            "logloss_mean": e.get("mean"),
        }
        for e in by_logloss
        if "variant" in e
    ]

    # Key pairwise signals (all available keys)
    pairwise = cross_loto_summary.get("pairwise_comparison_summary") or {}
    key_pairwise_signals: dict[str, Any] = {}
    for key, comp in pairwise.items():
        overall = comp.get("overall") or {}
        rc = int(overall.get("run_count") or 0)
        bp = int(overall.get("both_pass_count") or 0)
        key_pairwise_signals[key] = {
            "run_count": rc,
            "both_pass_count": bp,
            "both_pass_rate": round(bp / rc, 4) if rc > 0 else 0.0,
        }

    evidence = recommendation.get("evidence_summary") or {}

    return {
        "campaign_name": campaign_name,
        "profile_name": profile_name,
        "generated_at": generated_at,
        "started_at": started_at,
        "finished_at": finished_at,
        "campaign_dir": campaign_dir,
        "loto_types": cross_loto_summary.get("loto_types") or [],
        "preset": cross_loto_summary.get("preset") or "",
        "seeds": cross_loto_summary.get("seeds") or [],
        "recommended_next_action": recommendation.get("recommended_next_action") or "",
        "recommended_challenger": recommendation.get("recommended_challenger"),
        "keep_production_as_is": recommendation.get("keep_production_as_is", True),
        "whether_to_try_pma_or_isab_next": recommendation.get("whether_to_try_pma_or_isab_next", False),
        "best_variant_by_logloss": evidence.get("best_variant_by_logloss"),
        "consistent_promote_variants": evidence.get("consistent_promote_variants") or [],
        "blockers_count": len(recommendation.get("blockers_to_promotion") or []),
        "variant_ranking_summary": ranking_summary,
        "key_pairwise_signals": key_pairwise_signals,
    }


# ---------------------------------------------------------------------------
# Recommendation stability
# ---------------------------------------------------------------------------


def compute_recommendation_stability(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute how stable the recommendation has been across recent campaigns.

    Counts consecutive matching values from the most recent campaign backward.
    A streak of 3+ for run_more_seeds suggests running archcomp_full.
    A streak of 2+ for consider_promotion warrants careful promotion review.
    """
    if not history:
        return {
            "total_campaigns": 0,
            "latest_action": None,
            "latest_challenger": None,
            "consecutive_same_action": 0,
            "consecutive_same_challenger": 0,
            "consecutive_keep_production": 0,
            "consecutive_run_more_seeds": 0,
            "consecutive_positive_signal_for_settransformer": 0,
            "consecutive_positive_signal_for_deepsets": 0,
        }

    last_action = history[-1].get("recommended_next_action")
    last_challenger = history[-1].get("recommended_challenger")

    def _count_consecutive(key: str, target: Any) -> int:
        count = 0
        for entry in reversed(history):
            if entry.get(key) == target:
                count += 1
            else:
                break
        return count

    consecutive_same_action = _count_consecutive("recommended_next_action", last_action)
    consecutive_same_challenger = _count_consecutive("recommended_challenger", last_challenger)
    consecutive_keep_production = _count_consecutive("keep_production_as_is", True)
    consecutive_run_more_seeds = _count_consecutive("recommended_next_action", "run_more_seeds")

    # Per-variant pairwise positive signal streaks.
    # "Positive signal for settransformer" = whether_to_try_pma_or_isab_next is True
    # (triggered when settransformer_vs_deepsets both_pass_rate >= 0.5).
    consecutive_positive_signal_for_settransformer = 0
    for entry in reversed(history):
        if entry.get("whether_to_try_pma_or_isab_next", False):
            consecutive_positive_signal_for_settransformer += 1
        else:
            break

    # "Positive signal for deepsets" = deepsets_vs_legacy both_pass_count > 0
    consecutive_positive_signal_for_deepsets = 0
    for entry in reversed(history):
        pw = entry.get("key_pairwise_signals") or {}
        ds_vs_legacy = pw.get("deepsets_vs_legacy") or {}
        if int(ds_vs_legacy.get("both_pass_count") or 0) > 0:
            consecutive_positive_signal_for_deepsets += 1
        else:
            break

    return {
        "total_campaigns": len(history),
        "latest_action": last_action,
        "latest_challenger": last_challenger,
        "consecutive_same_action": consecutive_same_action,
        "consecutive_same_challenger": consecutive_same_challenger,
        "consecutive_keep_production": consecutive_keep_production,
        "consecutive_run_more_seeds": consecutive_run_more_seeds,
        "consecutive_positive_signal_for_settransformer": consecutive_positive_signal_for_settransformer,
        "consecutive_positive_signal_for_deepsets": consecutive_positive_signal_for_deepsets,
    }


# ---------------------------------------------------------------------------
# Campaign history I/O
# ---------------------------------------------------------------------------


def append_campaign_to_history(
    history: list[dict[str, Any]],
    entry: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return a new history list with the entry appended.

    If a campaign with the same campaign_name already exists, it is replaced.
    History is ordered chronologically (oldest first).
    """
    new_history = [e for e in history if e.get("campaign_name") != entry.get("campaign_name")]
    new_history.append(entry)
    return new_history


def load_campaign_history(data_dir: str | Path = "data") -> list[dict[str, Any]]:
    """Load existing campaign history from data_dir, or return an empty list."""
    history_path = Path(data_dir) / "campaign_history.json"
    if not history_path.exists():
        return []
    with open(history_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return list(payload.get("campaigns") or [])


def save_campaign_history(
    history: list[dict[str, Any]],
    stability: dict[str, Any],
    data_dir: str | Path = "data",
) -> Path:
    """Save campaign history + stability to data_dir/campaign_history.json."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "campaign_history.json"
    payload = {
        "schema_version": CAMPAIGN_HISTORY_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_campaigns": len(history),
        "recommendation_stability": stability,
        "campaigns": history,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return path


def build_campaign_history_csv(history: list[dict[str, Any]]) -> str:
    """Build a tabular CSV summary of campaign history."""
    fieldnames = [
        "campaign_name",
        "profile_name",
        "generated_at",
        "loto_types",
        "preset",
        "seeds",
        "recommended_next_action",
        "recommended_challenger",
        "keep_production_as_is",
        "best_variant_by_logloss",
        "consistent_promote_variants",
        "blockers_count",
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for entry in history:
        writer.writerow({
            "campaign_name": entry.get("campaign_name") or "",
            "profile_name": entry.get("profile_name") or "",
            "generated_at": entry.get("generated_at") or "",
            "loto_types": "|".join(entry.get("loto_types") or []),
            "preset": entry.get("preset") or "",
            "seeds": "|".join(str(s) for s in (entry.get("seeds") or [])),
            "recommended_next_action": entry.get("recommended_next_action") or "",
            "recommended_challenger": entry.get("recommended_challenger") or "",
            "keep_production_as_is": "true" if entry.get("keep_production_as_is") else "false",
            "best_variant_by_logloss": entry.get("best_variant_by_logloss") or "",
            "consistent_promote_variants": "|".join(entry.get("consistent_promote_variants") or []),
            "blockers_count": entry.get("blockers_count") or 0,
        })
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Diff report
# ---------------------------------------------------------------------------


def _fmt(value: float | None, decimals: int = 4) -> str:
    return f"{value:.{decimals}f}" if value is not None else "N/A"


def _diff_str(old: float | None, new: float | None, decimals: int = 4) -> str:
    """Return diff string: '+0.0012', '-0.0034', or 'N/A'."""
    if old is None or new is None:
        return "N/A"
    diff = new - old
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:.{decimals}f}"


def _variant_ranking_map(ranking_summary: list[dict]) -> dict[str, dict]:
    """Build {variant: {rank, logloss_mean}} mapping from ranking_summary."""
    return {e["variant"]: e for e in ranking_summary if "variant" in e}


def build_diff_report(
    prev_entry: dict[str, Any],
    curr_entry: dict[str, Any],
    *,
    stability: dict[str, Any] | None = None,
) -> str:
    """Build a Markdown diff report comparing two consecutive campaign entries.

    This is the first document a human should read when reviewing a new campaign.
    It highlights what changed and whether signals are strengthening or weakening.
    """
    lines: list[str] = []

    prev_name = prev_entry.get("campaign_name", "?")
    curr_name = curr_entry.get("campaign_name", "?")

    lines.append("# Campaign Diff Report")
    lines.append("")
    lines.append(f"Comparing: **{prev_name}** → **{curr_name}**")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")

    # --- Campaign overview ---
    lines.append("## Campaign Overview")
    lines.append("")
    lines.append("| | Previous | Current |")
    lines.append("|-|----------|---------|")
    lines.append(f"| campaign_name | {prev_name} | {curr_name} |")
    lines.append(
        f"| profile_name | {prev_entry.get('profile_name', '?')} "
        f"| {curr_entry.get('profile_name', '?')} |"
    )
    lines.append(
        f"| generated_at | {prev_entry.get('generated_at', '?')} "
        f"| {curr_entry.get('generated_at', '?')} |"
    )
    lines.append(
        f"| preset | {prev_entry.get('preset', '?')} | {curr_entry.get('preset', '?')} |"
    )
    lines.append(
        f"| seeds | {prev_entry.get('seeds', '?')} | {curr_entry.get('seeds', '?')} |"
    )
    lines.append(
        f"| loto_types | {', '.join(prev_entry.get('loto_types') or [])} "
        f"| {', '.join(curr_entry.get('loto_types') or [])} |"
    )
    lines.append("")

    # --- Recommendation change ---
    lines.append("## Recommendation Change")
    lines.append("")

    prev_action = prev_entry.get("recommended_next_action", "?")
    curr_action = curr_entry.get("recommended_next_action", "?")
    action_changed = prev_action != curr_action
    action_marker = " ⚡ **CHANGED**" if action_changed else " ✓ (unchanged)"
    lines.append(
        f"**recommended_next_action**: `{prev_action}` → `{curr_action}`{action_marker}"
    )
    lines.append("")

    prev_challenger = prev_entry.get("recommended_challenger") or "None"
    curr_challenger = curr_entry.get("recommended_challenger") or "None"
    challenger_changed = prev_challenger != curr_challenger
    challenger_marker = " ⚡ **CHANGED**" if challenger_changed else " ✓ (unchanged)"
    lines.append(
        f"**recommended_challenger**: `{prev_challenger}` → `{curr_challenger}`{challenger_marker}"
    )
    lines.append("")

    prev_keep = "Yes" if prev_entry.get("keep_production_as_is") else "No"
    curr_keep = "Yes" if curr_entry.get("keep_production_as_is") else "No"
    keep_changed = prev_keep != curr_keep
    keep_marker = " ⚡ **CHANGED**" if keep_changed else " ✓ (unchanged)"
    lines.append(f"**keep_production_as_is**: {prev_keep} → {curr_keep}{keep_marker}")
    lines.append("")

    prev_blockers = int(prev_entry.get("blockers_count") or 0)
    curr_blockers = int(curr_entry.get("blockers_count") or 0)
    blocker_diff = curr_blockers - prev_blockers
    blocker_marker = f" ({'+' if blocker_diff >= 0 else ''}{blocker_diff})"
    lines.append(f"**blockers_count**: {prev_blockers} → {curr_blockers}{blocker_marker}")
    lines.append("")

    # --- Variant ranking change ---
    lines.append("## Variant Ranking Change")
    lines.append("")
    lines.append("*(by logloss, lower is better)*")
    lines.append("")
    lines.append(
        "| variant | prev_rank | curr_rank | rank_Δ | prev_logloss | curr_logloss | logloss_Δ |"
    )
    lines.append(
        "|---------|-----------|-----------|--------|--------------|--------------|-----------|"
    )

    prev_ranking = _variant_ranking_map(prev_entry.get("variant_ranking_summary") or [])
    curr_ranking = _variant_ranking_map(curr_entry.get("variant_ranking_summary") or [])
    all_variants = sorted(set(list(prev_ranking.keys()) + list(curr_ranking.keys())))

    for v in all_variants:
        prev_r = prev_ranking.get(v) or {}
        curr_r = curr_ranking.get(v) or {}
        prev_rank = prev_r.get("rank") or "—"
        curr_rank = curr_r.get("rank") or "—"
        prev_ll = prev_r.get("logloss_mean")
        curr_ll = curr_r.get("logloss_mean")

        if isinstance(prev_rank, int) and isinstance(curr_rank, int):
            rank_diff = curr_rank - prev_rank
            rank_diff_str = f"{'+' if rank_diff >= 0 else ''}{rank_diff}"
        else:
            rank_diff_str = "N/A"

        lines.append(
            f"| **{v}** | {prev_rank} | {curr_rank} | {rank_diff_str} | "
            f"{_fmt(prev_ll)} | {_fmt(curr_ll)} | {_diff_str(prev_ll, curr_ll)} |"
        )
    lines.append("")

    # --- Pairwise signal change ---
    lines.append("## Pairwise Signal Change")
    lines.append("")
    lines.append("*(both_pass_count = runs where both CI and permutation test pass)*")
    lines.append("")
    lines.append(
        "| comparison | prev both_pass/run | curr both_pass/run "
        "| prev rate | curr rate | rate_Δ |"
    )
    lines.append(
        "|------------|-------------------|--------------------"
        "|-----------|-----------|--------|"
    )

    prev_pw = prev_entry.get("key_pairwise_signals") or {}
    curr_pw = curr_entry.get("key_pairwise_signals") or {}
    all_keys = sorted(set(list(prev_pw.keys()) + list(curr_pw.keys())))

    for key in all_keys:
        pp = prev_pw.get(key) or {}
        cp = curr_pw.get(key) or {}
        prev_bp = int(pp.get("both_pass_count") or 0)
        prev_rc = int(pp.get("run_count") or 0)
        curr_bp = int(cp.get("both_pass_count") or 0)
        curr_rc = int(cp.get("run_count") or 0)
        prev_rate = pp.get("both_pass_rate")
        curr_rate = cp.get("both_pass_rate")
        lines.append(
            f"| {key} | {prev_bp}/{prev_rc} | {curr_bp}/{curr_rc} | "
            f"{_fmt(prev_rate, 3) if prev_rate is not None else 'N/A'} | "
            f"{_fmt(curr_rate, 3) if curr_rate is not None else 'N/A'} | "
            f"{_diff_str(prev_rate, curr_rate, 3)} |"
        )
    lines.append("")

    # --- Recommendation stability ---
    if stability:
        lines.append("## Recommendation Stability")
        lines.append("")
        lines.append(f"- **Total campaigns**: {stability.get('total_campaigns', 0)}")
        lines.append(f"- **Latest action**: `{stability.get('latest_action') or 'N/A'}`")
        lines.append(
            f"- **Consecutive same action**: {stability.get('consecutive_same_action', 0)}"
        )
        lines.append(
            f"- **Consecutive same challenger**: {stability.get('consecutive_same_challenger', 0)}"
        )
        lines.append(
            f"- **Consecutive keep_production**: {stability.get('consecutive_keep_production', 0)}"
        )
        lines.append(
            f"- **Consecutive run_more_seeds**: {stability.get('consecutive_run_more_seeds', 0)}"
        )
        lines.append("")

        # Stability guidance
        consec = stability.get("consecutive_same_action", 0)
        action = stability.get("latest_action") or ""
        if action == "run_more_seeds" and consec >= 3:
            lines.append(
                f"> **Stability signal**: `run_more_seeds` has appeared in "
                f"{consec} consecutive campaigns.  "
            )
            lines.append(
                "> Consider running the `archcomp_full` profile to increase seed count."
            )
        elif action == "hold" and consec >= 3:
            lines.append(
                f"> **Stability signal**: `hold` has appeared in {consec} consecutive campaigns.  "
            )
            lines.append("> No consistent winner found — architecture differentiation may be limited.")
        elif action == "consider_promotion" and consec >= 2:
            lines.append(
                f"> **Stability signal**: `consider_promotion` has appeared in "
                f"{consec} consecutive campaigns.  "
            )
            lines.append(
                "> Strong evidence for promotion — review per-loto details before proceeding."
            )
        lines.append("")

    # --- PMA/ISAB guidance ---
    lines.append("## PMA / ISAB Next Steps")
    lines.append("")
    prev_pma = prev_entry.get("whether_to_try_pma_or_isab_next", False)
    curr_pma = curr_entry.get("whether_to_try_pma_or_isab_next", False)
    if curr_pma and not prev_pma:
        lines.append(
            "**PMA/ISAB signal newly appeared** ⭐ — "
            "`settransformer_vs_deepsets` both_pass_rate ≥ 0.5."
        )
        lines.append("Consider exploring PMA / ISAB pooling in the next iteration.")
    elif curr_pma and prev_pma:
        lines.append(
            "**PMA/ISAB signal persists** — consistent attention benefit detected across campaigns."
        )
        lines.append("PMA / ISAB exploration is now well-motivated.")
    elif not curr_pma and prev_pma:
        lines.append(
            "**PMA/ISAB signal disappeared** — "
            "`settransformer_vs_deepsets` advantage is no longer consistent."
        )
        lines.append("Wait for another campaign before proceeding with PMA / ISAB.")
    else:
        lines.append(
            "**PMA/ISAB signal: not yet triggered** — "
            "settransformer has not shown consistent advantage over deepsets."
        )
    lines.append("")

    lines.append("---")
    lines.append(
        f"*Diff report generated by campaign_manager.py v{CAMPAIGN_DIFF_REPORT_SCHEMA_VERSION}*"
    )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Save all campaign artifacts
# ---------------------------------------------------------------------------


def save_campaign_artifacts(
    history: list[dict[str, Any]],
    data_dir: str | Path = "data",
) -> dict[str, str]:
    """Save campaign history JSON + CSV and a diff report (if ≥ 2 campaigns).

    Returns dict of {artifact_name: absolute_path_str}.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    stability = compute_recommendation_stability(history)
    paths: dict[str, str] = {}

    # History JSON
    history_path = save_campaign_history(history, stability, data_dir)
    paths["campaign_history.json"] = str(history_path)

    # History CSV
    csv_content = build_campaign_history_csv(history)
    csv_path = data_dir / "campaign_history.csv"
    csv_path.write_text(csv_content, encoding="utf-8")
    paths["campaign_history.csv"] = str(csv_path)

    # Diff report (requires at least 2 campaigns)
    if len(history) >= 2:
        diff_md = build_diff_report(history[-2], history[-1], stability=stability)
        diff_path = data_dir / "campaign_diff_report.md"
        diff_path.write_text(diff_md, encoding="utf-8")
        paths["campaign_diff_report.md"] = str(diff_path)

    return paths
