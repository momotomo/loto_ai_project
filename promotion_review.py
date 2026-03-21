"""promotion_review.py

Promotion review artifacts for the loto-ai campaign infrastructure.

This module builds three artifacts from accepted campaign history only
(accepted_for_decision_use=True, counts_toward_promotion_readiness=True):

1. accepted_campaign_summary  — accepted-only view of campaign history
   data/accepted_campaign_summary.json
   data/accepted_campaign_summary.md

2. promotion_review_readiness — "can we enter promotion review now?"
   data/promotion_review_readiness.json
   data/promotion_review_readiness.md

3. accepted_campaign_review_bundle — combined view for human review
   data/accepted_campaign_review_bundle.json
   data/accepted_campaign_review_bundle.md

Key distinction
---------------
  Comparable campaign  : same experimental conditions (trend analysis OK)
  Accepted campaign    : additionally satisfies the decision benchmark policy
                         (counts toward promotion readiness)
  Review-ready         : enough accepted evidence to enter promotion review

"Ready for promotion review" ≠ "promote to production".
Production is NEVER changed automatically.

Schema versions
---------------
  ACCEPTED_CAMPAIGN_SUMMARY_SCHEMA_VERSION   = 1
  PROMOTION_REVIEW_READINESS_SCHEMA_VERSION  = 1
  ACCEPTED_REVIEW_BUNDLE_SCHEMA_VERSION      = 1
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ACCEPTED_CAMPAIGN_SUMMARY_SCHEMA_VERSION = 1
PROMOTION_REVIEW_READINESS_SCHEMA_VERSION = 1
ACCEPTED_REVIEW_BUNDLE_SCHEMA_VERSION = 1

# Minimum consecutive accepted positive signals to be review-ready
_MIN_CONSECUTIVE_ACCEPTED_POSITIVE = 2
# Minimum total accepted campaigns to be review-ready
_MIN_ACCEPTED_CAMPAIGNS = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _accepted_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return only entries where accepted_for_decision_use=True."""
    return [e for e in history if e.get("accepted_for_decision_use", False)]


def _counts_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return only entries where counts_toward_promotion_readiness=True."""
    return [e for e in history if e.get("counts_toward_promotion_readiness", False)]


def _consecutive_positive_signal(accepted: list[dict[str, Any]]) -> int:
    """Count consecutive accepted campaigns (most-recent first) with positive action.

    Positive = recommended_next_action in {consider_promotion, run_more_seeds}.
    """
    positive_actions = {"consider_promotion", "run_more_seeds"}
    count = 0
    for entry in reversed(accepted):
        if entry.get("recommended_next_action") in positive_actions:
            count += 1
        else:
            break
    return count


def _consecutive_settransformer_signal(accepted: list[dict[str, Any]]) -> int:
    """Count consecutive accepted campaigns with whether_to_try_pma_or_isab_next=True."""
    count = 0
    for entry in reversed(accepted):
        if entry.get("whether_to_try_pma_or_isab_next", False):
            count += 1
        else:
            break
    return count


def _action_distribution(entries: list[dict[str, Any]]) -> dict[str, int]:
    dist: dict[str, int] = {}
    for e in entries:
        a = e.get("recommended_next_action") or "unknown"
        dist[a] = dist.get(a, 0) + 1
    return dist


def _challenger_distribution(entries: list[dict[str, Any]]) -> dict[str, int]:
    dist: dict[str, int] = {}
    for e in entries:
        c = e.get("recommended_challenger") or "none"
        dist[c] = dist.get(c, 0) + 1
    return dist


# ---------------------------------------------------------------------------
# 1. Accepted Campaign Summary
# ---------------------------------------------------------------------------


def build_accepted_campaign_summary(
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a summary of accepted campaigns only.

    This is the accepted-only view of campaign history — separated from
    the raw campaign history so that promotion readiness decisions are
    based exclusively on policy-compliant campaigns.

    Returns a JSON-serialisable dict.
    """
    accepted = _accepted_history(history)
    counts = _counts_history(history)

    accepted_names = [e.get("campaign_name", "?") for e in accepted]
    counts_names = [e.get("campaign_name", "?") for e in counts]

    action_dist = _action_distribution(accepted)
    challenger_dist = _challenger_distribution(accepted)

    consec_positive = _consecutive_positive_signal(accepted)
    consec_st = _consecutive_settransformer_signal(accepted)

    # Accepted-only pairwise signal streaks
    consec_st_streak = 0
    for e in reversed(accepted):
        if e.get("whether_to_try_pma_or_isab_next", False):
            consec_st_streak += 1
        else:
            break

    # Latest accepted entry
    latest_accepted = accepted[-1] if accepted else None

    # Profile/benchmark summary across accepted campaigns
    profile_counts: dict[str, int] = {}
    benchmark_counts: dict[str, int] = {}
    loto_type_sets: list[list[str]] = []
    seed_counts: list[int] = []
    variant_sets: list[set[str]] = []

    for e in accepted:
        p = e.get("profile_name") or "unknown"
        profile_counts[p] = profile_counts.get(p, 0) + 1
        b = e.get("decision_benchmark_name") or e.get("benchmark_name") or "unknown"
        benchmark_counts[b] = benchmark_counts.get(b, 0) + 1
        loto_type_sets.append(e.get("loto_types") or [])
        seed_counts.append(len(e.get("seeds") or []))
        emv = e.get("evaluation_model_variants") or ""
        if emv:
            variant_sets.append({v.strip() for v in emv.split(",") if v.strip()})

    # Action/challenger history (most recent last)
    action_history = [
        {
            "campaign_name": e.get("campaign_name", "?"),
            "action": e.get("recommended_next_action"),
            "challenger": e.get("recommended_challenger"),
            "keep_production": e.get("keep_production_as_is", True),
            "pma_signal": e.get("whether_to_try_pma_or_isab_next", False),
        }
        for e in accepted
    ]

    # Comparability summary (accepted pairs)
    comparability_issues: list[str] = []
    for i in range(1, len(accepted)):
        prev = accepted[i - 1]
        curr = accepted[i]
        # Simple check: same profile + same loto coverage
        if set(prev.get("loto_types") or []) != set(curr.get("loto_types") or []):
            comparability_issues.append(
                f"loto_types differ between {prev.get('campaign_name','?')} "
                f"and {curr.get('campaign_name','?')}"
            )

    return {
        "schema_version": ACCEPTED_CAMPAIGN_SUMMARY_SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "total_campaigns_in_history": len(history),
        "accepted_campaign_count": len(accepted),
        "counts_toward_promotion_readiness_count": len(counts),
        "accepted_campaign_names": accepted_names,
        "counts_toward_promotion_readiness_names": counts_names,
        "action_history": action_history,
        "action_distribution": action_dist,
        "challenger_distribution": challenger_dist,
        "consecutive_accepted_positive_signals": consec_positive,
        "consecutive_accepted_settransformer_signal": consec_st_streak,
        "profile_distribution": profile_counts,
        "benchmark_distribution": benchmark_counts,
        "seed_counts_per_accepted_campaign": seed_counts,
        "loto_types_per_accepted_campaign": loto_type_sets,
        "comparability_issues_within_accepted": comparability_issues,
        "latest_accepted_campaign_name": (
            latest_accepted.get("campaign_name") if latest_accepted else None
        ),
        "latest_accepted_action": (
            latest_accepted.get("recommended_next_action") if latest_accepted else None
        ),
        "latest_accepted_challenger": (
            latest_accepted.get("recommended_challenger") if latest_accepted else None
        ),
        "latest_accepted_pma_signal": (
            latest_accepted.get("whether_to_try_pma_or_isab_next", False)
            if latest_accepted else False
        ),
    }


def build_accepted_campaign_summary_md(summary: dict[str, Any]) -> str:
    """Build a Markdown accepted campaign summary."""
    lines: list[str] = []
    lines.append("# Accepted Campaign Summary (Decision-Evidence Only)")
    lines.append("")
    lines.append(f"Generated: {summary.get('generated_at', _now_iso())}")
    lines.append("")
    lines.append(
        "> **This document shows only campaigns accepted by the Decision Benchmark Policy.**  "
    )
    lines.append(
        "> Non-accepted campaigns (archcomp_lite, non-comparable, etc.) are excluded.  "
    )
    lines.append(
        "> Use this as the basis for promotion readiness decisions.  "
    )
    lines.append(
        "> See `accepted_campaign_review_bundle.md` for the full review view."
    )
    lines.append("")

    total = summary.get("total_campaigns_in_history", 0)
    n_accepted = summary.get("accepted_campaign_count", 0)
    n_counts = summary.get("counts_toward_promotion_readiness_count", 0)

    lines.append("## Counts")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total campaigns in history | {total} |")
    lines.append(f"| Accepted for decision use | **{n_accepted}** |")
    lines.append(f"| Counting toward promotion readiness | **{n_counts}** |")
    lines.append(f"| Consecutive accepted positive signals | {summary.get('consecutive_accepted_positive_signals', 0)} |")
    lines.append(f"| Consecutive accepted settransformer signal | {summary.get('consecutive_accepted_settransformer_signal', 0)} |")
    lines.append("")

    accepted_names = summary.get("accepted_campaign_names") or []
    if accepted_names:
        lines.append("## Accepted Campaigns")
        lines.append("")
        for name in accepted_names:
            lines.append(f"- `{name}`")
        lines.append("")
    else:
        lines.append("## Accepted Campaigns")
        lines.append("")
        lines.append("*No accepted campaigns yet.*")
        lines.append("")

    # Action/challenger history table
    action_history = summary.get("action_history") or []
    if action_history:
        lines.append("## Action / Challenger History (Accepted Only)")
        lines.append("")
        lines.append("| Campaign | Action | Challenger | PMA signal | keep_production |")
        lines.append("|----------|--------|------------|------------|-----------------|")
        for r in action_history:
            keep = "Yes" if r.get("keep_production") else "No"
            pma = "⭐" if r.get("pma_signal") else "—"
            lines.append(
                f"| `{r.get('campaign_name','?')}` "
                f"| `{r.get('action') or '?'}` "
                f"| {r.get('challenger') or '—'} "
                f"| {pma} | {keep} |"
            )
        lines.append("")

    # Action distribution
    action_dist = summary.get("action_distribution") or {}
    if action_dist:
        lines.append("## Action Distribution (Accepted Only)")
        lines.append("")
        lines.append("| Action | Count |")
        lines.append("|--------|-------|")
        for action, count in sorted(action_dist.items(), key=lambda x: -x[1]):
            lines.append(f"| `{action}` | {count} |")
        lines.append("")

    # Challenger distribution
    challenger_dist = summary.get("challenger_distribution") or {}
    if challenger_dist:
        lines.append("## Challenger Distribution (Accepted Only)")
        lines.append("")
        lines.append("| Challenger | Count |")
        lines.append("|------------|-------|")
        for c, count in sorted(challenger_dist.items(), key=lambda x: -x[1]):
            lines.append(f"| `{c}` | {count} |")
        lines.append("")

    # Comparability
    issues = summary.get("comparability_issues_within_accepted") or []
    if issues:
        lines.append("## Comparability Issues (Within Accepted Campaigns)")
        lines.append("")
        for issue in issues:
            lines.append(f"- ⚠️ {issue}")
        lines.append("")
    else:
        lines.append("## Comparability (Within Accepted Campaigns)")
        lines.append("")
        if n_accepted >= 2:
            lines.append("✅ No loto_types differences detected between consecutive accepted campaigns.")
        else:
            lines.append("*N/A — fewer than 2 accepted campaigns.*")
        lines.append("")

    lines.append("---")
    lines.append(
        f"*Accepted campaign summary generated by promotion_review.py "
        f"v{ACCEPTED_CAMPAIGN_SUMMARY_SCHEMA_VERSION}*"
    )
    lines.append("")
    return "\n".join(lines)


def save_accepted_campaign_summary_artifacts(
    history: list[dict[str, Any]],
    data_dir: str | Path = "data",
) -> dict[str, str]:
    """Save accepted_campaign_summary.json and .md to data_dir."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    summary = build_accepted_campaign_summary(history)
    paths: dict[str, str] = {}

    json_path = data_dir / "accepted_campaign_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")
    paths["accepted_campaign_summary.json"] = str(json_path)

    md_path = data_dir / "accepted_campaign_summary.md"
    md_path.write_text(build_accepted_campaign_summary_md(summary), encoding="utf-8")
    paths["accepted_campaign_summary.md"] = str(md_path)

    return paths


# ---------------------------------------------------------------------------
# 2. Promotion Review Readiness
# ---------------------------------------------------------------------------


def build_promotion_review_readiness(
    history: list[dict[str, Any]],
    stability: dict[str, Any],
    promotion_gate: dict[str, Any] | None = None,
    regression_alert: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Determine if we can enter promotion review based on accepted campaigns only.

    This is distinct from the promotion_gate in governance_layer:
      - promotion_gate uses ALL campaigns in the window
      - promotion_review_readiness uses ONLY accepted campaigns

    "Ready for promotion review" means:
      (1) ≥2 consecutive accepted campaigns with positive signal
      (2) promotion gate status is green (optional but checked)
      (3) no active HIGH regression alert
      (4) total accepted campaigns ≥ 2

    Returns a JSON-serialisable dict.
    """
    accepted = _accepted_history(history)
    counts = _counts_history(history)

    n_accepted = len(accepted)
    n_counts = len(counts)

    blockers: list[str] = []
    conditions_passed: list[str] = []

    # Condition 1: minimum accepted campaigns
    if n_accepted >= _MIN_ACCEPTED_CAMPAIGNS:
        conditions_passed.append(
            f"accepted_campaign_count: {n_accepted} accepted campaign(s) "
            f"(≥{_MIN_ACCEPTED_CAMPAIGNS} required)"
        )
    else:
        blockers.append(
            f"Insufficient accepted campaigns: {n_accepted} "
            f"(need ≥{_MIN_ACCEPTED_CAMPAIGNS}). "
            "Run archcomp or archcomp_full campaigns that satisfy the decision benchmark policy."
        )

    # Condition 2: consecutive positive signal from accepted campaigns
    consec_positive = _consecutive_positive_signal(accepted)
    positive_actions = {"consider_promotion", "run_more_seeds"}
    latest_accepted_action = (
        accepted[-1].get("recommended_next_action") if accepted else None
    )
    if consec_positive >= _MIN_CONSECUTIVE_ACCEPTED_POSITIVE:
        conditions_passed.append(
            f"consecutive_accepted_positive_signals: {consec_positive} consecutive "
            f"accepted campaigns with positive action `{latest_accepted_action}`"
        )
    else:
        if not accepted:
            blockers.append(
                "No accepted campaigns yet — cannot evaluate positive signal streak."
            )
        elif latest_accepted_action not in positive_actions:
            blockers.append(
                f"Latest accepted action `{latest_accepted_action}` is not a positive signal. "
                f"Need ≥{_MIN_CONSECUTIVE_ACCEPTED_POSITIVE} consecutive accepted campaigns "
                "with `consider_promotion` or `run_more_seeds`."
            )
        else:
            blockers.append(
                f"Consecutive accepted positive signal too short: {consec_positive} "
                f"(need ≥{_MIN_CONSECUTIVE_ACCEPTED_POSITIVE}). "
                "Run another accepted campaign with the same positive action."
            )

    # Condition 3: no HIGH regression alert
    alert_level = (regression_alert or {}).get("alert_level", "none")
    if alert_level != "high":
        conditions_passed.append(
            f"no_high_regression: regression alert level is '{alert_level}'"
        )
    else:
        blockers.append(
            "Active HIGH regression alert — resolve before entering promotion review."
        )

    # Condition 4: promotion gate status (informational — gate uses all campaigns)
    gate_status = (promotion_gate or {}).get("gate_status", "unknown")
    if gate_status == "green":
        conditions_passed.append(
            "promotion_gate_green: promotion gate is GREEN "
            "(all gate conditions met including non-accepted campaigns)"
        )
    elif gate_status == "yellow":
        conditions_passed.append(
            "promotion_gate_yellow: promotion gate is YELLOW "
            "(some gate conditions met)"
        )
    else:
        if promotion_gate is not None:
            blockers.append(
                f"promotion_gate_{gate_status}: promotion gate is {gate_status.upper()}. "
                "Resolve gate blockers before promotion review. "
                "See `promotion_gate.md` for details."
            )

    # Condition 5: consistent consistent_promote_variants in latest accepted campaign
    if accepted:
        latest_accepted = accepted[-1]
        consistent_promote = latest_accepted.get("consistent_promote_variants") or []
        candidate = latest_accepted.get("recommended_challenger")
        if consistent_promote:
            conditions_passed.append(
                f"consistent_promote_in_latest_accepted: "
                f"{', '.join(consistent_promote)} passed guardrails"
            )
        else:
            blockers.append(
                "No variant passed promotion guardrails in the latest accepted campaign "
                "(consistent_promote_variants is empty)."
            )
    else:
        candidate = None
        latest_accepted = None

    # Determine readiness
    ready = len(blockers) == 0

    # Candidate variant
    if not candidate and accepted:
        # Fall back to latest accepted challenger
        for e in reversed(accepted):
            c = e.get("recommended_challenger")
            if c:
                candidate = c
                break

    # Accepted campaign window info
    accepted_names = [e.get("campaign_name", "?") for e in accepted]

    # Build rationale
    if ready:
        rationale = (
            f"All {len(conditions_passed)} readiness condition(s) met using "
            f"{n_accepted} accepted campaign(s). "
            f"Candidate variant: `{candidate or 'TBD'}`. "
            "You may enter promotion review — see the full review bundle "
            "in `accepted_campaign_review_bundle.md` before proceeding."
        )
        recommended_next_step = (
            "1. Read `accepted_campaign_review_bundle.md` in full.\n"
            "2. Confirm the signal is genuine per-loto "
            "(see `campaigns/<name>/cross_loto_report.md` for each accepted campaign).\n"
            "3. Verify no active data drift or external changes since the last campaign.\n"
            "4. If satisfied, begin the production training review process (manual step).\n"
            "⚠️ Production is NEVER changed automatically — this is a human decision."
        )
    else:
        rationale = (
            f"{len(blockers)} blocker(s) prevent entering promotion review. "
            f"Accepted campaigns: {n_accepted}. "
            "Run more accepted campaigns (archcomp or archcomp_full) to accumulate evidence."
        )
        recommended_next_step = (
            "Resolve the blockers listed above, then re-run an archcomp or archcomp_full campaign. "
            "Do NOT enter promotion review until all blockers are cleared."
        )

    # Accepted-only stability from the stability dict
    consec_action_acc = stability.get("consecutive_same_action_accepted_only", 0)
    consec_st_acc = stability.get(
        "consecutive_positive_signal_for_settransformer_accepted_only", 0
    )

    return {
        "schema_version": PROMOTION_REVIEW_READINESS_SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "ready_for_promotion_review": ready,
        "candidate_variant": candidate,
        "accepted_campaign_count": n_accepted,
        "counts_toward_promotion_readiness_count": n_counts,
        "accepted_campaign_window": accepted_names,
        "consecutive_accepted_positive_signals": consec_positive,
        "consecutive_accepted_settransformer_signal": _consecutive_settransformer_signal(accepted),
        "consecutive_same_action_accepted_only": consec_action_acc,
        "consecutive_positive_signal_for_settransformer_accepted_only": consec_st_acc,
        "promotion_gate_status": gate_status,
        "regression_alert_level": alert_level,
        "conditions_passed": conditions_passed,
        "blockers": blockers,
        "rationale": rationale,
        "recommended_next_step": recommended_next_step,
    }


def build_promotion_review_readiness_md(readiness: dict[str, Any]) -> str:
    """Build a Markdown promotion review readiness document."""
    ready = readiness.get("ready_for_promotion_review", False)
    ready_emoji = "✅" if ready else "❌"
    ready_label = "READY FOR PROMOTION REVIEW" if ready else "NOT YET READY"

    lines: list[str] = []
    lines.append("# Promotion Review Readiness")
    lines.append("")
    lines.append(f"Generated: {readiness.get('generated_at', _now_iso())}")
    lines.append("")
    lines.append(f"## Status: {ready_emoji} {ready_label}")
    lines.append("")
    lines.append(
        "> **This determines whether to ENTER promotion review — not whether to promote.**  "
    )
    lines.append(
        "> Production is NEVER changed automatically. This is a human decision gate only.  "
    )
    lines.append(
        "> Based exclusively on accepted campaigns (decision benchmark policy compliant)."
    )
    lines.append("")

    lines.append(readiness.get("rationale", ""))
    lines.append("")

    n_accepted = readiness.get("accepted_campaign_count", 0)
    n_counts = readiness.get("counts_toward_promotion_readiness_count", 0)
    candidate = readiness.get("candidate_variant") or "TBD"
    consec_pos = readiness.get("consecutive_accepted_positive_signals", 0)
    consec_st = readiness.get("consecutive_accepted_settransformer_signal", 0)
    gate_status = readiness.get("promotion_gate_status", "unknown")
    alert_level = readiness.get("regression_alert_level", "none")

    gate_emoji = {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(gate_status, "❓")
    alert_emoji = {"none": "✅", "low": "⚠️", "medium": "🔶", "high": "🔴"}.get(alert_level, "❓")

    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| accepted_campaign_count | **{n_accepted}** |")
    lines.append(f"| counts_toward_promotion_readiness | {n_counts} |")
    lines.append(f"| candidate_variant | `{candidate}` |")
    lines.append(f"| consecutive_accepted_positive_signals | {consec_pos} |")
    lines.append(f"| consecutive_accepted_settransformer_signal | {consec_st} |")
    lines.append(f"| promotion_gate_status | {gate_emoji} `{gate_status}` |")
    lines.append(f"| regression_alert_level | {alert_emoji} `{alert_level}` |")
    lines.append("")

    accepted_window = readiness.get("accepted_campaign_window") or []
    if accepted_window:
        lines.append("## Accepted Campaign Window")
        lines.append("")
        for name in accepted_window:
            lines.append(f"- `{name}`")
        lines.append("")

    conditions_passed = readiness.get("conditions_passed") or []
    if conditions_passed:
        lines.append("## Conditions Passed ✅")
        lines.append("")
        for c in conditions_passed:
            lines.append(f"- {c}")
        lines.append("")

    blockers = readiness.get("blockers") or []
    if blockers:
        lines.append("## Blockers ❌")
        lines.append("")
        for b in blockers:
            lines.append(f"- ❌ {b}")
        lines.append("")

    lines.append("## Recommended Next Step")
    lines.append("")
    for step_line in readiness.get("recommended_next_step", "").splitlines():
        lines.append(step_line)
    lines.append("")

    if ready:
        lines.append(
            "> ✅ **You may enter promotion review.** Read the full accepted campaign review "
            "bundle before any production decision. See `accepted_campaign_review_bundle.md`."
        )
    else:
        lines.append(
            "> ❌ **Do not enter promotion review yet.** Accumulate more accepted evidence "
            "by running archcomp or archcomp_full campaigns."
        )
    lines.append("")

    lines.append("---")
    lines.append(
        f"*Promotion review readiness generated by promotion_review.py "
        f"v{PROMOTION_REVIEW_READINESS_SCHEMA_VERSION}*"
    )
    lines.append("")
    return "\n".join(lines)


def save_promotion_review_readiness_artifacts(
    history: list[dict[str, Any]],
    stability: dict[str, Any],
    data_dir: str | Path = "data",
    promotion_gate: dict[str, Any] | None = None,
    regression_alert: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Save promotion_review_readiness.json and .md to data_dir."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    readiness = build_promotion_review_readiness(
        history,
        stability,
        promotion_gate=promotion_gate,
        regression_alert=regression_alert,
    )
    paths: dict[str, str] = {}

    json_path = data_dir / "promotion_review_readiness.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(readiness, f, indent=2, ensure_ascii=False)
        f.write("\n")
    paths["promotion_review_readiness.json"] = str(json_path)

    md_path = data_dir / "promotion_review_readiness.md"
    md_path.write_text(build_promotion_review_readiness_md(readiness), encoding="utf-8")
    paths["promotion_review_readiness.md"] = str(md_path)

    return paths


# ---------------------------------------------------------------------------
# 3. Accepted Campaign Review Bundle
# ---------------------------------------------------------------------------


def build_accepted_campaign_review_bundle(
    history: list[dict[str, Any]],
    stability: dict[str, Any],
    promotion_gate: dict[str, Any] | None = None,
    regression_alert: dict[str, Any] | None = None,
    comparability_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the full accepted campaign review bundle.

    This is the primary document for human reviewers — combining the
    accepted campaign summary, promotion review readiness, and key
    evidence into one place.

    Returns a JSON-serialisable dict.
    """
    accepted = _accepted_history(history)
    counts = _counts_history(history)

    summary = build_accepted_campaign_summary(history)
    readiness = build_promotion_review_readiness(
        history,
        stability,
        promotion_gate=promotion_gate,
        regression_alert=regression_alert,
    )

    ready = readiness.get("ready_for_promotion_review", False)
    gate_status = (promotion_gate or {}).get("gate_status", "unknown")
    alert_level = (regression_alert or {}).get("alert_level", "none")
    comp_ok = (comparability_result or {}).get("overall_comparable", None)
    comp_severity = (comparability_result or {}).get("overall_severity", "unknown")

    # Latest accepted evidence
    latest_accepted = accepted[-1] if accepted else None
    latest_accepted_evidence: dict[str, Any] = {}
    if latest_accepted:
        latest_accepted_evidence = {
            "campaign_name": latest_accepted.get("campaign_name"),
            "profile_name": latest_accepted.get("profile_name"),
            "benchmark_name": latest_accepted.get("decision_benchmark_name"),
            "generated_at": latest_accepted.get("generated_at"),
            "seeds": latest_accepted.get("seeds"),
            "loto_types": latest_accepted.get("loto_types"),
            "recommended_next_action": latest_accepted.get("recommended_next_action"),
            "recommended_challenger": latest_accepted.get("recommended_challenger"),
            "keep_production_as_is": latest_accepted.get("keep_production_as_is"),
            "best_variant_by_logloss": latest_accepted.get("best_variant_by_logloss"),
            "consistent_promote_variants": latest_accepted.get("consistent_promote_variants") or [],
            "blockers_count": latest_accepted.get("blockers_count", 0),
            "variant_ranking_summary": latest_accepted.get("variant_ranking_summary") or [],
            "whether_to_try_pma_or_isab_next": latest_accepted.get("whether_to_try_pma_or_isab_next", False),
        }

    # Why accepted evidence is/is not enough
    why_not_ready: list[str] = readiness.get("blockers") or []
    why_ready: list[str] = readiness.get("conditions_passed") or []

    return {
        "schema_version": ACCEPTED_REVIEW_BUNDLE_SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "ready_for_promotion_review": ready,
        "candidate_variant": readiness.get("candidate_variant"),
        "accepted_campaign_count": len(accepted),
        "counts_toward_promotion_readiness_count": len(counts),
        "accepted_campaign_names": [e.get("campaign_name", "?") for e in accepted],
        "counts_toward_promotion_readiness_names": [
            e.get("campaign_name", "?") for e in counts
        ],
        "consecutive_accepted_positive_signals": readiness.get(
            "consecutive_accepted_positive_signals", 0
        ),
        "consecutive_accepted_settransformer_signal": readiness.get(
            "consecutive_accepted_settransformer_signal", 0
        ),
        "promotion_gate_status": gate_status,
        "regression_alert_level": alert_level,
        "comparability_ok": comp_ok,
        "comparability_severity": comp_severity,
        "conditions_passed": why_ready,
        "blockers": why_not_ready,
        "rationale": readiness.get("rationale", ""),
        "recommended_next_step": readiness.get("recommended_next_step", ""),
        "action_history": summary.get("action_history") or [],
        "action_distribution": summary.get("action_distribution") or {},
        "challenger_distribution": summary.get("challenger_distribution") or {},
        "latest_accepted_evidence": latest_accepted_evidence,
        "accepted_campaign_summary": summary,
    }


def build_accepted_campaign_review_bundle_md(bundle: dict[str, Any]) -> str:
    """Build a Markdown accepted campaign review bundle for human reviewers."""
    ready = bundle.get("ready_for_promotion_review", False)
    ready_emoji = "✅" if ready else "❌"
    ready_label = "READY FOR PROMOTION REVIEW" if ready else "NOT YET READY"

    gate_status = bundle.get("promotion_gate_status", "unknown")
    alert_level = bundle.get("regression_alert_level", "none")
    gate_emoji = {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(gate_status, "❓")
    alert_emoji = {"none": "✅", "low": "⚠️", "medium": "🔶", "high": "🔴"}.get(alert_level, "❓")
    comp_ok = bundle.get("comparability_ok")
    comp_sev = bundle.get("comparability_severity", "unknown")
    if comp_ok is True:
        comp_emoji = {"ok": "✅", "warning": "⚠️", "error": "❌"}.get(comp_sev, "✅")
    elif comp_ok is False:
        comp_emoji = "❌"
    else:
        comp_emoji = "❓"

    lines: list[str] = []
    lines.append("# Accepted Campaign Review Bundle")
    lines.append("")
    lines.append(f"Generated: {bundle.get('generated_at', _now_iso())}")
    lines.append("")
    lines.append(
        "> **Start here for promotion review decisions.**  "
    )
    lines.append(
        "> This bundle aggregates all accepted campaign evidence in one place.  "
    )
    lines.append(
        "> Read in order: Overview → Accepted Evidence → Readiness → Next Steps.  "
    )
    lines.append(
        "> See `governance_report.md` for the full governance context."
    )
    lines.append("")

    # --- Quick overview ---
    lines.append("## Quick Overview")
    lines.append("")
    lines.append(f"| | Status |")
    lines.append(f"|--|--------|")
    lines.append(f"| **Promotion review readiness** | {ready_emoji} **{ready_label}** |")
    lines.append(f"| Promotion gate (all campaigns) | {gate_emoji} {gate_status.upper()} |")
    lines.append(f"| Regression alert | {alert_emoji} {alert_level.upper()} |")
    lines.append(f"| Comparability | {comp_emoji} {comp_sev.upper()} |")
    lines.append(f"| Accepted campaigns | **{bundle.get('accepted_campaign_count', 0)}** |")
    lines.append(
        f"| Consecutive accepted positive signals | "
        f"{bundle.get('consecutive_accepted_positive_signals', 0)} |"
    )
    lines.append(f"| Candidate variant | `{bundle.get('candidate_variant') or 'TBD'}` |")
    lines.append("")

    lines.append(bundle.get("rationale", ""))
    lines.append("")

    # --- Readiness conditions ---
    conditions_passed = bundle.get("conditions_passed") or []
    blockers = bundle.get("blockers") or []

    if conditions_passed:
        lines.append("## Readiness Conditions Passed ✅")
        lines.append("")
        for c in conditions_passed:
            lines.append(f"- {c}")
        lines.append("")

    if blockers:
        lines.append("## Blockers Preventing Promotion Review ❌")
        lines.append("")
        lines.append(
            "> Resolve all blockers before entering promotion review."
        )
        lines.append("")
        for b in blockers:
            lines.append(f"- ❌ {b}")
        lines.append("")

    # --- Accepted campaigns list ---
    accepted_names = bundle.get("accepted_campaign_names") or []
    counts_names = bundle.get("counts_toward_promotion_readiness_names") or []

    lines.append("## Accepted Campaigns (Decision Evidence)")
    lines.append("")
    if accepted_names:
        lines.append(
            "The following campaigns satisfy the Decision Benchmark Policy "
            "and may be used as promotion decision evidence:"
        )
        lines.append("")
        for name in accepted_names:
            marker = "✅" if name in counts_names else "⚠️"
            lines.append(f"- {marker} `{name}`")
        lines.append("")
        if len(accepted_names) != len(counts_names):
            lines.append(
                "> ⚠️ Some accepted campaigns are not counting toward promotion readiness. "
                "Check `accepted_campaign_summary.md` for details."
            )
            lines.append("")
    else:
        lines.append("*No accepted campaigns in history yet.*")
        lines.append("")

    # --- Action/challenger history ---
    action_history = bundle.get("action_history") or []
    if action_history:
        lines.append("## Action / Challenger History (Accepted Only)")
        lines.append("")
        lines.append("| Campaign | Action | Challenger | PMA signal | keep_production |")
        lines.append("|----------|--------|------------|------------|-----------------|")
        for r in action_history:
            keep = "Yes" if r.get("keep_production") else "No"
            pma = "⭐" if r.get("pma_signal") else "—"
            lines.append(
                f"| `{r.get('campaign_name','?')}` "
                f"| `{r.get('action') or '?'}` "
                f"| {r.get('challenger') or '—'} "
                f"| {pma} | {keep} |"
            )
        lines.append("")

    # --- Latest accepted evidence ---
    evidence = bundle.get("latest_accepted_evidence") or {}
    if evidence:
        lines.append("## Latest Accepted Evidence")
        lines.append("")
        lines.append(f"Campaign: **{evidence.get('campaign_name', '?')}**")
        lines.append(f"Profile: `{evidence.get('profile_name', '?')}` | "
                     f"Benchmark: `{evidence.get('benchmark_name', '?')}`")
        lines.append(f"Seeds: `{evidence.get('seeds', [])}` | "
                     f"Loto types: `{evidence.get('loto_types', [])}`")
        lines.append("")
        lines.append("| Field | Value |")
        lines.append("|-------|-------|")
        lines.append(
            f"| recommended_next_action | `{evidence.get('recommended_next_action', '?')}` |"
        )
        lines.append(
            f"| recommended_challenger | `{evidence.get('recommended_challenger') or 'none'}` |"
        )
        lines.append(
            f"| keep_production_as_is | {'Yes' if evidence.get('keep_production_as_is', True) else '**No**'} |"
        )
        lines.append(
            f"| best_variant_by_logloss | `{evidence.get('best_variant_by_logloss') or 'N/A'}` |"
        )
        consistent_promote = evidence.get("consistent_promote_variants") or []
        lines.append(
            f"| consistent_promote_variants | `{', '.join(consistent_promote) or 'none'}` |"
        )
        lines.append(f"| blockers_count | {evidence.get('blockers_count', 0)} |")
        pma = evidence.get("whether_to_try_pma_or_isab_next", False)
        lines.append(f"| pma_isab_signal | {'⭐ Yes' if pma else '— No'} |")
        lines.append("")

        ranking = evidence.get("variant_ranking_summary") or []
        if ranking:
            lines.append("### Variant Ranking (by logloss, lower is better)")
            lines.append("")
            lines.append("| Rank | Variant | logloss_mean |")
            lines.append("|------|---------|-------------|")
            for r in ranking:
                ll = r.get("logloss_mean")
                ll_str = f"{ll:.4f}" if ll is not None else "N/A"
                lines.append(f"| {r.get('rank','?')} | **{r.get('variant','?')}** | {ll_str} |")
            lines.append("")

    # --- Next step ---
    lines.append("## Recommended Next Step")
    lines.append("")
    for step_line in bundle.get("recommended_next_step", "").splitlines():
        lines.append(step_line)
    lines.append("")

    if ready:
        lines.append(
            "> ✅ **You may enter promotion review.** "
            "Review all accepted campaign evidence and confirm the signal is genuine "
            "before any production change. Production is NEVER changed automatically."
        )
    else:
        lines.append(
            "> ❌ **Do not enter promotion review yet.** "
            "Run more archcomp or archcomp_full campaigns to accumulate accepted evidence."
        )
    lines.append("")

    lines.append("---")
    lines.append(
        f"*Accepted campaign review bundle generated by promotion_review.py "
        f"v{ACCEPTED_REVIEW_BUNDLE_SCHEMA_VERSION}*"
    )
    lines.append("")
    return "\n".join(lines)


def save_accepted_campaign_review_bundle_artifacts(
    history: list[dict[str, Any]],
    stability: dict[str, Any],
    data_dir: str | Path = "data",
    promotion_gate: dict[str, Any] | None = None,
    regression_alert: dict[str, Any] | None = None,
    comparability_result: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Save accepted_campaign_review_bundle.json and .md to data_dir."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    bundle = build_accepted_campaign_review_bundle(
        history,
        stability,
        promotion_gate=promotion_gate,
        regression_alert=regression_alert,
        comparability_result=comparability_result,
    )
    paths: dict[str, str] = {}

    json_path = data_dir / "accepted_campaign_review_bundle.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, ensure_ascii=False)
        f.write("\n")
    paths["accepted_campaign_review_bundle.json"] = str(json_path)

    md_path = data_dir / "accepted_campaign_review_bundle.md"
    md_path.write_text(
        build_accepted_campaign_review_bundle_md(bundle), encoding="utf-8"
    )
    paths["accepted_campaign_review_bundle.md"] = str(md_path)

    return paths
