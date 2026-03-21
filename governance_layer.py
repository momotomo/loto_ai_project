"""governance_layer.py

Governance layer for the loto-ai campaign infrastructure.

Generates four governance artifacts from campaign history:

1. trend_summary  — rolling-window trend of variant rankings, metrics,
                    recommendations, and pairwise signals.
2. regression_alert — compares latest campaign against recent baseline and
                    flags deterioration in metrics, rankings, or signals.
3. promotion_gate — readiness gate: "is it safe to enter promotion review?"
                    Does NOT update production; only advises whether to proceed.
4. governance_report — combined Markdown summary for human operators.

Output paths (all under data_dir):
  data/trend_summary.json
  data/trend_summary.md
  data/regression_alert.json
  data/regression_alert.md
  data/promotion_gate.json
  data/promotion_gate.md
  data/governance_report.md

Reading order for operators
----------------------------
Start with data/governance_report.md — it brings together all signals in one place.
For details drill into individual JSON/MD artifacts.

Schema versions
---------------
  TREND_SUMMARY_SCHEMA_VERSION    = 1
  REGRESSION_ALERT_SCHEMA_VERSION = 1
  PROMOTION_GATE_SCHEMA_VERSION   = 1
  GOVERNANCE_REPORT_SCHEMA_VERSION = 1
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from comparability_checker import check_history_comparability, save_comparability_artifacts
from decision_policy import (
    DECISION_BENCHMARK_POLICY,
    build_benchmark_lock_md,
    build_benchmark_lock_artifact,
    check_campaign_acceptance,
    build_campaign_acceptance_md,
    save_benchmark_lock_artifacts,
    save_campaign_acceptance_artifacts,
)

TREND_SUMMARY_SCHEMA_VERSION = 1
REGRESSION_ALERT_SCHEMA_VERSION = 1
PROMOTION_GATE_SCHEMA_VERSION = 1
GOVERNANCE_REPORT_SCHEMA_VERSION = 1

# Default rolling window for trend analysis
DEFAULT_TREND_WINDOW = 5

# Thresholds
_LOGLOSS_REGRESSION_THRESHOLD = 0.005   # Δlogloss > this = notable regression
_RANK_DROP_THRESHOLD = 1                # rank drop ≥ this = notable drop
_PAIRWISE_DROP_THRESHOLD = 0.1          # both_pass_rate drop ≥ this = signal loss

# Promotion gate conditions (min values required to pass each condition)
_MIN_CONSECUTIVE_CHALLENGER = 2
_MIN_CONSECUTIVE_ACTION = 2
_MIN_SEEDS = 3
_MIN_LOTO_TYPES = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _trend_direction(values: list[float]) -> str:
    """Return 'improving', 'worsening', 'stable', or 'insufficient_data'.

    For logloss/brier/ece: lower is better, so declining values = improving.
    Uses first-half vs second-half mean comparison.
    """
    if len(values) < 2:
        return "insufficient_data"
    mid = len(values) // 2
    first_half = values[:mid]
    second_half = values[mid:]
    if not first_half or not second_half:
        return "insufficient_data"
    first_mean = sum(first_half) / len(first_half)
    second_mean = sum(second_half) / len(second_half)
    delta = second_mean - first_mean
    if abs(delta) < 1e-6:
        return "stable"
    # Lower is better for all tracked metrics
    return "improving" if delta < 0 else "worsening"


def _rank_trend_direction(values: list[int | None]) -> str:
    """Return 'improving' (rank going down = better), 'worsening', 'stable', or 'insufficient_data'."""
    clean = [v for v in values if v is not None]
    if len(clean) < 2:
        return "insufficient_data"
    mid = len(clean) // 2
    first_mean = sum(clean[:mid]) / mid
    second_mean = sum(clean[mid:]) / (len(clean) - mid)
    delta = second_mean - first_mean
    if abs(delta) < 0.5:
        return "stable"
    return "improving" if delta < 0 else "worsening"


# ---------------------------------------------------------------------------
# 1. Trend Summary
# ---------------------------------------------------------------------------


def build_trend_summary(
    history: list[dict[str, Any]],
    window_size: int = DEFAULT_TREND_WINDOW,
) -> dict[str, Any]:
    """Build a rolling-window trend summary from campaign history.

    Covers:
      - variant rank history and trend
      - logloss trend per variant
      - recommendation / challenger history
      - pairwise signal history
      - keep_production_as_is streak
    """
    window = history[-window_size:] if len(history) > window_size else history
    campaigns_considered = [e.get("campaign_name", "?") for e in window]

    # -- Variant rank history --
    all_variants: set[str] = set()
    for entry in window:
        for r in (entry.get("variant_ranking_summary") or []):
            if "variant" in r:
                all_variants.add(r["variant"])

    variant_rank_history: dict[str, Any] = {}
    for v in sorted(all_variants):
        ranks: list[int | None] = []
        logloss_vals: list[float] = []
        for entry in window:
            found = False
            for r in (entry.get("variant_ranking_summary") or []):
                if r.get("variant") == v:
                    ranks.append(r.get("rank"))
                    ll = r.get("logloss_mean")
                    if ll is not None:
                        logloss_vals.append(float(ll))
                    found = True
                    break
            if not found:
                ranks.append(None)
        variant_rank_history[v] = {
            "ranks": ranks,
            "rank_trend": _rank_trend_direction(ranks),
        }

    # -- Metric (logloss) trends per variant --
    metric_trends: dict[str, Any] = {}
    for v in sorted(all_variants):
        logloss_vals = []
        for entry in window:
            for r in (entry.get("variant_ranking_summary") or []):
                if r.get("variant") == v:
                    ll = r.get("logloss_mean")
                    if ll is not None:
                        logloss_vals.append(float(ll))
                    break
        metric_trends[v] = {
            "logloss": {
                "values": logloss_vals,
                "trend": _trend_direction(logloss_vals),
            }
        }

    # -- Recommendation history --
    recommendation_history = [
        {
            "campaign_name": e.get("campaign_name", "?"),
            "action": e.get("recommended_next_action"),
            "challenger": e.get("recommended_challenger"),
            "keep_production": e.get("keep_production_as_is", True),
        }
        for e in window
    ]

    # -- keep_production streak (from most recent, consecutive True) --
    keep_streak = 0
    for entry in reversed(window):
        if entry.get("keep_production_as_is", True):
            keep_streak += 1
        else:
            break

    # -- Pairwise signal history --
    all_pairwise_keys: set[str] = set()
    for entry in window:
        all_pairwise_keys.update((entry.get("key_pairwise_signals") or {}).keys())

    pairwise_signal_history: dict[str, list[dict]] = {}
    for key in sorted(all_pairwise_keys):
        pairwise_signal_history[key] = [
            {
                "campaign_name": e.get("campaign_name", "?"),
                "both_pass_rate": (
                    (e.get("key_pairwise_signals") or {}).get(key) or {}
                ).get("both_pass_rate"),
                "both_pass_count": (
                    (e.get("key_pairwise_signals") or {}).get(key) or {}
                ).get("both_pass_count"),
                "run_count": (
                    (e.get("key_pairwise_signals") or {}).get(key) or {}
                ).get("run_count"),
            }
            for e in window
        ]

    # -- Summary of consistent patterns --
    action_counts: dict[str, int] = {}
    for e in window:
        a = e.get("recommended_next_action") or "unknown"
        action_counts[a] = action_counts.get(a, 0) + 1
    dominant_action = max(action_counts, key=lambda k: action_counts[k]) if action_counts else None

    challenger_counts: dict[str, int] = {}
    for e in window:
        c = e.get("recommended_challenger") or "none"
        challenger_counts[c] = challenger_counts.get(c, 0) + 1
    dominant_challenger = (
        max(challenger_counts, key=lambda k: challenger_counts[k])
        if challenger_counts else None
    )

    # -- Comparability assessment for the window --
    comp_result = check_history_comparability(window)
    comparability_note = comp_result.get("summary", "")
    comparability_severity = comp_result.get("overall_severity", "ok")
    comparability_overall = comp_result.get("overall_comparable", True)

    return {
        "schema_version": TREND_SUMMARY_SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "tracked_window_size": window_size,
        "total_campaigns": len(history),
        "campaigns_considered": campaigns_considered,
        "variant_rank_history": variant_rank_history,
        "metric_trends": metric_trends,
        "recommendation_history": recommendation_history,
        "keep_production_streak": keep_streak,
        "pairwise_signal_history": pairwise_signal_history,
        "dominant_action": dominant_action,
        "dominant_challenger": dominant_challenger,
        "action_distribution": action_counts,
        "comparability_overall": comparability_overall,
        "comparability_severity": comparability_severity,
        "comparability_note": comparability_note,
    }


def build_trend_summary_md(trend: dict[str, Any]) -> str:
    """Build a Markdown trend summary document."""
    lines: list[str] = []
    lines.append("# Trend Summary")
    lines.append("")
    lines.append(f"Generated: {trend.get('generated_at', '')}")
    lines.append(
        f"Window: last {trend.get('tracked_window_size', '?')} campaigns "
        f"(total {trend.get('total_campaigns', 0)} in history)"
    )
    lines.append("")
    lines.append(
        f"Campaigns considered: "
        + ", ".join(f"**{c}**" for c in (trend.get("campaigns_considered") or []))
    )
    lines.append("")

    # Comparability note
    comp_sev = trend.get("comparability_severity", "ok")
    comp_overall = trend.get("comparability_overall", True)
    comp_note = trend.get("comparability_note", "")
    if comp_sev == "error":
        lines.append(
            f"> ❌ **Comparability issue**: {comp_note}  "
        )
        lines.append(
            "> Trend analysis below may be unreliable — campaigns were run under different conditions."
        )
    elif comp_sev == "warning":
        lines.append(f"> ⚠️ **Comparability warning**: {comp_note}")
    else:
        lines.append(f"> ✅ **Comparability**: {comp_note or 'All campaigns in window are comparable.'}")
    lines.append("")

    # Recommendation history
    lines.append("## Recommendation History")
    lines.append("")
    lines.append("| Campaign | Action | Challenger | keep_production |")
    lines.append("|----------|--------|------------|-----------------|")
    for r in (trend.get("recommendation_history") or []):
        keep = "Yes" if r.get("keep_production") else "No"
        lines.append(
            f"| {r.get('campaign_name','?')} | `{r.get('action') or '?'}` "
            f"| {r.get('challenger') or '—'} | {keep} |"
        )
    lines.append("")

    dom = trend.get("dominant_action") or "?"
    dom_c = trend.get("dominant_challenger") or "?"
    lines.append(f"**Dominant action**: `{dom}`")
    lines.append(f"**Dominant challenger**: `{dom_c}`")
    lines.append(f"**keep_production streak**: {trend.get('keep_production_streak', 0)} campaigns")
    lines.append("")

    # Variant rank history
    lines.append("## Variant Rank History")
    lines.append("")
    lines.append("*(rank: 1 = best, lower logloss)*")
    lines.append("")

    campaigns = trend.get("campaigns_considered") or []
    rank_history = trend.get("variant_rank_history") or {}
    metric_trends = trend.get("metric_trends") or {}

    # Build header
    header_cols = ["variant"] + [f"r{i+1}" for i in range(len(campaigns))] + ["rank_trend", "logloss_trend"]
    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append("|" + "|".join("-" * max(len(c), 3) for c in header_cols) + "|")

    for v in sorted(rank_history.keys()):
        ranks = rank_history[v].get("ranks") or []
        rank_trend = rank_history[v].get("rank_trend", "?")
        ll_trend = (metric_trends.get(v) or {}).get("logloss", {}).get("trend", "?")
        rank_strs = [str(r) if r is not None else "—" for r in ranks]
        row = [v] + rank_strs + [rank_trend, ll_trend]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Logloss trend table
    lines.append("## Logloss Trend")
    lines.append("")
    lines.append("*(lower is better; 'improving' means logloss is decreasing over the window)*")
    lines.append("")

    lines.append("| variant | " + " | ".join(f"ll({c[:8]})" for c in campaigns) + " | trend |")
    lines.append("|---------|" + "|".join("------" for _ in campaigns) + "|-------|")
    for v in sorted(metric_trends.keys()):
        ll_data = (metric_trends.get(v) or {}).get("logloss") or {}
        vals = ll_data.get("values") or []
        trend_dir = ll_data.get("trend", "?")
        val_strs = [f"{x:.4f}" for x in vals] if vals else []
        # Pad if fewer values than campaigns
        while len(val_strs) < len(campaigns):
            val_strs.append("—")
        lines.append("| " + v + " | " + " | ".join(val_strs) + f" | {trend_dir} |")
    lines.append("")

    # Pairwise signal history
    pairwise_hist = trend.get("pairwise_signal_history") or {}
    if pairwise_hist:
        lines.append("## Pairwise Signal History")
        lines.append("")
        lines.append("*(both_pass_rate = fraction of seeds where both CI and permutation tests pass)*")
        lines.append("")
        lines.append("| comparison | " + " | ".join(c[:8] for c in campaigns) + " |")
        lines.append("|------------|" + "|".join("------" for _ in campaigns) + "|")
        for key in sorted(pairwise_hist.keys()):
            entries = pairwise_hist[key]
            rates = []
            for e in entries:
                r = e.get("both_pass_rate")
                rates.append(f"{r:.3f}" if r is not None else "—")
            lines.append(f"| {key} | " + " | ".join(rates) + " |")
        lines.append("")

    lines.append("---")
    lines.append(
        f"*Trend summary generated by governance_layer.py v{TREND_SUMMARY_SCHEMA_VERSION}*"
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. Regression Alert
# ---------------------------------------------------------------------------


def build_regression_alert(
    history: list[dict[str, Any]],
    window_size: int = DEFAULT_TREND_WINDOW,
) -> dict[str, Any]:
    """Compare latest campaign against recent baseline and flag regressions.

    Baseline = average of prior campaigns in the window (excluding the latest).
    Alert levels:
      none   — no significant changes detected
      low    — minor metric regression or single ranking drop
      medium — ranking drop of core variants, logloss regression > threshold,
               or pairwise signal loss
      high   — multiple simultaneous regressions or recommendation reversal
    """
    if not history:
        return {
            "schema_version": REGRESSION_ALERT_SCHEMA_VERSION,
            "generated_at": _now_iso(),
            "alert_level": "none",
            "latest_campaign": None,
            "baseline_campaigns": [],
            "affected_variants": [],
            "ranking_drop": {},
            "metric_regressions": {},
            "pairwise_signal_loss": {},
            "recommendation_instability": {},
            "suspected_causes": [],
            "summary": "No campaign history available.",
        }

    latest = history[-1]
    window = history[-window_size:] if len(history) > window_size else history
    baseline_entries = window[:-1]  # All but latest

    latest_name = latest.get("campaign_name", "?")
    baseline_names = [e.get("campaign_name", "?") for e in baseline_entries]

    alert_signals: list[str] = []
    affected_variants: set[str] = set()
    ranking_drop: dict[str, Any] = {}
    metric_regressions: dict[str, Any] = {}
    pairwise_signal_loss: dict[str, Any] = {}

    # -- Ranking drop analysis --
    if baseline_entries:
        # Build average rank per variant in baseline
        baseline_rank_sum: dict[str, list[int]] = {}
        for entry in baseline_entries:
            for r in (entry.get("variant_ranking_summary") or []):
                v = r.get("variant")
                rank = r.get("rank")
                if v and rank is not None:
                    baseline_rank_sum.setdefault(v, []).append(rank)
        baseline_avg_rank: dict[str, float] = {
            v: sum(ranks) / len(ranks)
            for v, ranks in baseline_rank_sum.items()
            if ranks
        }

        latest_rank_map: dict[str, int] = {}
        for r in (latest.get("variant_ranking_summary") or []):
            v = r.get("variant")
            rank = r.get("rank")
            if v and rank is not None:
                latest_rank_map[v] = rank

        for v, curr_rank in latest_rank_map.items():
            prev_avg = baseline_avg_rank.get(v)
            if prev_avg is None:
                continue
            drop = curr_rank - prev_avg
            if drop >= _RANK_DROP_THRESHOLD:
                ranking_drop[v] = {
                    "prev_avg_rank": round(prev_avg, 2),
                    "curr_rank": curr_rank,
                    "drop": round(drop, 2),
                }
                affected_variants.add(v)
                signal = f"{v} rank dropped from {prev_avg:.1f} → {curr_rank}"
                if drop >= 2:
                    alert_signals.extend([signal, signal])  # count double for severity
                else:
                    alert_signals.append(signal)

        # -- Logloss regression --
        baseline_ll_sum: dict[str, list[float]] = {}
        for entry in baseline_entries:
            for r in (entry.get("variant_ranking_summary") or []):
                v = r.get("variant")
                ll = r.get("logloss_mean")
                if v and ll is not None:
                    baseline_ll_sum.setdefault(v, []).append(float(ll))
        baseline_avg_ll: dict[str, float] = {
            v: sum(lls) / len(lls)
            for v, lls in baseline_ll_sum.items()
            if lls
        }

        latest_ll_map: dict[str, float] = {}
        for r in (latest.get("variant_ranking_summary") or []):
            v = r.get("variant")
            ll = r.get("logloss_mean")
            if v and ll is not None:
                latest_ll_map[v] = float(ll)

        for v, curr_ll in latest_ll_map.items():
            prev_avg_ll = baseline_avg_ll.get(v)
            if prev_avg_ll is None:
                continue
            delta = curr_ll - prev_avg_ll
            if delta > _LOGLOSS_REGRESSION_THRESHOLD:
                metric_regressions[v] = {
                    "metric": "logloss",
                    "prev_avg": round(prev_avg_ll, 5),
                    "curr": round(curr_ll, 5),
                    "delta": round(delta, 5),
                    "direction": "worsened",
                }
                affected_variants.add(v)
                alert_signals.append(
                    f"{v} logloss worsened: {prev_avg_ll:.4f} → {curr_ll:.4f} (Δ+{delta:.4f})"
                )

        # -- Pairwise signal loss --
        baseline_rate_sum: dict[str, list[float]] = {}
        for entry in baseline_entries:
            for key, sig in (entry.get("key_pairwise_signals") or {}).items():
                r = sig.get("both_pass_rate")
                if r is not None:
                    baseline_rate_sum.setdefault(key, []).append(float(r))
        baseline_avg_rate: dict[str, float] = {
            k: sum(rs) / len(rs)
            for k, rs in baseline_rate_sum.items()
            if rs
        }

        latest_pw = latest.get("key_pairwise_signals") or {}
        for key, baseline_avg in baseline_avg_rate.items():
            curr_sig = latest_pw.get(key) or {}
            curr_rate = curr_sig.get("both_pass_rate")
            if curr_rate is None:
                continue
            drop = baseline_avg - float(curr_rate)
            if drop >= _PAIRWISE_DROP_THRESHOLD:
                pairwise_signal_loss[key] = {
                    "prev_avg_rate": round(baseline_avg, 4),
                    "curr_rate": round(float(curr_rate), 4),
                    "drop": round(drop, 4),
                }
                alert_signals.append(
                    f"Pairwise signal loss in {key}: "
                    f"{baseline_avg:.3f} → {curr_rate:.3f} (Δ-{drop:.3f})"
                )

    # -- Recommendation instability --
    prev_action = history[-2].get("recommended_next_action") if len(history) >= 2 else None
    curr_action = latest.get("recommended_next_action")
    prev_challenger = history[-2].get("recommended_challenger") if len(history) >= 2 else None
    curr_challenger = latest.get("recommended_challenger")
    action_changed = prev_action != curr_action if prev_action is not None else False
    challenger_changed = prev_challenger != curr_challenger if prev_challenger is not None else False

    recommendation_instability = {
        "action_changed": action_changed,
        "prev_action": prev_action,
        "curr_action": curr_action,
        "challenger_changed": challenger_changed,
        "prev_challenger": prev_challenger,
        "curr_challenger": curr_challenger,
    }
    # Recommendation reversal from positive to negative = strong signal
    positive_actions = {"consider_promotion", "run_more_seeds"}
    if action_changed and prev_action in positive_actions and curr_action == "hold":
        alert_signals.extend([
            f"Recommendation regressed: {prev_action} → hold",
            f"Recommendation regressed: {prev_action} → hold",
        ])
    elif action_changed:
        alert_signals.append(f"Recommendation changed: {prev_action} → {curr_action}")

    # -- Alert level --
    n = len(alert_signals)
    if n == 0:
        alert_level = "none"
        summary = "No significant regression detected since the previous campaign."
    elif n <= 1:
        alert_level = "low"
        summary = f"Minor regression detected ({n} signal). Review metric trends."
    elif n <= 3:
        alert_level = "medium"
        summary = (
            f"Notable regression detected ({n} signals). "
            "Review ranking drops and metric trends before proceeding."
        )
    else:
        alert_level = "high"
        summary = (
            f"Significant regression detected ({n} signals across metrics, "
            "rankings, and/or recommendation). Investigate before any promotion decision."
        )

    # -- Suspected causes (heuristic) --
    suspected_causes: list[str] = []
    if ranking_drop:
        top_dropper = max(ranking_drop, key=lambda v: ranking_drop[v].get("drop", 0))
        suspected_causes.append(
            f"{top_dropper} had the largest rank drop "
            f"(Δ{ranking_drop[top_dropper].get('drop', 0):.1f}). "
            "Possible seed variance or data shift."
        )
    if metric_regressions:
        suspected_causes.append(
            f"Logloss regression in: {', '.join(metric_regressions.keys())}. "
            "Consider increasing seeds or switching to archcomp_full profile."
        )
    if pairwise_signal_loss:
        suspected_causes.append(
            f"Pairwise signal loss in: {', '.join(pairwise_signal_loss.keys())}. "
            "Signal may be noise; additional seeds needed for confirmation."
        )
    if action_changed and not any([ranking_drop, metric_regressions]):
        suspected_causes.append(
            "Recommendation instability without metric regression — "
            "may be borderline threshold crossing. Run more seeds."
        )

    # -- Comparability check (latest vs baseline) --
    window = history[-window_size:] if len(history) > window_size else history
    comp_result = check_history_comparability(window)
    comparability_caution = not comp_result.get("overall_comparable", True)
    comparability_note = comp_result.get("summary", "")
    if comparability_caution and alert_level != "high":
        summary = (
            f"[COMPARABILITY CAUTION] {summary}  "
            "Note: one or more campaign pairs are not fully comparable — "
            "this alert may reflect condition changes rather than genuine regression."
        )

    return {
        "schema_version": REGRESSION_ALERT_SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "alert_level": alert_level,
        "comparability_caution": comparability_caution,
        "comparability_note": comparability_note,
        "latest_campaign": latest_name,
        "baseline_campaigns": baseline_names,
        "affected_variants": sorted(affected_variants),
        "ranking_drop": ranking_drop,
        "metric_regressions": metric_regressions,
        "pairwise_signal_loss": pairwise_signal_loss,
        "recommendation_instability": recommendation_instability,
        "suspected_causes": suspected_causes,
        "summary": summary,
    }


def build_regression_alert_md(alert: dict[str, Any]) -> str:
    """Build a Markdown regression alert document."""
    level = alert.get("alert_level", "none")
    level_emoji = {"none": "✅", "low": "⚠️", "medium": "🔶", "high": "🔴"}.get(level, "❓")

    lines: list[str] = []
    lines.append("# Regression Alert")
    lines.append("")
    lines.append(f"Generated: {alert.get('generated_at', '')}")
    lines.append("")
    lines.append(f"## Alert Level: {level_emoji} **{level.upper()}**")
    lines.append("")
    lines.append(f"**{alert.get('summary', '')}**")
    lines.append("")

    latest = alert.get("latest_campaign", "?")
    baseline = ", ".join(alert.get("baseline_campaigns") or [])
    lines.append(f"- **Latest campaign**: {latest}")
    lines.append(f"- **Baseline campaigns**: {baseline or 'none'}")
    affected = ", ".join(alert.get("affected_variants") or []) or "none"
    lines.append(f"- **Affected variants**: {affected}")
    lines.append("")

    ranking_drop = alert.get("ranking_drop") or {}
    if ranking_drop:
        lines.append("## Ranking Drops")
        lines.append("")
        lines.append("| variant | prev_avg_rank | curr_rank | drop |")
        lines.append("|---------|--------------|-----------|------|")
        for v, d in sorted(ranking_drop.items(), key=lambda x: -x[1].get("drop", 0)):
            lines.append(
                f"| **{v}** | {d.get('prev_avg_rank','?')} "
                f"| {d.get('curr_rank','?')} | +{d.get('drop','?')} |"
            )
        lines.append("")

    metric_reg = alert.get("metric_regressions") or {}
    if metric_reg:
        lines.append("## Metric Regressions")
        lines.append("")
        lines.append("| variant | metric | prev_avg | curr | delta |")
        lines.append("|---------|--------|---------|------|-------|")
        for v, d in sorted(metric_reg.items()):
            lines.append(
                f"| **{v}** | {d.get('metric','?')} "
                f"| {d.get('prev_avg','?')} | {d.get('curr','?')} "
                f"| +{d.get('delta','?')} |"
            )
        lines.append("")

    pw_loss = alert.get("pairwise_signal_loss") or {}
    if pw_loss:
        lines.append("## Pairwise Signal Loss")
        lines.append("")
        lines.append("| comparison | prev_avg_rate | curr_rate | drop |")
        lines.append("|------------|-------------|-----------|------|")
        for key, d in sorted(pw_loss.items()):
            lines.append(
                f"| {key} | {d.get('prev_avg_rate','?')} "
                f"| {d.get('curr_rate','?')} | -{d.get('drop','?')} |"
            )
        lines.append("")

    rec_inst = alert.get("recommendation_instability") or {}
    if rec_inst.get("action_changed") or rec_inst.get("challenger_changed"):
        lines.append("## Recommendation Instability")
        lines.append("")
        if rec_inst.get("action_changed"):
            lines.append(
                f"- **Action changed**: `{rec_inst.get('prev_action','?')}` "
                f"→ `{rec_inst.get('curr_action','?')}`"
            )
        if rec_inst.get("challenger_changed"):
            lines.append(
                f"- **Challenger changed**: `{rec_inst.get('prev_challenger','?')}` "
                f"→ `{rec_inst.get('curr_challenger','?')}`"
            )
        lines.append("")

    causes = alert.get("suspected_causes") or []
    if causes:
        lines.append("## Suspected Causes")
        lines.append("")
        for cause in causes:
            lines.append(f"- {cause}")
        lines.append("")

    if level == "none":
        lines.append(
            "> No action required — continue monitoring with the next campaign."
        )
    elif level == "low":
        lines.append(
            "> Review trends but no immediate action required. "
            "Consider running archcomp_full if this persists."
        )
    elif level == "medium":
        lines.append(
            "> **Investigate before any promotion decision.** "
            "Check seed variance and consider additional campaign runs."
        )
    else:
        lines.append(
            "> **Do not proceed with promotion.** "
            "Resolve regression signals before re-evaluating."
        )
    lines.append("")

    lines.append("---")
    lines.append(
        f"*Regression alert generated by governance_layer.py v{REGRESSION_ALERT_SCHEMA_VERSION}*"
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. Promotion Readiness Gate
# ---------------------------------------------------------------------------


def build_promotion_gate(
    history: list[dict[str, Any]],
    stability: dict[str, Any],
    latest_recommendation: dict[str, Any] | None = None,
    regression_alert: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a promotion readiness gate artifact.

    This gate answers: "Is it safe to enter the promotion review phase?"
    It does NOT update production — it only advises whether to proceed.

    Gate status:
      green  — key conditions met; proceed to promotion review
      yellow — some conditions met; more evidence recommended
      red    — conditions not met or active regression; do not proceed

    Conditions checked (all are informational, not gates in isolation):
      1. consistent_challenger: same challenger recommended ≥ N consecutive times
      2. positive_action_signal: consider_promotion or run_more_seeds streak ≥ N
      3. consider_promotion_active: latest action == consider_promotion
      4. no_high_regression: regression alert level is not high
      5. consistent_promote_in_latest: consistent_promote_variants non-empty
      6. enough_seeds: seeds ≥ 3 in latest campaign
      7. enough_loto_types: loto_types ≥ 2 in latest campaign
      8. consecutive_positive_pma_signal: whether_to_try_pma_or_isab_next true ≥ 2 campaigns
    """
    if not history:
        return {
            "schema_version": PROMOTION_GATE_SCHEMA_VERSION,
            "generated_at": _now_iso(),
            "gate_status": "red",
            "candidate_variant": None,
            "evidence_window": 0,
            "conditions_passed": [],
            "blockers": ["No campaign history available."],
            "rationale": "Cannot evaluate readiness without campaign history.",
            "next_required_action": "Run at least one campaign with run_campaign.py.",
        }

    latest = history[-1]
    window = history[-DEFAULT_TREND_WINDOW:]

    candidate = latest.get("recommended_challenger")
    evidence_window = len(window)

    conditions_passed: list[str] = []
    blockers: list[str] = []

    # Condition 1: consistent challenger
    consec_challenger = stability.get("consecutive_same_challenger", 0)
    if consec_challenger >= _MIN_CONSECUTIVE_CHALLENGER:
        conditions_passed.append(
            f"consistent_challenger: {candidate!r} recommended "
            f"{consec_challenger} consecutive times"
        )
    else:
        blockers.append(
            f"Challenger not yet consistent: {candidate!r} recommended "
            f"{consec_challenger} consecutive time(s) (need ≥{_MIN_CONSECUTIVE_CHALLENGER})"
        )

    # Condition 2: positive action signal
    latest_action = stability.get("latest_action", "")
    consec_action = stability.get("consecutive_same_action", 0)
    if latest_action in ("consider_promotion", "run_more_seeds") and consec_action >= _MIN_CONSECUTIVE_ACTION:
        conditions_passed.append(
            f"positive_action_signal: `{latest_action}` for "
            f"{consec_action} consecutive campaigns"
        )
    else:
        blockers.append(
            f"No sustained positive action signal: latest=`{latest_action}`, "
            f"consecutive={consec_action} (need ≥{_MIN_CONSECUTIVE_ACTION} for "
            "consider_promotion or run_more_seeds)"
        )

    # Condition 3: consider_promotion active
    if latest_action == "consider_promotion":
        conditions_passed.append(
            "consider_promotion_active: latest campaign recommends consider_promotion"
        )
    # (not adding as blocker — it's a stronger signal when present, but not required)

    # Condition 4: no high regression
    alert_level = (regression_alert or {}).get("alert_level", "none")
    if alert_level != "high":
        conditions_passed.append(
            f"no_high_regression: regression alert level is '{alert_level}'"
        )
    else:
        blockers.append(
            "Active HIGH regression alert — resolve before promotion review."
        )

    # Condition 5: consistent promote variants in latest
    consistent_promote = latest.get("consistent_promote_variants") or []
    if consistent_promote:
        conditions_passed.append(
            f"consistent_promote_in_latest: {', '.join(consistent_promote)} "
            "passed promotion guardrails in ≥50% of loto_types"
        )
    else:
        blockers.append(
            "No variant passed promotion guardrails (consistent_promote_variants is empty)."
        )

    # Condition 6: enough seeds
    latest_seeds = latest.get("seeds") or []
    if len(latest_seeds) >= _MIN_SEEDS:
        conditions_passed.append(
            f"enough_seeds: {len(latest_seeds)} seeds evaluated"
        )
    else:
        blockers.append(
            f"Insufficient seeds: {len(latest_seeds)} (need ≥{_MIN_SEEDS} for reliable signal)"
        )

    # Condition 7: enough loto_types
    latest_loto_types = latest.get("loto_types") or []
    if len(latest_loto_types) >= _MIN_LOTO_TYPES:
        conditions_passed.append(
            f"enough_loto_types: {len(latest_loto_types)} loto_types evaluated"
        )
    else:
        blockers.append(
            f"Insufficient loto_types: {len(latest_loto_types)} "
            f"(need ≥{_MIN_LOTO_TYPES} for cross-loto signal)"
        )

    # Condition 8: consecutive PMA/ISAB positive signal
    consec_pma = sum(
        1 for e in reversed(window)
        if e.get("whether_to_try_pma_or_isab_next", False)
    )
    # Count only consecutive from most recent
    consec_pma_streak = 0
    for e in reversed(window):
        if e.get("whether_to_try_pma_or_isab_next", False):
            consec_pma_streak += 1
        else:
            break
    if consec_pma_streak >= 2:
        conditions_passed.append(
            f"consecutive_positive_pma_signal: PMA/ISAB signal in "
            f"{consec_pma_streak} consecutive campaigns"
        )

    # Condition 9: comparability (campaigns must be comparable for gate to be green)
    comp_result = check_history_comparability(window)
    comparability_ok = comp_result.get("overall_comparable", True)
    comparability_severity = comp_result.get("overall_severity", "ok")
    comparability_note = comp_result.get("summary", "")
    if comparability_ok:
        if comparability_severity == "ok":
            conditions_passed.append(
                "comparability_ok: all campaigns in evidence window are fully comparable"
            )
        else:
            conditions_passed.append(
                f"comparability_ok: campaigns are comparable with warnings — {comparability_note}"
            )
    else:
        blockers.append(
            f"comparability_error: {comparability_note}  "
            "Cannot trust trend-based promotion signal when campaigns are not comparable."
        )

    # -- Determine gate status --
    n_passed = len(conditions_passed)
    n_blockers = len(blockers)

    # Must-have for green: no high regression, consistent_promote_variants, consistent challenger,
    # and comparability_ok
    must_have_for_green = [
        "no_high_regression",
        "consistent_promote_in_latest",
        "consistent_challenger",
        "comparability_ok",
    ]
    has_must_haves = all(
        any(c.startswith(key) for c in conditions_passed)
        for key in must_have_for_green
    )

    if has_must_haves and n_blockers <= 1:
        gate_status = "green"
    elif n_passed >= 2 and alert_level != "high":
        gate_status = "yellow"
    else:
        gate_status = "red"

    # Build rationale
    if gate_status == "green":
        rationale = (
            f"Gate is GREEN: {candidate!r} shows consistent positive signal "
            f"across {len(latest_loto_types)} loto_types and {len(latest_seeds)} seeds. "
            "Proceed to promotion review."
        )
        next_action = (
            "Review per-loto eval_report details and calibration recommendation. "
            "Run production training only after manual review confirms the signal is genuine."
        )
    elif gate_status == "yellow":
        rationale = (
            f"Gate is YELLOW: Some conditions met ({n_passed} passed, {n_blockers} blockers). "
            "Signal is promising but not yet conclusive."
        )
        next_action = (
            "Address blockers above. Consider running archcomp_full for more seeds, "
            "or wait for another campaign to confirm the trend."
        )
    else:
        rationale = (
            f"Gate is RED: Too many blockers ({n_blockers}) for promotion review. "
            "Current evidence is insufficient or contradictory."
        )
        next_action = (
            "Resolve blockers above before reconsidering promotion. "
            "Run another campaign (archcomp or archcomp_full) to gather more evidence."
        )

    return {
        "schema_version": PROMOTION_GATE_SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "gate_status": gate_status,
        "candidate_variant": candidate,
        "evidence_window": evidence_window,
        "comparability_ok": comparability_ok,
        "comparability_severity": comparability_severity,
        "comparability_note": comparability_note,
        "conditions_passed": conditions_passed,
        "blockers": blockers,
        "rationale": rationale,
        "next_required_action": next_action,
    }


def build_promotion_gate_md(gate: dict[str, Any]) -> str:
    """Build a Markdown promotion readiness gate document."""
    status = gate.get("gate_status", "red")
    emoji = {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(status, "❓")

    lines: list[str] = []
    lines.append("# Promotion Readiness Gate")
    lines.append("")
    lines.append(f"Generated: {gate.get('generated_at', '')}")
    lines.append("")
    lines.append(f"## Gate Status: {emoji} **{status.upper()}**")
    lines.append("")
    lines.append(f"> {gate.get('rationale', '')}")
    lines.append("")

    candidate = gate.get("candidate_variant") or "none"
    evidence = gate.get("evidence_window", 0)
    lines.append(f"- **Candidate variant**: `{candidate}`")
    lines.append(f"- **Evidence window**: {evidence} recent campaign(s)")
    lines.append("")

    passed = gate.get("conditions_passed") or []
    if passed:
        lines.append("## Conditions Passed ✅")
        lines.append("")
        for c in passed:
            lines.append(f"- {c}")
        lines.append("")

    blockers = gate.get("blockers") or []
    if blockers:
        lines.append("## Blockers ❌")
        lines.append("")
        for b in blockers:
            lines.append(f"- {b}")
        lines.append("")

    lines.append("## Next Required Action")
    lines.append("")
    lines.append(gate.get("next_required_action", ""))
    lines.append("")

    lines.append("> **Important**: This gate indicates whether to *enter promotion review*,")
    lines.append("> not whether to *promote to production*. Production is never changed automatically.")
    lines.append("")

    lines.append("---")
    lines.append(
        f"*Promotion gate generated by governance_layer.py v{PROMOTION_GATE_SCHEMA_VERSION}*"
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. Governance Report (combined Markdown)
# ---------------------------------------------------------------------------


def build_governance_report(
    trend_summary: dict[str, Any],
    regression_alert: dict[str, Any],
    promotion_gate: dict[str, Any],
    stability: dict[str, Any],
    latest_entry: dict[str, Any] | None = None,
    comparability_result: dict[str, Any] | None = None,
    acceptance_result: dict[str, Any] | None = None,
) -> str:
    """Build a combined governance Markdown report for human operators.

    This is the first document to read after a campaign run.
    It provides a concise decision surface across all governance signals,
    starting with the decision benchmark policy so the reader knows whether
    this campaign is accepted for decision use at all, followed by
    comparability, and then trend/regression/promotion conclusions.
    """
    lines: list[str] = []
    lines.append("# Governance Report")
    lines.append("")
    lines.append(f"Generated: {_now_iso()}")
    lines.append("")

    if latest_entry:
        campaign_name = latest_entry.get("campaign_name", "?")
        lines.append(f"Latest campaign: **{campaign_name}**")
        lines.append("")

    lines.append(
        "> **Reading order**: Start here. For details, see "
        "`campaign_acceptance.md` → `benchmark_lock.md` → "
        "`comparability_report.md` → `trend_summary.md` → `regression_alert.md` → "
        "`promotion_gate.md` → `campaign_diff_report.md` → "
        "evidence pack in `campaigns/<name>/cross_loto_report.md`."
    )
    lines.append("")

    # --- Decision Benchmark Policy summary ---
    lines.append("## Decision Benchmark Policy")
    lines.append("")
    lines.append(
        "> **Read this first**: Only campaigns that satisfy the Decision Benchmark Policy "
        "are counted toward promotion readiness. "
        "See `benchmark_lock.md` for the full policy."
    )
    lines.append("")
    active_bms = DECISION_BENCHMARK_POLICY.get("active_decision_benchmarks") or []
    excluded = DECISION_BENCHMARK_POLICY.get("excluded_profiles") or []
    lines.append(f"- **Active decision benchmarks**: `{'`, `'.join(active_bms)}`")
    lines.append(f"- **Excluded (sanity only)**: `{'`, `'.join(excluded)}`")
    lines.append(
        f"- **comparability_required**: "
        f"`{DECISION_BENCHMARK_POLICY.get('comparability_required', True)}`"
    )
    lines.append("")

    # --- Current Campaign Acceptance ---
    lines.append("## Current Campaign Acceptance")
    lines.append("")
    if acceptance_result is not None:
        acc_ok = acceptance_result.get("accepted_for_decision_use", False)
        counts = acceptance_result.get("counts_toward_promotion_readiness", False)
        acc_emoji = "✅" if acc_ok else "❌"
        acc_label = "ACCEPTED" if acc_ok else "NOT ACCEPTED"
        counts_emoji = "✅" if counts else "❌"
        lines.append(f"**Status**: {acc_emoji} {acc_label}")
        lines.append(
            f"**Counts toward promotion readiness**: {counts_emoji} `{counts}`"
        )
        lines.append("")
        lines.append(acceptance_result.get("rationale", ""))
        lines.append("")
        acc_failed = acceptance_result.get("failed_requirements") or []
        acc_warnings = acceptance_result.get("warnings") or []
        if acc_failed:
            lines.append("Failed requirements:")
            for f in acc_failed:
                lines.append(f"  - ❌ {f}")
            lines.append("")
        if acc_warnings:
            lines.append("Acceptance warnings:")
            for w in acc_warnings:
                lines.append(f"  - ⚠️ {w}")
            lines.append("")
        if not acc_ok:
            lines.append(
                "> ⚠️ **This campaign is excluded from promotion readiness counting.** "
                "Trend analysis below is for reference only. "
                "See `campaign_acceptance.md` and `benchmark_lock.md` for details."
            )
            lines.append("")
    elif latest_entry is not None:
        # Derive from entry fields
        acc_ok = latest_entry.get("accepted_for_decision_use", False)
        counts = latest_entry.get("counts_toward_promotion_readiness", False)
        acc_emoji = "✅" if acc_ok else "❌"
        acc_label = "ACCEPTED" if acc_ok else "NOT ACCEPTED"
        counts_emoji = "✅" if counts else "❌"
        lines.append(f"**Status**: {acc_emoji} {acc_label}")
        lines.append(
            f"**Counts toward promotion readiness**: {counts_emoji} `{counts}`"
        )
        lines.append("")
        if not acc_ok:
            lines.append(
                "> ⚠️ **This campaign is excluded from promotion readiness counting.** "
                "See `campaign_acceptance.md` for details."
            )
            lines.append("")
    else:
        lines.append("*Acceptance check not available — run with run_campaign.py to generate.*")
        lines.append("")

    # --- Accepted-only stability note ---
    lines.append("## Whether This Campaign Counts Toward Promotion Readiness")
    lines.append("")
    n_accepted = stability.get("total_accepted_campaigns", 0)
    consec_action_acc = stability.get("consecutive_same_action_accepted_only", 0)
    consec_st_acc = stability.get("consecutive_positive_signal_for_settransformer_accepted_only", 0)
    if acceptance_result is not None:
        counts_val = acceptance_result.get("counts_toward_promotion_readiness", False)
    elif latest_entry is not None:
        counts_val = latest_entry.get("counts_toward_promotion_readiness", False)
    else:
        counts_val = False

    if counts_val:
        lines.append(
            "> ✅ **This campaign counts.** "
            "The accepted-only stability metrics below reflect this campaign."
        )
    else:
        lines.append(
            "> ❌ **This campaign does NOT count.** "
            "Comparable campaigns with wrong benchmark/profile/coverage "
            "do not contribute to promotion readiness. "
            "Only `archcomp` and `archcomp_full` campaigns are counted."
        )
    lines.append("")
    lines.append("| Accepted-Only Metric | Value |")
    lines.append("|----------------------|-------|")
    lines.append(f"| Total accepted campaigns | {n_accepted} |")
    lines.append(f"| Consecutive same action (accepted only) | {consec_action_acc} |")
    lines.append(f"| Consecutive settransformer signal (accepted only) | {consec_st_acc} |")
    lines.append("")
    if not counts_val:
        lines.append(
            "> ℹ️ **Note**: Even if this campaign is comparable, the promotion readiness "
            "review requires accepted campaigns only. "
            "Run `archcomp` or `archcomp_full` to accumulate accepted evidence."
        )
        lines.append("")

    # --- Comparability (read before drawing any trend/regression conclusions) ---
    lines.append("## Comparability")
    lines.append("")
    lines.append(
        "> **Check this before trends**: trend and regression conclusions are only valid when "
        "campaigns are comparable (same benchmark, loto coverage, variants, and calibration methods)."
    )
    lines.append("")
    if comparability_result is not None:
        comp_sev = comparability_result.get("overall_severity", "ok")
        comp_ok = comparability_result.get("overall_comparable", True)
        comp_emoji = {"ok": "✅", "warning": "⚠️", "error": "❌"}.get(comp_sev, "?")
        comp_label = "COMPARABLE" if comp_ok else "NOT COMPARABLE"
        lines.append(f"**Status**: {comp_emoji} {comp_label}")
        lines.append("")
        lines.append(comparability_result.get("summary", ""))
        lines.append("")
        if comp_sev == "error":
            lines.append(
                "> ❌ **Action required**: Campaigns are not comparable.  "
                "Trend analysis, regression alerts, and promotion gate below may be unreliable.  "
                "See `comparability_report.md` for details."
            )
        elif comp_sev == "warning":
            lines.append(
                "> ⚠️ Campaigns are comparable with caveats.  "
                "Review `comparability_report.md` before acting on trends."
            )
        else:
            lines.append("> ✅ All campaigns in the evidence window are fully comparable.")
    else:
        # Derive from trend_summary or promotion_gate if no explicit result
        comp_sev = trend_summary.get("comparability_severity") or promotion_gate.get("comparability_severity", "ok")
        comp_ok = trend_summary.get("comparability_overall") if trend_summary.get("comparability_overall") is not None else True
        comp_note = trend_summary.get("comparability_note", "")
        comp_emoji = {"ok": "✅", "warning": "⚠️", "error": "❌"}.get(comp_sev, "?")
        comp_label = "COMPARABLE" if comp_ok else "NOT COMPARABLE"
        lines.append(f"**Status**: {comp_emoji} {comp_label}")
        lines.append("")
        if comp_note:
            lines.append(comp_note)
        lines.append("")
    lines.append("")

    # --- Current recommendation ---
    lines.append("## Current Recommendation")
    lines.append("")
    if latest_entry:
        action = latest_entry.get("recommended_next_action", "?")
        challenger = latest_entry.get("recommended_challenger") or "none"
        keep = "Yes" if latest_entry.get("keep_production_as_is", True) else "**No**"
        blockers_count = latest_entry.get("blockers_count", 0)
        lines.append(f"| Field | Value |")
        lines.append(f"|-------|-------|")
        lines.append(f"| recommended_next_action | `{action}` |")
        lines.append(f"| recommended_challenger | `{challenger}` |")
        lines.append(f"| keep_production_as_is | {keep} |")
        lines.append(f"| blockers_count | {blockers_count} |")
        lines.append("")

        pma = latest_entry.get("whether_to_try_pma_or_isab_next", False)
        if pma:
            lines.append(
                "> ⭐ **PMA/ISAB signal is active** — "
                "settransformer shows consistent attention benefit over deepsets."
            )
        else:
            lines.append(
                "> settransformer has not yet shown consistent advantage over deepsets. "
                "PMA/ISAB exploration should wait."
            )
        lines.append("")

    # --- Gate status ---
    gate_status = promotion_gate.get("gate_status", "red")
    gate_emoji = {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(gate_status, "❓")
    lines.append("## Promotion Readiness Gate")
    lines.append("")
    lines.append(f"**{gate_emoji} {gate_status.upper()}** — {promotion_gate.get('rationale', '')}")
    lines.append("")
    blockers = promotion_gate.get("blockers") or []
    if blockers:
        lines.append("Blockers:")
        for b in blockers:
            lines.append(f"  - {b}")
    else:
        lines.append("No blockers.")
    lines.append("")

    # --- Regression alert ---
    alert_level = regression_alert.get("alert_level", "none")
    alert_emoji = {"none": "✅", "low": "⚠️", "medium": "🔶", "high": "🔴"}.get(alert_level, "❓")
    lines.append("## Regression Alert")
    lines.append("")
    lines.append(
        f"**{alert_emoji} {alert_level.upper()}** — {regression_alert.get('summary', '')}"
    )
    affected = ", ".join(regression_alert.get("affected_variants") or [])
    if affected:
        lines.append(f"Affected variants: {affected}")
    lines.append("")

    # --- Recommendation stability ---
    lines.append("## Recommendation Stability")
    lines.append("")
    total = stability.get("total_campaigns", 0)
    latest_action = stability.get("latest_action") or "?"
    consec_action = stability.get("consecutive_same_action", 0)
    consec_challenger = stability.get("consecutive_same_challenger", 0)
    consec_keep = stability.get("consecutive_keep_production", 0)
    consec_rms = stability.get("consecutive_run_more_seeds", 0)
    consec_st = stability.get("consecutive_positive_signal_for_settransformer", 0)
    consec_ds = stability.get("consecutive_positive_signal_for_deepsets", 0)

    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total campaigns | {total} |")
    lines.append(f"| Latest action | `{latest_action}` |")
    lines.append(f"| Consecutive same action | {consec_action} |")
    lines.append(f"| Consecutive same challenger | {consec_challenger} |")
    lines.append(f"| Consecutive keep_production | {consec_keep} |")
    lines.append(f"| Consecutive run_more_seeds | {consec_rms} |")
    lines.append(f"| Consecutive settransformer+ signal | {consec_st} |")
    lines.append(f"| Consecutive deepsets+ signal | {consec_ds} |")
    lines.append("")

    # Stability guidance
    if consec_rms >= 3:
        lines.append(
            f"> **Signal**: `run_more_seeds` for {consec_rms} consecutive campaigns. "
            "→ Run `archcomp_full` (5 seeds, default preset)."
        )
    elif latest_action == "consider_promotion" and consec_action >= 2:
        lines.append(
            f"> **Signal**: `consider_promotion` for {consec_action} consecutive campaigns. "
            "→ Strong evidence — review promotion gate details."
        )
    elif latest_action == "hold" and consec_action >= 3:
        lines.append(
            f"> **Signal**: `hold` for {consec_action} consecutive campaigns. "
            "→ Architecture differentiation may be limited at current scale."
        )
    if consec_st >= 2:
        lines.append(
            f"> **Signal**: settransformer positive signal for {consec_st} consecutive campaigns. "
            "→ Attention benefit is consistent. PMA/ISAB exploration is motivated."
        )
    lines.append("")

    # --- Trend overview ---
    dom_action = trend_summary.get("dominant_action", "?")
    dom_challenger = trend_summary.get("dominant_challenger", "?")
    keep_streak = trend_summary.get("keep_production_streak", 0)
    window = trend_summary.get("tracked_window_size", 5)
    lines.append("## Recent Trend Overview")
    lines.append("")
    lines.append(
        f"Over the last {window} campaigns: "
        f"dominant action = `{dom_action}`, "
        f"dominant challenger = `{dom_challenger}`, "
        f"keep_production streak = {keep_streak}."
    )
    lines.append("")
    # Rank trend per variant
    rank_history = trend_summary.get("variant_rank_history") or {}
    metric_trends = trend_summary.get("metric_trends") or {}
    if rank_history:
        lines.append("| variant | rank_trend | logloss_trend |")
        lines.append("|---------|-----------|--------------|")
        for v in sorted(rank_history.keys()):
            rt = rank_history[v].get("rank_trend", "?")
            lt = (metric_trends.get(v) or {}).get("logloss", {}).get("trend", "?")
            lines.append(f"| {v} | {rt} | {lt} |")
        lines.append("")

    # --- Why production is not / should be changed ---
    lines.append("## Production Status")
    lines.append("")
    if latest_entry and not latest_entry.get("keep_production_as_is", True):
        lines.append(
            "> **Consider promotion**: The latest campaign recommends considering promotion. "
            "Review the promotion gate and per-loto evidence before any production change."
        )
    else:
        lines.append(
            "> **Production remains unchanged.** "
            "No variant has met all promotion guardrails consistently. "
            "This is the safe default — only change after gate is GREEN and manual review."
        )
    lines.append("")

    # --- PMA / ISAB / HPO guidance ---
    lines.append("## PMA / ISAB / HPO Guidance")
    lines.append("")
    pma_active = (latest_entry or {}).get("whether_to_try_pma_or_isab_next", False)
    if consec_st >= 2 and pma_active:
        lines.append(
            "**PMA/ISAB exploration is motivated** (settransformer positive signal ≥ 2 campaigns)."
        )
        lines.append("")
        lines.append("Conditions to proceed:")
        lines.append(
            "1. `settransformer_vs_deepsets` both_pass_rate ≥ 0.5 for ≥ 2 consecutive campaigns"
        )
        lines.append(
            "2. `consecutive_positive_signal_for_settransformer` ≥ 2 in stability report"
        )
        lines.append("3. No active high regression alert")
    elif consec_st == 1:
        lines.append(
            "settransformer showed positive signal in the latest campaign only. "
            "Wait for ≥ 2 consecutive campaigns before exploring PMA/ISAB."
        )
    else:
        lines.append(
            "settransformer has not shown consistent advantage over deepsets. "
            "PMA/ISAB/HPO exploration should wait."
        )
    lines.append("")
    lines.append("Conditions to proceed to HPO (separate from PMA/ISAB):")
    lines.append("- A single variant consistently wins across all loto_types and ≥ 5 seeds")
    lines.append("- Gate is GREEN and promotion review confirms the advantage is genuine")
    lines.append("- PMA/ISAB improvement (if applicable) has already been confirmed")
    lines.append("")

    lines.append("---")
    lines.append(
        f"*Governance report generated by governance_layer.py v{GOVERNANCE_REPORT_SCHEMA_VERSION}*"
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Save all governance artifacts
# ---------------------------------------------------------------------------


def save_governance_artifacts(
    history: list[dict[str, Any]],
    stability: dict[str, Any],
    data_dir: str | Path = "data",
    window_size: int = DEFAULT_TREND_WINDOW,
    latest_recommendation: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Generate and save all four governance artifacts to data_dir.

    Returns {artifact_name: absolute_path_str}.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}
    latest_entry = history[-1] if history else None

    # 0a. Benchmark lock artifacts (always regenerated)
    lock_paths = save_benchmark_lock_artifacts(data_dir=data_dir)
    paths.update(lock_paths)

    # 0b. Comparability report (generated before governance; used by governance report)
    comp_paths = save_comparability_artifacts(history, data_dir=data_dir)
    paths.update(comp_paths)
    # Also keep the comparability result in memory for the governance report
    from comparability_checker import check_history_comparability as _check_compat
    window_for_comp = history[-window_size:] if len(history) > window_size else history
    comparability_result = _check_compat(window_for_comp)

    # 0c. Campaign acceptance (for latest campaign)
    acceptance_result: dict[str, Any] | None = None
    if latest_entry is not None:
        # Determine comparability_ok for the latest campaign pair
        comp_ok_for_latest: bool | None = None
        if len(history) >= 2:
            from comparability_checker import check_pair_comparability as _check_pair
            pair_result = _check_pair(history[-2], history[-1])
            comp_ok_for_latest = pair_result.get("comparable", None)
        acceptance_paths = save_campaign_acceptance_artifacts(
            latest_entry,
            data_dir=data_dir,
            comparability_ok=comp_ok_for_latest,
        )
        paths.update(acceptance_paths)
        from decision_policy import check_campaign_acceptance as _check_accept
        acceptance_result = _check_accept(latest_entry, comparability_ok=comp_ok_for_latest)

    # 1. Trend summary
    trend = build_trend_summary(history, window_size=window_size)
    trend_json_path = data_dir / "trend_summary.json"
    _write_json(trend, trend_json_path)
    paths["trend_summary.json"] = str(trend_json_path)

    trend_md = build_trend_summary_md(trend)
    trend_md_path = data_dir / "trend_summary.md"
    trend_md_path.write_text(trend_md, encoding="utf-8")
    paths["trend_summary.md"] = str(trend_md_path)

    # 2. Regression alert
    alert = build_regression_alert(history, window_size=window_size)
    alert_json_path = data_dir / "regression_alert.json"
    _write_json(alert, alert_json_path)
    paths["regression_alert.json"] = str(alert_json_path)

    alert_md = build_regression_alert_md(alert)
    alert_md_path = data_dir / "regression_alert.md"
    alert_md_path.write_text(alert_md, encoding="utf-8")
    paths["regression_alert.md"] = str(alert_md_path)

    # 3. Promotion gate
    gate = build_promotion_gate(
        history,
        stability,
        latest_recommendation=latest_recommendation,
        regression_alert=alert,
    )
    gate_json_path = data_dir / "promotion_gate.json"
    _write_json(gate, gate_json_path)
    paths["promotion_gate.json"] = str(gate_json_path)

    gate_md = build_promotion_gate_md(gate)
    gate_md_path = data_dir / "promotion_gate.md"
    gate_md_path.write_text(gate_md, encoding="utf-8")
    paths["promotion_gate.md"] = str(gate_md_path)

    # 4. Governance report (includes comparability + acceptance sections)
    gov_md = build_governance_report(
        trend_summary=trend,
        regression_alert=alert,
        promotion_gate=gate,
        stability=stability,
        latest_entry=latest_entry,
        comparability_result=comparability_result,
        acceptance_result=acceptance_result,
    )
    gov_path = data_dir / "governance_report.md"
    gov_path.write_text(gov_md, encoding="utf-8")
    paths["governance_report.md"] = str(gov_path)

    return paths


def _write_json(payload: dict[str, Any], path: Path) -> None:
    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
