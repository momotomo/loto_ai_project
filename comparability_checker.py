"""comparability_checker.py

Checks whether campaign history entries are comparable — i.e. whether metric
differences between campaigns reflect genuine model quality differences rather
than differences in experimental conditions (profile, loto coverage, seeds,
calibration methods, or data).

Two campaigns are comparable when they share:
  - the same benchmark (or compatible benchmarks per benchmark_registry.py)
  - the same loto_types coverage
  - the same model variant set (evaluation_model_variants)
  - compatible calibration methods (evaluation_calibration_methods)
  - seed counts satisfying the benchmark minimum
  - the same data fingerprint family (if fingerprints are available)

If a pair fails any hard check, the pair is marked not-comparable (severity=error).
If only soft issues exist (different benchmark family within compatible set,
mismatched seed counts, or missing fingerprints), the pair is marked as
comparable-with-warnings (severity=warning).

Output artifacts
----------------
  data/comparability_report.json  — machine-readable comparability summary
  data/comparability_report.md    — human-readable comparability report

Schema
------
  COMPARABILITY_SCHEMA_VERSION = 1
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark_registry import (
    BENCHMARK_DEFINITIONS,
    benchmarks_are_compatible,
    resolve_benchmark_for_profile,
)

COMPARABILITY_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _variants_from_entry(entry: dict) -> set[str]:
    """Extract variant set from a campaign entry (evaluation_model_variants field)."""
    emv = entry.get("evaluation_model_variants")
    if emv:
        return set(v.strip() for v in emv.split(",") if v.strip())
    # Fall back to keys in variant_ranking_summary
    ranking = entry.get("variant_ranking_summary") or []
    return {r.get("variant") for r in ranking if r.get("variant")}


def _cal_methods_from_entry(entry: dict) -> set[str]:
    """Extract calibration methods set from a campaign entry."""
    ecm = entry.get("evaluation_calibration_methods")
    if ecm:
        return set(v.strip() for v in ecm.split(",") if v.strip())
    return set()


def _resolve_benchmark(entry: dict) -> str:
    """Return the benchmark name for a campaign entry."""
    bm = entry.get("benchmark_name")
    if bm:
        return bm
    return resolve_benchmark_for_profile(entry.get("profile_name", ""))


# ---------------------------------------------------------------------------
# Pair comparability check
# ---------------------------------------------------------------------------


def check_pair_comparability(
    entry_a: dict[str, Any],
    entry_b: dict[str, Any],
) -> dict[str, Any]:
    """Check if two campaign entries are comparable.

    Returns a dict with:
      - comparable: bool — True if comparable (possibly with warnings)
      - severity: 'ok' | 'warning' | 'error'
      - failed_checks: list[str] — hard failures that block comparability
      - warnings: list[str] — soft mismatches (comparable but with caveats)
      - rationale: str — human-readable summary
      - suggested_action: str — what to do about it
    """
    failed_checks: list[str] = []
    warnings: list[str] = []

    name_a = entry_a.get("campaign_name", "?")
    name_b = entry_b.get("campaign_name", "?")

    bm_a = _resolve_benchmark(entry_a)
    bm_b = _resolve_benchmark(entry_b)

    # --- Benchmark compatibility ---
    if not benchmarks_are_compatible(bm_a, bm_b):
        failed_checks.append(
            f"benchmark mismatch: {name_a!r} uses {bm_a!r}, "
            f"{name_b!r} uses {bm_b!r} (not compatible — see benchmark_registry.py)"
        )
    elif bm_a != bm_b:
        warnings.append(
            f"benchmark family match (compatible): {bm_a!r} ↔ {bm_b!r} — "
            "results may differ due to different seed counts or presets"
        )

    # --- Profile ---
    prof_a = entry_a.get("profile_name", "")
    prof_b = entry_b.get("profile_name", "")
    if prof_a != prof_b:
        warnings.append(
            f"profile mismatch: {prof_a!r} vs {prof_b!r} — "
            "same benchmark family but potentially different training preset"
        )

    # --- loto_types (hard check) ---
    loto_a = set(entry_a.get("loto_types") or [])
    loto_b = set(entry_b.get("loto_types") or [])
    if loto_a != loto_b:
        failed_checks.append(
            f"loto_type mismatch: {sorted(loto_a)} vs {sorted(loto_b)}"
        )

    # --- Variant set (hard check when both available) ---
    var_a = _variants_from_entry(entry_a)
    var_b = _variants_from_entry(entry_b)
    if var_a and var_b and var_a != var_b:
        failed_checks.append(
            f"variant_set mismatch: {sorted(var_a)} vs {sorted(var_b)}"
        )

    # --- Calibration methods (hard check when both available) ---
    cal_a = _cal_methods_from_entry(entry_a)
    cal_b = _cal_methods_from_entry(entry_b)
    if cal_a and cal_b and cal_a != cal_b:
        failed_checks.append(
            f"calibration_methods mismatch: {sorted(cal_a)} vs {sorted(cal_b)}"
        )

    # --- Seed count ---
    seeds_a = entry_a.get("seeds") or []
    seeds_b = entry_b.get("seeds") or []
    sc_a = len(seeds_a)
    sc_b = len(seeds_b)
    if sc_a != sc_b:
        warnings.append(
            f"seed_count mismatch: {sc_a} vs {sc_b} — "
            "variance estimates may not be directly comparable"
        )
    # Check against benchmark minimum
    defn = BENCHMARK_DEFINITIONS.get(bm_a) or {}
    min_seeds = defn.get("min_seed_count") or 0
    if min_seeds and (sc_a < min_seeds or sc_b < min_seeds):
        failed_checks.append(
            f"seed_count below benchmark {bm_a!r} minimum ({min_seeds}): "
            f"{name_a!r} has {sc_a}, {name_b!r} has {sc_b}"
        )

    # --- Data fingerprint (soft check) ---
    fp_a = entry_a.get("data_fingerprints") or {}
    fp_b = entry_b.get("data_fingerprints") or {}
    if fp_a and fp_b:
        common_loto = set(fp_a.keys()) & set(fp_b.keys())
        mismatched = sorted(lt for lt in common_loto if fp_a[lt] != fp_b[lt])
        if mismatched:
            warnings.append(
                f"data_fingerprint mismatch for loto_types: {mismatched} — "
                "campaigns used different data; metric differences may reflect "
                "data changes rather than model quality changes"
            )
    elif not fp_a and not fp_b:
        warnings.append(
            "data_fingerprints not recorded for either campaign — "
            "cannot verify data consistency (add fingerprints for stronger comparability)"
        )
    else:
        warnings.append(
            "data_fingerprints available for only one campaign — "
            "cannot verify data consistency"
        )

    # --- Determine overall status ---
    comparable = len(failed_checks) == 0
    if failed_checks:
        severity = "error"
    elif warnings:
        severity = "warning"
    else:
        severity = "ok"

    # Build human-readable rationale
    if comparable and severity == "ok":
        rationale = (
            f"{name_a!r} and {name_b!r} are fully comparable: "
            "same benchmark, loto_types, variants, and calibration methods."
        )
        suggested_action = "Proceed with metric comparison."
    elif comparable:
        rationale = (
            f"{name_a!r} and {name_b!r} are comparable with caveats: "
            + "; ".join(warnings[:2])
            + ("..." if len(warnings) > 2 else "")
        )
        suggested_action = (
            "Treat metric differences with caution — "
            "review warnings before drawing conclusions."
        )
    else:
        rationale = (
            f"{name_a!r} and {name_b!r} are NOT directly comparable: "
            + "; ".join(failed_checks[:2])
            + ("..." if len(failed_checks) > 2 else "")
        )
        suggested_action = (
            "Do not compare metrics directly. "
            "Run both campaigns under the same benchmark conditions before comparing."
        )

    return {
        "schema_version": COMPARABILITY_SCHEMA_VERSION,
        "campaign_a": name_a,
        "campaign_b": name_b,
        "benchmark_a": bm_a,
        "benchmark_b": bm_b,
        "comparable": comparable,
        "severity": severity,
        "failed_checks": failed_checks,
        "warnings": warnings,
        "rationale": rationale,
        "suggested_action": suggested_action,
    }


# ---------------------------------------------------------------------------
# History-wide comparability check
# ---------------------------------------------------------------------------


def check_history_comparability(
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Check comparability across consecutive campaign pairs in history.

    Returns:
      - overall_comparable: bool
      - overall_severity: 'ok' | 'warning' | 'error'
      - total_campaigns: int
      - comparable_pairs: int
      - warning_pairs: int
      - incomparable_pairs: int
      - pairs: list of pair check results
      - summary: str
    """
    if len(history) < 2:
        return {
            "schema_version": COMPARABILITY_SCHEMA_VERSION,
            "generated_at": _now_iso(),
            "total_campaigns": len(history),
            "comparable_pairs": 0,
            "warning_pairs": 0,
            "incomparable_pairs": 0,
            "overall_comparable": True,
            "overall_severity": "ok",
            "pairs": [],
            "summary": (
                "Fewer than 2 campaigns in history — "
                "comparability check not applicable."
            ),
        }

    pairs: list[dict] = []
    for i in range(len(history) - 1):
        result = check_pair_comparability(history[i], history[i + 1])
        pairs.append(result)

    comparable_count = sum(1 for p in pairs if p["comparable"])
    warning_count = sum(1 for p in pairs if p["severity"] == "warning")
    incomparable_count = len(pairs) - comparable_count

    overall_comparable = incomparable_count == 0
    if incomparable_count > 0:
        overall_severity = "error"
    elif warning_count > 0:
        overall_severity = "warning"
    else:
        overall_severity = "ok"

    if overall_severity == "ok":
        summary = (
            f"All {len(pairs)} consecutive campaign pair(s) are fully comparable."
        )
    elif overall_severity == "warning":
        summary = (
            f"{len(pairs)} pair(s) checked; all comparable but "
            f"{warning_count} pair(s) have warnings.  "
            "Review warnings before drawing trend conclusions."
        )
    else:
        summary = (
            f"{len(pairs)} pair(s) checked; {incomparable_count} pair(s) are "
            "NOT comparable.  "
            "Trend analysis and regression alerts may be unreliable."
        )

    return {
        "schema_version": COMPARABILITY_SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "total_campaigns": len(history),
        "comparable_pairs": comparable_count,
        "warning_pairs": warning_count,
        "incomparable_pairs": incomparable_count,
        "overall_comparable": overall_comparable,
        "overall_severity": overall_severity,
        "pairs": pairs,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def build_comparability_report_md(result: dict[str, Any]) -> str:
    """Build a Markdown comparability report from a check_history_comparability result."""
    lines: list[str] = []
    lines.append("# Comparability Report")
    lines.append("")
    lines.append(f"Generated: {result.get('generated_at', _now_iso())}")
    lines.append("")

    severity = result.get("overall_severity", "ok")
    status_label = {
        "ok": "✅ COMPARABLE",
        "warning": "⚠️ COMPARABLE WITH WARNINGS",
        "error": "❌ NOT COMPARABLE",
    }.get(severity, "? UNKNOWN")
    lines.append(f"## Overall Status: {status_label}")
    lines.append("")
    lines.append(result.get("summary", ""))
    lines.append("")

    # Summary table
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total campaigns | {result.get('total_campaigns', 0)} |")
    lines.append(f"| Fully comparable pairs | {result.get('comparable_pairs', 0)} |")
    lines.append(f"| Pairs with warnings | {result.get('warning_pairs', 0)} |")
    lines.append(f"| Incomparable pairs | {result.get('incomparable_pairs', 0)} |")
    lines.append("")

    # Per-pair details
    pairs = result.get("pairs") or []
    if pairs:
        lines.append("## Consecutive Pair Details")
        lines.append("")
        for pair in pairs:
            sev = pair.get("severity", "ok")
            emoji = {"ok": "✅", "warning": "⚠️", "error": "❌"}.get(sev, "?")
            lines.append(
                f"### {emoji} {pair.get('campaign_a', '?')} → {pair.get('campaign_b', '?')}"
            )
            lines.append("")
            comparable = pair.get("comparable")
            lines.append(f"- **Comparable**: {'Yes' if comparable else 'No'}")
            lines.append(f"- **Severity**: `{sev}`")
            lines.append(
                f"- **Benchmarks**: `{pair.get('benchmark_a', '?')}` → `{pair.get('benchmark_b', '?')}`"
            )
            lines.append("")

            failed = pair.get("failed_checks") or []
            if failed:
                lines.append("**Failed checks** (hard failures — blocks comparability):")
                for f in failed:
                    lines.append(f"  - ❌ {f}")
                lines.append("")

            warnings_list = pair.get("warnings") or []
            if warnings_list:
                lines.append("**Warnings** (soft — comparable with caveats):")
                for w in warnings_list:
                    lines.append(f"  - ⚠️ {w}")
                lines.append("")

            lines.append(f"**Rationale**: {pair.get('rationale', '')}")
            lines.append("")
            lines.append(f"**Suggested action**: {pair.get('suggested_action', '')}")
            lines.append("")

    lines.append("---")
    lines.append(
        f"*Comparability report generated by comparability_checker.py "
        f"v{COMPARABILITY_SCHEMA_VERSION}*"
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Save artifacts
# ---------------------------------------------------------------------------


def save_comparability_artifacts(
    history: list[dict[str, Any]],
    data_dir: str | Path = "data",
) -> dict[str, str]:
    """Save comparability report JSON + Markdown to data_dir.

    Returns dict of {artifact_name: absolute_path_str}.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    result = check_history_comparability(history)
    paths: dict[str, str] = {}

    json_path = data_dir / "comparability_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")
    paths["comparability_report.json"] = str(json_path)

    md_path = data_dir / "comparability_report.md"
    md_path.write_text(build_comparability_report_md(result), encoding="utf-8")
    paths["comparability_report.md"] = str(md_path)

    return paths
