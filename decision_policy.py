"""decision_policy.py

Decision Benchmark Policy for the loto-ai campaign infrastructure.

This module defines and enforces the conditions under which a campaign is
accepted as valid evidence for promotion decision-making.

Key concepts
------------
Comparable vs. Accepted
  - **Comparable** (comparability_checker.py): two campaigns measured the same
    way — same benchmark, loto coverage, variants, calibration, and data.
  - **Accepted for decision use**: the campaign additionally satisfies the
    stricter promotion-decision policy (archcomp / archcomp_full benchmark,
    full loto coverage, all four variants, all three calibration methods,
    comparability_ok = True, and archcomp_lite is explicitly excluded).

Why this distinction matters
  A comparable campaign is enough for trend analysis.
  An *accepted* campaign is required for promotion readiness counting.
  archcomp_lite campaigns may be comparable within themselves but are never
  accepted for promotion decisions because they cover only loto6.

Output artifacts (all written under data_dir)
  data/benchmark_lock.json          — current decision policy (machine-readable)
  data/benchmark_lock.md            — current decision policy (human-readable)
  data/campaign_acceptance.json     — whether the latest campaign is accepted
  data/campaign_acceptance.md       — human-readable acceptance verdict

Schema versions
---------------
  DECISION_POLICY_SCHEMA_VERSION = 1
  CAMPAIGN_ACCEPTANCE_SCHEMA_VERSION = 1
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

DECISION_POLICY_SCHEMA_VERSION = 1
CAMPAIGN_ACCEPTANCE_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Decision Benchmark Policy definition
# ---------------------------------------------------------------------------

#: The single authoritative decision policy for promotion-readiness counting.
#: All fields are human-readable and machine-checkable.
DECISION_BENCHMARK_POLICY: dict[str, Any] = {
    "schema_version": DECISION_POLICY_SCHEMA_VERSION,
    "policy_name": "standard_promotion_decision_policy",

    # Which benchmarks are valid for promotion decisions
    "active_decision_benchmarks": ["archcomp", "archcomp_full"],

    # Profiles that map to valid decision benchmarks
    "allowed_profiles": ["archcomp", "archcomp_full"],

    # archcomp_lite is explicitly excluded
    "excluded_profiles": ["archcomp_lite"],
    "excluded_profiles_reason": (
        "archcomp_lite covers loto6 only and uses a reduced seed count. "
        "It is intended for quick sanity checks, NOT for promotion decisions."
    ),

    # Required loto_types (must cover all three)
    "required_loto_types": ["loto6", "loto7", "miniloto"],

    # Required model variant set (must evaluate all four)
    "required_variants": ["legacy", "multihot", "deepsets", "settransformer"],

    # Required calibration methods (must evaluate all three)
    "required_calibration_methods": ["none", "temperature", "isotonic"],

    # Minimum seeds per profile
    "minimum_seed_policy": {
        "archcomp": 3,
        "archcomp_full": 5,
        "description": (
            "archcomp requires ≥3 seeds; archcomp_full requires ≥5 seeds. "
            "Fewer seeds yield unreliable variance estimates."
        ),
    },

    # Comparability is mandatory for acceptance
    "comparability_required": True,
    "comparability_required_reason": (
        "A campaign that is not comparable with its predecessors cannot contribute "
        "to a reliable trend signal. comparable=False is always a hard blocker."
    ),

    # Promotion review policy
    "promotion_review_policy": (
        "To enter promotion review, the operator must observe: "
        "(1) ≥2 consecutive accepted campaigns with recommend_next_action in "
        "{consider_promotion, run_more_seeds}, "
        "(2) promotion gate status = green, "
        "(3) manual per-loto evidence review confirms signal is genuine, "
        "(4) no active HIGH regression alert. "
        "Production is never changed automatically."
    ),

    # What 'accepted_only' counts mean
    "accepted_only_counting_policy": (
        "Stability metrics (consecutive_same_action, consecutive_positive_signal_*) "
        "are computed over ALL campaigns for trend awareness. "
        "The *_accepted_only variants count only campaigns where "
        "accepted_for_decision_use=True and counts_toward_promotion_readiness=True. "
        "Use the accepted_only counts for promotion readiness decisions."
    ),

    # PMA/ISAB pre-condition
    "pma_isab_precondition_policy": (
        "Before exploring PMA / ISAB or HPO: "
        "(1) ≥2 consecutive ACCEPTED campaigns with positive settransformer signal, "
        "(2) promotion gate is GREEN on an accepted campaign, "
        "(3) no active HIGH regression alert. "
        "Do NOT proceed based on archcomp_lite or non-accepted campaigns."
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _set_from_csv(csv_str: str | None) -> set[str]:
    if not csv_str:
        return set()
    return {v.strip() for v in csv_str.split(",") if v.strip()}


# ---------------------------------------------------------------------------
# Benchmark lock artifact
# ---------------------------------------------------------------------------


def build_benchmark_lock_artifact() -> dict[str, Any]:
    """Return the current benchmark lock as a JSON-serialisable dict."""
    policy = dict(DECISION_BENCHMARK_POLICY)
    policy["generated_at"] = _now_iso()
    # Enrich with benchmark definitions for the active benchmarks
    policy["active_benchmark_definitions"] = {
        name: dict(BENCHMARK_DEFINITIONS[name])
        for name in policy["active_decision_benchmarks"]
        if name in BENCHMARK_DEFINITIONS
    }
    return policy


def build_benchmark_lock_md(lock: dict[str, Any]) -> str:
    """Build a Markdown benchmark lock document."""
    lines: list[str] = []
    lines.append("# Decision Benchmark Policy (Benchmark Lock)")
    lines.append("")
    lines.append(f"Generated: {lock.get('generated_at', _now_iso())}")
    lines.append("")
    lines.append(
        "> **This document defines the conditions a campaign must satisfy to be**"
    )
    lines.append(
        "> **counted as valid evidence for promotion decision-making.**"
    )
    lines.append("")
    lines.append(
        "> Read `campaign_acceptance.md` to check if the latest campaign meets these conditions."
    )
    lines.append("")

    lines.append("## Active Decision Benchmarks")
    lines.append("")
    for bm in lock.get("active_decision_benchmarks") or []:
        lines.append(f"- **{bm}**")
    lines.append("")
    lines.append(
        f"Excluded (sanity only): `{'`, `'.join(lock.get('excluded_profiles') or [])}`"
    )
    lines.append(f"> {lock.get('excluded_profiles_reason', '')}")
    lines.append("")

    lines.append("## Required Conditions for Acceptance")
    lines.append("")
    lines.append("| Condition | Value |")
    lines.append("|-----------|-------|")
    lines.append(
        f"| allowed_profiles | `{'`, `'.join(lock.get('allowed_profiles') or [])}` |"
    )
    lines.append(
        f"| required_loto_types | `{'`, `'.join(lock.get('required_loto_types') or [])}` |"
    )
    lines.append(
        f"| required_variants | `{'`, `'.join(lock.get('required_variants') or [])}` |"
    )
    lines.append(
        f"| required_calibration_methods | `{'`, `'.join(lock.get('required_calibration_methods') or [])}` |"
    )
    lines.append(
        f"| comparability_required | `{lock.get('comparability_required', True)}` |"
    )
    min_seed = lock.get("minimum_seed_policy") or {}
    for profile, count in min_seed.items():
        if profile == "description":
            continue
        lines.append(f"| min_seeds ({profile}) | `{count}` |")
    lines.append("")

    lines.append("## Promotion Review Policy")
    lines.append("")
    lines.append(lock.get("promotion_review_policy", ""))
    lines.append("")

    lines.append("## Accepted-Only Counting Policy")
    lines.append("")
    lines.append(lock.get("accepted_only_counting_policy", ""))
    lines.append("")

    lines.append("## PMA / ISAB / HPO Pre-Condition")
    lines.append("")
    lines.append(lock.get("pma_isab_precondition_policy", ""))
    lines.append("")

    lines.append("## Active Benchmark Definitions")
    lines.append("")
    for name, defn in (lock.get("active_benchmark_definitions") or {}).items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"- Description: {defn.get('description', '')}")
        lines.append(f"- loto_types: `{defn.get('expected_loto_types', [])}`")
        lines.append(f"- min_seed_count: `{defn.get('min_seed_count', 0)}`")
        lines.append(f"- expected_variants: `{defn.get('expected_variants', [])}`")
        lines.append(
            f"- expected_calibration_methods: `{defn.get('expected_calibration_methods', [])}`"
        )
        lines.append("")

    lines.append("---")
    lines.append(
        f"*Decision benchmark policy generated by decision_policy.py v{DECISION_POLICY_SCHEMA_VERSION}*"
    )
    lines.append("")
    return "\n".join(lines)


def save_benchmark_lock_artifacts(
    data_dir: str | Path = "data",
) -> dict[str, str]:
    """Save benchmark_lock.json and benchmark_lock.md to data_dir.

    Returns dict of {artifact_name: absolute_path_str}.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    lock = build_benchmark_lock_artifact()
    paths: dict[str, str] = {}

    json_path = data_dir / "benchmark_lock.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(lock, f, indent=2, ensure_ascii=False)
        f.write("\n")
    paths["benchmark_lock.json"] = str(json_path)

    md_path = data_dir / "benchmark_lock.md"
    md_path.write_text(build_benchmark_lock_md(lock), encoding="utf-8")
    paths["benchmark_lock.md"] = str(md_path)

    return paths


# ---------------------------------------------------------------------------
# Campaign acceptance check
# ---------------------------------------------------------------------------


def check_campaign_acceptance(
    entry: dict[str, Any],
    *,
    comparability_ok: bool | None = None,
) -> dict[str, Any]:
    """Check whether a campaign entry is accepted for decision use.

    Parameters
    ----------
    entry:
        A campaign history entry (as built by campaign_manager.build_campaign_entry).
    comparability_ok:
        Override the comparability status.  If None, the function checks
        ``entry.get("comparability_ok")`` or defaults to True (unknown = not a
        hard failure for acceptance — but will raise a warning).

    Returns
    -------
    dict with:
      - accepted_for_decision_use: bool
      - counts_toward_promotion_readiness: bool (same as accepted for now)
      - decision_benchmark_name: str (the benchmark name of this campaign)
      - benchmark_name: str (alias)
      - profile_name: str
      - failed_requirements: list[str]  — hard failures (blocks acceptance)
      - warnings: list[str]             — soft issues (does not block)
      - rationale: str
      - can_count_toward_promotion_readiness: bool (alias for counts_toward_*)
    """
    policy = DECISION_BENCHMARK_POLICY
    failed: list[str] = []
    warnings_list: list[str] = []

    profile_name = entry.get("profile_name", "")
    benchmark_name = entry.get("benchmark_name") or resolve_benchmark_for_profile(profile_name)
    campaign_name = entry.get("campaign_name", "?")

    # --- 1. Profile / benchmark must be in allowed list ---
    allowed_profiles = set(policy.get("allowed_profiles") or [])
    excluded_profiles = set(policy.get("excluded_profiles") or [])

    if profile_name in excluded_profiles:
        failed.append(
            f"profile `{profile_name}` is excluded from decision use: "
            + policy.get("excluded_profiles_reason", "not a decision benchmark")
        )
    elif profile_name and profile_name not in allowed_profiles:
        failed.append(
            f"profile `{profile_name}` is not in allowed_profiles "
            f"({sorted(allowed_profiles)})"
        )

    # --- 2. Benchmark must be one of the active decision benchmarks ---
    active_bms = set(policy.get("active_decision_benchmarks") or [])
    # Check benchmark compatibility: a campaign's benchmark must be in active set
    # or compatible with one of them.  For now we require exact membership.
    if benchmark_name not in active_bms and benchmark_name not in excluded_profiles:
        # Could be a forward-compat unknown benchmark
        failed.append(
            f"benchmark `{benchmark_name}` is not an active decision benchmark "
            f"({sorted(active_bms)})"
        )

    # --- 3. loto_types must match required set ---
    required_loto = set(policy.get("required_loto_types") or [])
    actual_loto = set(entry.get("loto_types") or [])
    if required_loto and actual_loto != required_loto:
        missing_loto = required_loto - actual_loto
        extra_loto = actual_loto - required_loto
        msg = f"loto_types mismatch: required {sorted(required_loto)}, got {sorted(actual_loto)}"
        if missing_loto:
            msg += f" (missing: {sorted(missing_loto)})"
        if extra_loto:
            msg += f" (extra: {sorted(extra_loto)})"
        failed.append(msg)

    # --- 4. Variants must match required set ---
    required_variants = set(policy.get("required_variants") or [])
    actual_variants = _set_from_csv(entry.get("evaluation_model_variants"))
    if required_variants and actual_variants:
        if actual_variants != required_variants:
            failed.append(
                f"variant_set mismatch: required {sorted(required_variants)}, "
                f"got {sorted(actual_variants)}"
            )
    elif required_variants and not actual_variants:
        warnings_list.append(
            "evaluation_model_variants not recorded in campaign entry — "
            "cannot verify variant set"
        )

    # --- 5. Calibration methods must match required set ---
    required_cal = set(policy.get("required_calibration_methods") or [])
    actual_cal = _set_from_csv(entry.get("evaluation_calibration_methods"))
    if required_cal and actual_cal:
        if actual_cal != required_cal:
            failed.append(
                f"calibration_methods mismatch: required {sorted(required_cal)}, "
                f"got {sorted(actual_cal)}"
            )
    elif required_cal and not actual_cal:
        warnings_list.append(
            "evaluation_calibration_methods not recorded in campaign entry — "
            "cannot verify calibration methods"
        )

    # --- 6. Seed count must meet benchmark minimum ---
    min_seed_policy = policy.get("minimum_seed_policy") or {}
    min_seeds_for_profile = min_seed_policy.get(profile_name)
    if min_seeds_for_profile is None and benchmark_name in BENCHMARK_DEFINITIONS:
        min_seeds_for_profile = BENCHMARK_DEFINITIONS[benchmark_name].get("min_seed_count")
    actual_seeds = entry.get("seeds") or []
    if min_seeds_for_profile and len(actual_seeds) < min_seeds_for_profile:
        failed.append(
            f"seed_count too low for `{profile_name}`: "
            f"required ≥{min_seeds_for_profile}, got {len(actual_seeds)}"
        )

    # --- 7. Comparability must be ok ---
    if comparability_ok is None:
        # Try to read from entry; default to None = unknown
        comparability_ok = entry.get("comparability_ok")

    if comparability_ok is False:
        failed.append(
            "comparability_ok=False: this campaign is not comparable with its predecessor. "
            "Cannot use as decision evidence."
        )
    elif comparability_ok is None:
        warnings_list.append(
            "comparability_ok not determined — comparability check not yet run. "
            "Run after a second campaign to verify."
        )

    # --- Determine acceptance ---
    accepted = len(failed) == 0
    counts_toward = accepted  # For now: accepted = counts toward

    # Build rationale
    if accepted and not warnings_list:
        rationale = (
            f"Campaign `{campaign_name}` meets all decision benchmark policy requirements. "
            f"Benchmark: `{benchmark_name}`. "
            "This campaign counts toward promotion readiness."
        )
    elif accepted:
        rationale = (
            f"Campaign `{campaign_name}` is accepted for decision use with caveats. "
            f"Benchmark: `{benchmark_name}`. "
            "Review warnings before using for promotion readiness."
        )
    else:
        rationale = (
            f"Campaign `{campaign_name}` is NOT accepted for decision use. "
            f"Benchmark: `{benchmark_name}`. "
            f"{len(failed)} requirement(s) failed. "
            "This campaign is excluded from promotion readiness counting."
        )

    return {
        "schema_version": CAMPAIGN_ACCEPTANCE_SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "campaign_name": campaign_name,
        "profile_name": profile_name,
        "benchmark_name": benchmark_name,
        "decision_benchmark_name": benchmark_name,
        "accepted_for_decision_use": accepted,
        "counts_toward_promotion_readiness": counts_toward,
        "can_count_toward_promotion_readiness": counts_toward,
        "failed_requirements": failed,
        "warnings": warnings_list,
        "rationale": rationale,
    }


def build_campaign_acceptance_md(result: dict[str, Any]) -> str:
    """Build a Markdown campaign acceptance document."""
    accepted = result.get("accepted_for_decision_use", False)
    counts = result.get("counts_toward_promotion_readiness", False)
    failed = result.get("failed_requirements") or []
    warnings_list = result.get("warnings") or []

    status_emoji = "✅" if accepted else "❌"
    status_label = "ACCEPTED" if accepted else "NOT ACCEPTED"

    lines: list[str] = []
    lines.append("# Campaign Acceptance Report")
    lines.append("")
    lines.append(f"Generated: {result.get('generated_at', _now_iso())}")
    lines.append("")
    lines.append(f"Campaign: **{result.get('campaign_name', '?')}**")
    lines.append("")
    lines.append(f"## Acceptance Status: {status_emoji} {status_label}")
    lines.append("")
    lines.append(result.get("rationale", ""))
    lines.append("")

    lines.append("| Field | Value |")
    lines.append("|-------|-------|")
    lines.append(f"| profile_name | `{result.get('profile_name', '?')}` |")
    lines.append(f"| benchmark_name | `{result.get('benchmark_name', '?')}` |")
    lines.append(
        f"| accepted_for_decision_use | `{result.get('accepted_for_decision_use')}` |"
    )
    lines.append(
        f"| counts_toward_promotion_readiness | `{result.get('counts_toward_promotion_readiness')}` |"
    )
    lines.append("")

    if failed:
        lines.append("## Failed Requirements ❌")
        lines.append("")
        lines.append(
            "> These failures prevent this campaign from being used as decision evidence."
        )
        lines.append("")
        for f in failed:
            lines.append(f"- ❌ {f}")
        lines.append("")

    if warnings_list:
        lines.append("## Warnings ⚠️")
        lines.append("")
        for w in warnings_list:
            lines.append(f"- ⚠️ {w}")
        lines.append("")

    if accepted:
        lines.append(
            "> ✅ **This campaign counts toward promotion readiness.** "
            "See `governance_report.md` for the full promotion gate status."
        )
    else:
        lines.append(
            "> ❌ **This campaign does NOT count toward promotion readiness.** "
            "It may still be useful for sanity checking but should not influence "
            "promotion decisions. See `benchmark_lock.md` for required conditions."
        )
    lines.append("")

    lines.append("---")
    lines.append(
        f"*Campaign acceptance generated by decision_policy.py v{CAMPAIGN_ACCEPTANCE_SCHEMA_VERSION}*"
    )
    lines.append("")
    return "\n".join(lines)


def save_campaign_acceptance_artifacts(
    entry: dict[str, Any],
    data_dir: str | Path = "data",
    *,
    comparability_ok: bool | None = None,
) -> dict[str, str]:
    """Save campaign_acceptance.json and campaign_acceptance.md to data_dir.

    Returns dict of {artifact_name: absolute_path_str}.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    result = check_campaign_acceptance(entry, comparability_ok=comparability_ok)
    paths: dict[str, str] = {}

    json_path = data_dir / "campaign_acceptance.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")
    paths["campaign_acceptance.json"] = str(json_path)

    md_path = data_dir / "campaign_acceptance.md"
    md_path.write_text(build_campaign_acceptance_md(result), encoding="utf-8")
    paths["campaign_acceptance.md"] = str(md_path)

    return paths
