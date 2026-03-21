"""benchmark_registry.py

Defines named benchmark definitions for campaign comparability assessment.

A benchmark definition specifies the conditions under which two campaigns can
be meaningfully compared:
  - which profiles are valid for this benchmark
  - expected loto_types coverage
  - expected model variant set
  - minimum seed count / allowed seed sets
  - expected calibration method set
  - evaluation window and fold policy labels
  - data freshness policy

Benchmark resolution
--------------------
Each campaign profile maps to a benchmark name.  Two campaigns are in the
same benchmark family if their benchmark names are the same, or if one
benchmark is declared compatible with the other.

Available benchmarks
--------------------
  archcomp_lite  : Quick sanity-check benchmark (loto6 only, 2 seeds).
                   NOT suitable for promotion decisions.
  archcomp       : Standard benchmark (all loto_types, 3 seeds).
                   Default for campaign decisions.
  archcomp_full  : Extended benchmark (all loto_types, 5 seeds).
                   Use when signals are borderline.

Compatibility rules
-------------------
  archcomp and archcomp_full are declared mutually compatible because they
  share the same loto_types / variant / calibration configuration.  Seed-count
  differences between them are flagged as warnings rather than hard failures,
  allowing cross-benchmark trend analysis with appropriate caveats.

  archcomp_lite is NOT compatible with archcomp / archcomp_full because it
  covers only loto6 — comparing it against a multi-loto benchmark is
  misleading by design.
"""

from __future__ import annotations

BENCHMARK_REGISTRY_SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Benchmark definitions
# ---------------------------------------------------------------------------

BENCHMARK_DEFINITIONS: dict[str, dict] = {
    "archcomp_lite": {
        "benchmark_name": "archcomp_lite",
        "allowed_profiles": ["archcomp_lite"],
        "compatible_benchmarks": [],
        "expected_loto_types": ["loto6"],
        "expected_variants": ["legacy", "multihot", "deepsets", "settransformer"],
        "min_seed_count": 2,
        "allowed_seed_sets": [[42, 123]],
        "expected_calibration_methods": ["none", "temperature", "isotonic"],
        "evaluation_window_policy": "fast_preset",
        "fold_policy": "fast_preset",
        "data_freshness_policy": "same_fingerprint_family",
        "description": (
            "Quick 2-seed loto6-only sanity check.  "
            "NOT suitable for promotion decisions."
        ),
    },
    "archcomp": {
        "benchmark_name": "archcomp",
        "allowed_profiles": ["archcomp"],
        "compatible_benchmarks": ["archcomp_full"],
        "expected_loto_types": ["loto6", "loto7", "miniloto"],
        "expected_variants": ["legacy", "multihot", "deepsets", "settransformer"],
        "min_seed_count": 3,
        "allowed_seed_sets": [[42, 123, 456]],
        "expected_calibration_methods": ["none", "temperature", "isotonic"],
        "evaluation_window_policy": "archcomp_preset",
        "fold_policy": "archcomp_preset",
        "data_freshness_policy": "same_fingerprint_family",
        "description": (
            "Standard 3-seed cross-loto comparison.  "
            "Default for campaign decisions."
        ),
    },
    "archcomp_full": {
        "benchmark_name": "archcomp_full",
        "allowed_profiles": ["archcomp_full"],
        "compatible_benchmarks": ["archcomp"],
        "expected_loto_types": ["loto6", "loto7", "miniloto"],
        "expected_variants": ["legacy", "multihot", "deepsets", "settransformer"],
        "min_seed_count": 5,
        "allowed_seed_sets": [[42, 123, 456, 789, 999]],
        "expected_calibration_methods": ["none", "temperature", "isotonic"],
        "evaluation_window_policy": "default_preset",
        "fold_policy": "default_preset",
        "data_freshness_policy": "same_fingerprint_family",
        "description": (
            "Extended 5-seed full-epoch comparison.  "
            "Use when signals are borderline or run_more_seeds is flagged."
        ),
    },
}

# Map from profile_name → benchmark_name
_PROFILE_TO_BENCHMARK: dict[str, str] = {
    "archcomp_lite": "archcomp_lite",
    "archcomp": "archcomp",
    "archcomp_full": "archcomp_full",
}

VALID_BENCHMARK_NAMES: list[str] = sorted(BENCHMARK_DEFINITIONS.keys())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_benchmark_for_profile(profile_name: str) -> str:
    """Return the benchmark name for a given campaign profile name.

    Falls back to profile_name itself if not in the registry (forward-compat).
    """
    return _PROFILE_TO_BENCHMARK.get(profile_name, profile_name)


def get_benchmark(benchmark_name: str) -> dict:
    """Return a copy of the benchmark definition dict.

    Raises ValueError for unknown benchmark names.
    """
    if benchmark_name not in BENCHMARK_DEFINITIONS:
        raise ValueError(
            f"Unknown benchmark: {benchmark_name!r}. "
            f"Valid benchmarks: {VALID_BENCHMARK_NAMES}"
        )
    return dict(BENCHMARK_DEFINITIONS[benchmark_name])


def benchmarks_are_compatible(benchmark_a: str, benchmark_b: str) -> bool:
    """Return True if benchmark_a and benchmark_b are declared compatible.

    Compatibility is symmetric: if A lists B in compatible_benchmarks, the
    check also passes when called as benchmarks_are_compatible(B, A).
    """
    if benchmark_a == benchmark_b:
        return True
    defn_a = BENCHMARK_DEFINITIONS.get(benchmark_a) or {}
    if benchmark_b in (defn_a.get("compatible_benchmarks") or []):
        return True
    defn_b = BENCHMARK_DEFINITIONS.get(benchmark_b) or {}
    return benchmark_a in (defn_b.get("compatible_benchmarks") or [])


def validate_campaign_against_benchmark(
    entry: dict,
    benchmark_def: dict,
) -> list[str]:
    """Return a list of failure strings for a campaign against a benchmark.

    An empty list means the campaign satisfies all benchmark conditions.
    Used internally by comparability_checker but exposed here for testing.
    """
    failures: list[str] = []

    # loto_types
    expected_loto = set(benchmark_def.get("expected_loto_types") or [])
    actual_loto = set(entry.get("loto_types") or [])
    if expected_loto and actual_loto != expected_loto:
        failures.append(
            f"loto_types mismatch: expected {sorted(expected_loto)}, "
            f"got {sorted(actual_loto)}"
        )

    # seed count
    min_seeds = benchmark_def.get("min_seed_count") or 0
    actual_seeds = entry.get("seeds") or []
    if min_seeds and len(actual_seeds) < min_seeds:
        failures.append(
            f"seed_count too low: expected >= {min_seeds}, got {len(actual_seeds)}"
        )

    # variant set (from evaluation_model_variants field if available)
    expected_variants = set(benchmark_def.get("expected_variants") or [])
    emv = entry.get("evaluation_model_variants")
    if emv and expected_variants:
        actual_variants = set(v.strip() for v in emv.split(",") if v.strip())
        if actual_variants != expected_variants:
            failures.append(
                f"variant_set mismatch: expected {sorted(expected_variants)}, "
                f"got {sorted(actual_variants)}"
            )

    # calibration methods
    expected_cal = set(benchmark_def.get("expected_calibration_methods") or [])
    ecm = entry.get("evaluation_calibration_methods")
    if ecm and expected_cal:
        actual_cal = set(v.strip() for v in ecm.split(",") if v.strip())
        if actual_cal != expected_cal:
            failures.append(
                f"calibration_methods mismatch: expected {sorted(expected_cal)}, "
                f"got {sorted(actual_cal)}"
            )

    return failures


def list_benchmarks() -> None:
    """Print all benchmark definitions to stdout."""
    print("Available benchmark definitions:")
    print()
    for name in VALID_BENCHMARK_NAMES:
        b = BENCHMARK_DEFINITIONS[name]
        print(f"  {name}")
        print(f"    profiles:   {b['allowed_profiles']}")
        print(f"    loto_types: {b['expected_loto_types']}")
        print(f"    min_seeds:  {b['min_seed_count']}")
        print(f"    compatible: {b['compatible_benchmarks'] or '(none)'}")
        print(f"    {b['description']}")
        print()
