"""scripts/run_campaign.py

Run a named comparison campaign using a predefined campaign profile.

A campaign bundles a cross-loto comparison into a named, versioned directory
with full artifact preservation and history tracking.  This is the recommended
entry point for ongoing comparison campaigns — prefer this over ad-hoc calls
to run_cross_loto.py when you want to track changes over time.

Available profiles
------------------
  archcomp_lite : Fast 2-seed single-loto (loto6) validation.
  archcomp      : Standard 3-seed all-loto comparison (DEFAULT).
  archcomp_full : Extended 5-seed full-preset — use when run_more_seeds persists.

Usage examples
--------------
List available profiles:

    python scripts/run_campaign.py --list_profiles

Run a named campaign with the default archcomp profile:

    python scripts/run_campaign.py --campaign_name 2026-03-21_archcomp

Run with an explicit profile:

    python scripts/run_campaign.py \\
        --campaign_name 2026-03-21_archcomp_full \\
        --profile archcomp_full

Override specific profile settings (e.g. test with fewer seeds):

    python scripts/run_campaign.py \\
        --campaign_name 2026-03-21_quick \\
        --profile archcomp_lite \\
        --seeds 42,123

Skip training and re-aggregate from existing comparison_summary files:

    python scripts/run_campaign.py \\
        --campaign_name 2026-03-21_archcomp \\
        --profile archcomp \\
        --skip_training

Campaign artifacts
------------------
campaigns/<campaign_name>/
    campaign_metadata.json         — profile, seeds, loto_types, timing
    cross_loto_summary.json        — full cross-loto summary
    recommendation.json            — next-action recommendation
    cross_loto_report.md           — human-readable evidence pack (READ THIS)
    variant_metrics.csv            — per-variant metrics table
    pairwise_comparisons.csv       — pairwise test results table
    recommendation_summary.csv     — one-row recommendation summary
    comparison_summary_{loto_type}.json   — per-loto aggregated summary

History artifacts (updated in data/ after each campaign)
---------------------------------------------------------
data/campaign_history.json     — all campaigns + recommendation stability
data/campaign_history.csv      — tabular history for spreadsheet analysis
data/campaign_diff_report.md   — diff vs previous campaign (READ THIS FIRST)

Reading order
-------------
1. data/campaign_diff_report.md          — what changed since last time?
2. data/campaign_history.json            — stability trend across campaigns
3. campaigns/<name>/cross_loto_report.md — detailed evidence pack

Notes
-----
- Production artifacts are NOT updated by campaign runs (skip_final_train=True).
- campaign_name must be unique; existing directories are not overwritten.
- History is persisted in data/campaign_history.json across campaign runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from campaign_manager import (  # noqa: E402
    append_campaign_to_history,
    build_campaign_entry,
    load_campaign_history,
    save_campaign_artifacts,
)
from campaign_profiles import (  # noqa: E402
    VALID_PROFILE_NAMES,
    list_profiles,
    resolve_profile,
)
from comparison_summary import (  # noqa: E402
    build_comparison_summary,
    get_comparison_summary_path,
    load_eval_report,
    save_comparison_summary,
)
from cross_loto_summary import (  # noqa: E402
    build_cross_loto_summary,
    build_recommendation,
    save_json,
)
from cross_loto_report import (  # noqa: E402
    build_run_metadata,
    save_report_artifacts,
)
from experiment_runner import execute_experiment, resolve_experiment_config  # noqa: E402
from config import LOTO_CONFIG  # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a named cross-loto comparison campaign using a predefined profile."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
profiles:
  archcomp_lite  fast 2-seed single-loto validation
  archcomp       standard 3-seed all-loto comparison (DEFAULT)
  archcomp_full  extended 5-seed full-preset comparison

examples:
  python scripts/run_campaign.py --list_profiles
  python scripts/run_campaign.py --campaign_name 2026-03-21_archcomp
  python scripts/run_campaign.py --campaign_name 2026-03-21_full --profile archcomp_full
  python scripts/run_campaign.py --campaign_name 2026-03-21_archcomp --skip_training
        """,
    )
    parser.add_argument(
        "--campaign_name",
        help=(
            "Unique name for this campaign run (e.g. '2026-03-21_archcomp').  "
            "Required unless --list_profiles is used."
        ),
    )
    parser.add_argument(
        "--profile",
        choices=VALID_PROFILE_NAMES,
        default="archcomp",
        help=f"Campaign profile to use (default: archcomp). Choices: {VALID_PROFILE_NAMES}",
    )
    parser.add_argument(
        "--list_profiles",
        action="store_true",
        default=False,
        help="List available campaign profiles and exit.",
    )

    # Optional per-run overrides (override profile values)
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seeds (overrides profile). Example: 42,123,456",
    )
    parser.add_argument(
        "--loto_types",
        default=None,
        help=(
            "Comma-separated loto_types (overrides profile). "
            f"Valid: {','.join(sorted(LOTO_CONFIG.keys()))}"
        ),
    )
    parser.add_argument(
        "--evaluation_model_variants",
        default=None,
        help="Comma-separated model variants to evaluate (overrides profile).",
    )

    # Infrastructure options
    parser.add_argument(
        "--campaigns_dir",
        default="campaigns",
        help="Root directory for campaign subdirectories (default: campaigns).",
    )
    parser.add_argument(
        "--data_dir",
        default="data",
        help="Directory for shared history artifacts (default: data).",
    )
    parser.add_argument(
        "--run_root",
        default="runs",
        help="Directory for per-seed run subdirectories (default: runs).",
    )
    parser.add_argument(
        "--repo_root",
        default=str(REPO_ROOT),
        help="Repository root (default: parent of scripts/).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Statistical significance threshold (default: 0.05).",
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        default=False,
        help=(
            "Skip model training; load existing comparison_summary_{loto_type}.json "
            "files and aggregate into cross-loto summary."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Seed / loto_type parsing
# ---------------------------------------------------------------------------


def _parse_seeds(seeds_str: str) -> list[int]:
    seeds = []
    for chunk in seeds_str.split(","):
        chunk = chunk.strip()
        if chunk:
            try:
                seeds.append(int(chunk))
            except ValueError:
                raise SystemExit(f"Invalid seed value: {chunk!r}")
    if not seeds:
        raise SystemExit("At least one seed is required.")
    return seeds


def _parse_loto_types(loto_types_str: str) -> list[str]:
    valid = set(LOTO_CONFIG.keys())
    types = []
    for chunk in loto_types_str.split(","):
        chunk = chunk.strip()
        if chunk:
            if chunk not in valid:
                raise SystemExit(
                    f"Invalid loto_type: {chunk!r}. Valid choices: {sorted(valid)}"
                )
            types.append(chunk)
    if not types:
        raise SystemExit("At least one loto_type is required.")
    return types


# ---------------------------------------------------------------------------
# Per-seed run helpers (mirrors run_cross_loto logic)
# ---------------------------------------------------------------------------


def _run_single_seed(
    *,
    loto_type: str,
    preset: str,
    seed: int,
    evaluation_model_variants: str,
    saved_calibration_method: str,
    evaluation_calibration_methods: str,
    repo_root: str,
    run_root: str,
) -> Path:
    config = {
        "loto_type": loto_type,
        "preset": preset,
        "seed": seed,
        "model_variant": "legacy",
        "evaluation_model_variants": evaluation_model_variants,
        "saved_calibration_method": saved_calibration_method,
        "evaluation_calibration_methods": evaluation_calibration_methods,
        "skip_final_train": True,
        "skip_legacy_holdout": False,
    }
    resolved_config = resolve_experiment_config(config)
    run_dir, summary = execute_experiment(
        resolved_config=resolved_config,
        repo_root=repo_root,
        run_root=run_root,
        requested_config=config,
    )
    if summary["status"] != "succeeded":
        raise SystemExit(
            f"Seed {seed} run failed for {loto_type}: {summary['status']}"
        )
    print(f"      Seed {seed}: run_dir={run_dir}")
    return run_dir


def _load_eval_report_from_run(run_dir: Path, loto_type: str) -> dict:
    report_path = run_dir / "artifacts" / "data" / f"eval_report_{loto_type}.json"
    if not report_path.exists():
        raise FileNotFoundError(f"eval_report not found: {report_path}")
    return load_eval_report(report_path)


def _run_loto_type_comparison(
    *,
    loto_type: str,
    preset: str,
    seeds: list[int],
    evaluation_model_variants: str,
    saved_calibration_method: str,
    evaluation_calibration_methods: str,
    repo_root: str,
    run_root: str,
    alpha: float,
    campaign_dir: Path,
) -> dict:
    """Run multi-seed comparison for one loto_type and return its comparison_summary."""
    print(f"\n    --- {loto_type}: running {len(seeds)} seed(s) ---")
    run_dirs: list[Path] = []
    for seed in seeds:
        run_dir = _run_single_seed(
            loto_type=loto_type,
            preset=preset,
            seed=seed,
            evaluation_model_variants=evaluation_model_variants,
            saved_calibration_method=saved_calibration_method,
            evaluation_calibration_methods=evaluation_calibration_methods,
            repo_root=repo_root,
            run_root=run_root,
        )
        run_dirs.append(run_dir)

    eval_reports: list[dict] = []
    for run_dir, seed in zip(run_dirs, seeds):
        try:
            report = _load_eval_report_from_run(run_dir, loto_type)
            eval_reports.append(report)
            print(f"      Loaded eval_report for seed {seed}: {run_dir.name}")
        except FileNotFoundError as exc:
            print(f"      WARNING: {exc} — skipping seed {seed}.")

    if not eval_reports:
        raise SystemExit(
            f"No eval_reports could be loaded for {loto_type}."
        )

    summary = build_comparison_summary(
        eval_reports=eval_reports,
        loto_type=loto_type,
        preset=preset,
        seeds=seeds,
        alpha=alpha,
    )
    # Save to campaign dir
    output_path = campaign_dir / f"comparison_summary_{loto_type}.json"
    save_comparison_summary(summary, output_path)
    print(f"      Per-loto summary saved: {output_path.name}")
    return summary


def _load_existing_comparison_summary(loto_type: str, campaign_dir: Path) -> dict | None:
    """Load existing comparison_summary for a loto_type from campaign dir, or data/."""
    # Try campaign dir first, then fall back to data/
    candidates = [
        campaign_dir / f"comparison_summary_{loto_type}.json",
        REPO_ROOT / "data" / f"comparison_summary_{loto_type}.json",
    ]
    for path in candidates:
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as exc:
                print(f"    WARNING: Failed to load {path}: {exc}")
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    if args.list_profiles:
        list_profiles()
        return

    if not args.campaign_name:
        raise SystemExit(
            "Error: --campaign_name is required (or use --list_profiles to see options)."
        )

    repo_root = str(Path(args.repo_root).resolve())

    # Resolve paths
    campaigns_dir = (
        Path(args.campaigns_dir)
        if Path(args.campaigns_dir).is_absolute()
        else Path(repo_root) / args.campaigns_dir
    )
    data_dir = (
        Path(args.data_dir)
        if Path(args.data_dir).is_absolute()
        else Path(repo_root) / args.data_dir
    )

    campaign_dir = campaigns_dir / args.campaign_name
    if campaign_dir.exists():
        raise SystemExit(
            f"Campaign directory already exists: {campaign_dir}\n"
            "Use a unique --campaign_name to avoid overwriting previous results."
        )
    campaign_dir.mkdir(parents=True, exist_ok=True)

    # Resolve profile + overrides
    overrides: dict = {}
    if args.seeds is not None:
        overrides["seeds"] = _parse_seeds(args.seeds)
    if args.loto_types is not None:
        overrides["loto_types"] = _parse_loto_types(args.loto_types)
    if args.evaluation_model_variants is not None:
        overrides["evaluation_model_variants"] = args.evaluation_model_variants

    profile = resolve_profile(args.profile, overrides or None)

    seeds: list[int] = profile["seeds"]
    loto_types: list[str] = profile["loto_types"]
    preset: str = profile["preset"]
    evaluation_model_variants: str = profile["evaluation_model_variants"]
    saved_calibration_method: str = profile["saved_calibration_method"]
    evaluation_calibration_methods: str = profile["evaluation_calibration_methods"]

    started_at = datetime.now(timezone.utc).isoformat()

    print(f"\n{'='*60}")
    print(f"Campaign: {args.campaign_name}")
    print(f"Profile:  {args.profile}")
    print(f"Preset:   {preset}")
    print(f"Seeds:    {seeds}")
    print(f"Loto:     {loto_types}")
    print(f"Variants: {evaluation_model_variants}")
    print(f"Dir:      {campaign_dir}")
    print(f"{'='*60}\n")

    # Run or load per-loto summaries
    per_loto_summaries: dict[str, dict] = {}
    source_summary_paths: dict[str, str] = {}

    if args.skip_training:
        print("--- Skipping training: loading existing comparison_summary files ---")
        for loto_type in loto_types:
            summary = _load_existing_comparison_summary(loto_type, campaign_dir)
            if summary is not None:
                per_loto_summaries[loto_type] = summary
                print(
                    f"  Loaded summary for {loto_type}: "
                    f"run_count={summary.get('run_count')}"
                )
            else:
                print(f"  WARNING: No comparison_summary found for {loto_type} — skipping.")
    else:
        for loto_type in loto_types:
            try:
                summary = _run_loto_type_comparison(
                    loto_type=loto_type,
                    preset=preset,
                    seeds=seeds,
                    evaluation_model_variants=evaluation_model_variants,
                    saved_calibration_method=saved_calibration_method,
                    evaluation_calibration_methods=evaluation_calibration_methods,
                    repo_root=repo_root,
                    run_root=args.run_root,
                    alpha=args.alpha,
                    campaign_dir=campaign_dir,
                )
                per_loto_summaries[loto_type] = summary
                source_summary_paths[loto_type] = str(
                    campaign_dir / f"comparison_summary_{loto_type}.json"
                )
            except SystemExit as exc:
                print(f"\n  ERROR for {loto_type}: {exc}. Continuing.")

    if not per_loto_summaries:
        raise SystemExit("No per-loto summaries available. Cannot build cross-loto summary.")

    # Build cross-loto summary
    print("\n--- Building cross-loto summary ---")
    cross_summary = build_cross_loto_summary(
        per_loto_summaries=per_loto_summaries,
        loto_types=list(per_loto_summaries.keys()),
        preset=preset,
        seeds=seeds,
        alpha=args.alpha,
    )
    save_json(cross_summary, campaign_dir / "cross_loto_summary.json")
    print(f"  Cross-loto summary saved: {campaign_dir / 'cross_loto_summary.json'}")

    # Build recommendation
    recommendation = build_recommendation(cross_summary)
    save_json(recommendation, campaign_dir / "recommendation.json")
    print(f"  Recommendation saved: {campaign_dir / 'recommendation.json'}")

    # Build run metadata
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{args.campaign_name}"
    run_metadata = build_run_metadata(
        loto_types=list(per_loto_summaries.keys()),
        preset=preset,
        seeds=seeds,
        evaluation_model_variants=evaluation_model_variants,
        alpha=args.alpha,
        source_summary_paths=source_summary_paths,
        run_id=run_id,
    )

    # Save evidence pack to campaign dir
    print("\n--- Generating evidence pack ---")
    saved_paths = save_report_artifacts(
        cross_summary,
        recommendation,
        data_dir=campaign_dir,
        run_metadata=run_metadata,
    )
    for name, path in saved_paths.items():
        print(f"  {name}: {path}")

    finished_at = datetime.now(timezone.utc).isoformat()

    # Save campaign_metadata.json
    campaign_metadata = {
        "campaign_name": args.campaign_name,
        "profile_name": args.profile,
        "profile": profile,
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "campaign_dir": str(campaign_dir),
        "data_dir": str(data_dir),
        "loto_types": loto_types,
        "seeds": seeds,
        "preset": preset,
        "evaluation_model_variants": evaluation_model_variants,
        "alpha": args.alpha,
        "skip_training": args.skip_training,
        "source_summary_paths": source_summary_paths,
    }
    save_json(campaign_metadata, campaign_dir / "campaign_metadata.json")
    print(f"\n  Campaign metadata: {campaign_dir / 'campaign_metadata.json'}")

    # Update campaign history
    print("\n--- Updating campaign history ---")
    history = load_campaign_history(data_dir)
    entry = build_campaign_entry(
        args.campaign_name,
        args.profile,
        cross_summary,
        recommendation,
        started_at=started_at,
        finished_at=finished_at,
        campaign_dir=str(campaign_dir),
    )
    history = append_campaign_to_history(history, entry)
    history_paths = save_campaign_artifacts(history, data_dir=data_dir)
    for name, path in history_paths.items():
        print(f"  {name}: {path}")

    # Human-readable summary
    rec = recommendation
    print(f"\n{'='*60}")
    print(f"Campaign {args.campaign_name} complete")
    print(f"  recommended_next_action: {rec.get('recommended_next_action')}")
    print(f"  recommended_challenger:  {rec.get('recommended_challenger')}")
    print(f"  keep_production_as_is:   {rec.get('keep_production_as_is')}")
    print(f"  pma_or_isab_next:        {rec.get('whether_to_try_pma_or_isab_next')}")
    for b in rec.get("blockers_to_promotion") or []:
        print(f"  blocker: {b}")

    print(f"\n=== Read first ===")
    diff_path = history_paths.get("campaign_diff_report.md")
    if diff_path:
        print(f"  Diff report:     {diff_path}")
    print(f"  Evidence pack:   {campaign_dir / 'cross_loto_report.md'}")
    print(f"  History:         {history_paths.get('campaign_history.json')}")
    print()


if __name__ == "__main__":
    main()
