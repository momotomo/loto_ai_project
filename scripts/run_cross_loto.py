"""scripts/run_cross_loto.py

Run architecture comparison across multiple loto_types and aggregate results
into a cross-loto summary artifact (data/cross_loto_summary.json) and a
recommendation artifact (data/recommendation.json).

Each loto_type is compared independently with multiple seeds using the same
logic as run_multi_seed.py.  Results are then combined into a single
cross-loto view that makes it easy to see which variant performs best across
the full range of lottery games.

Typical usage:

    python scripts/run_cross_loto.py \\
        --loto_types loto6,loto7,miniloto \\
        --preset archcomp \\
        --seeds 42,123,456 \\
        --evaluation_model_variants legacy,multihot,deepsets,settransformer \\
        --run_root runs

Minimal cross-loto smoke (reuses existing comparison_summary if available):

    python scripts/run_cross_loto.py \\
        --loto_types loto6 \\
        --preset smoke \\
        --seeds 42 \\
        --skip_training \\
        --run_root runs

Notes
-----
- Each loto_type's per-seed runs are saved under --run_root independently.
- Per-loto comparison summaries are written to
  data/comparison_summary_{loto_type}.json (same as run_multi_seed.py).
- The cross-loto summary is written to data/cross_loto_summary.json.
- The recommendation artifact is written to data/recommendation.json.
- Production artifacts are NOT updated unless --model_variant is set and
  --no_skip_final_train is passed explicitly.
- Use --skip_training to skip all model runs and only (re-)build the
  cross-loto summary and recommendation from existing comparison_summary files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from comparison_summary import (  # noqa: E402
    build_comparison_summary,
    get_comparison_summary_path,
    load_eval_report,
    save_comparison_summary,
)
from cross_loto_summary import (  # noqa: E402
    build_cross_loto_summary,
    build_recommendation,
    get_cross_loto_summary_path,
    get_recommendation_path,
    load_comparison_summary,
    save_json,
)
from experiment_runner import execute_experiment, resolve_experiment_config  # noqa: E402
from model_variants import MODEL_VARIANT_CHOICES  # noqa: E402
from config import LOTO_CONFIG  # noqa: E402

VALID_PRESETS = ["default", "fast", "smoke", "archcomp"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run architecture comparison across multiple loto_types and build "
            "a cross-loto summary and recommendation artifact."
        ),
    )
    parser.add_argument(
        "--loto_types",
        default=",".join(sorted(LOTO_CONFIG.keys())),
        help=(
            "Comma-separated list of loto_types to compare "
            f"(default: all — {','.join(sorted(LOTO_CONFIG.keys()))})"
        ),
    )
    parser.add_argument(
        "--preset",
        choices=VALID_PRESETS,
        default="archcomp",
        help="Preset to use for each seed run (default: archcomp)",
    )
    parser.add_argument(
        "--seeds",
        default="42,123,456",
        help="Comma-separated list of random seeds (default: 42,123,456)",
    )
    parser.add_argument(
        "--model_variant",
        choices=sorted(MODEL_VARIANT_CHOICES),
        default="legacy",
        help="Production model variant for final artifact (default: legacy)",
    )
    parser.add_argument(
        "--evaluation_model_variants",
        default="legacy,multihot,deepsets,settransformer",
        help=(
            "Comma-separated variants to evaluate "
            "(default: legacy,multihot,deepsets,settransformer)"
        ),
    )
    parser.add_argument(
        "--saved_calibration_method",
        default="none",
        help="Calibration method to save in the production artifact (default: none)",
    )
    parser.add_argument(
        "--evaluation_calibration_methods",
        default="none,temperature,isotonic",
        help=(
            "Comma-separated calibration methods to evaluate "
            "(default: none,temperature,isotonic)"
        ),
    )
    parser.add_argument(
        "--skip_final_train",
        action="store_true",
        default=True,
        help="Skip final production model training (default: True for comparison runs)",
    )
    parser.add_argument(
        "--no_skip_final_train",
        dest="skip_final_train",
        action="store_false",
        help="Enable final production model training (overrides --skip_final_train default)",
    )
    parser.add_argument(
        "--skip_legacy_holdout",
        action="store_true",
        default=False,
        help="Skip legacy holdout evaluation",
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        default=False,
        help=(
            "Skip all model training runs. Only aggregate existing "
            "comparison_summary_{loto_type}.json files into cross-loto summary."
        ),
    )
    parser.add_argument(
        "--run_root",
        default="runs",
        help="Directory where per-run subdirectories are created (default: runs)",
    )
    parser.add_argument(
        "--repo_root",
        default=str(REPO_ROOT),
        help="Repository root (default: parent of scripts/)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Statistical significance threshold for pairwise comparisons (default: 0.05)",
    )
    parser.add_argument(
        "--data_dir",
        default="data",
        help="Directory for output artifacts (default: data)",
    )
    return parser.parse_args()


def parse_seeds(seeds_str: str) -> list[int]:
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


def parse_loto_types(loto_types_str: str) -> list[str]:
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


def run_single_seed(
    *,
    loto_type: str,
    preset: str,
    seed: int,
    model_variant: str,
    evaluation_model_variants: str,
    saved_calibration_method: str,
    evaluation_calibration_methods: str,
    skip_final_train: bool,
    skip_legacy_holdout: bool,
    repo_root: str,
    run_root: str,
) -> Path:
    """Execute a single seed experiment and return its run directory."""
    config = {
        "loto_type": loto_type,
        "preset": preset,
        "seed": seed,
        "model_variant": model_variant,
        "evaluation_model_variants": evaluation_model_variants,
        "saved_calibration_method": saved_calibration_method,
        "evaluation_calibration_methods": evaluation_calibration_methods,
        "skip_final_train": skip_final_train,
        "skip_legacy_holdout": skip_legacy_holdout,
    }
    resolved_config = resolve_experiment_config(config)
    run_dir, summary = execute_experiment(
        resolved_config=resolved_config,
        repo_root=repo_root,
        run_root=run_root,
        requested_config=config,
    )
    if summary["status"] != "succeeded":
        raise SystemExit(f"Seed {seed} run failed for {loto_type}: {summary['status']}")
    print(f"    Seed {seed}: run_dir={run_dir}")
    return run_dir


def load_eval_report_from_run(run_dir: Path, loto_type: str) -> dict:
    """Load the eval_report artifact copied into a run directory."""
    report_path = run_dir / "artifacts" / "data" / f"eval_report_{loto_type}.json"
    if not report_path.exists():
        raise FileNotFoundError(f"eval_report not found in run directory: {report_path}")
    return load_eval_report(report_path)


def run_loto_type_comparison(
    *,
    loto_type: str,
    preset: str,
    seeds: list[int],
    model_variant: str,
    evaluation_model_variants: str,
    saved_calibration_method: str,
    evaluation_calibration_methods: str,
    skip_final_train: bool,
    skip_legacy_holdout: bool,
    repo_root: str,
    run_root: str,
    alpha: float,
    data_dir: str,
) -> dict:
    """Run multi-seed comparison for one loto_type and return its comparison_summary."""
    print(f"\n  --- {loto_type}: running {len(seeds)} seed(s) ---")
    run_dirs: list[Path] = []
    for seed in seeds:
        print(f"  Running seed {seed} for {loto_type}...")
        run_dir = run_single_seed(
            loto_type=loto_type,
            preset=preset,
            seed=seed,
            model_variant=model_variant,
            evaluation_model_variants=evaluation_model_variants,
            saved_calibration_method=saved_calibration_method,
            evaluation_calibration_methods=evaluation_calibration_methods,
            skip_final_train=skip_final_train,
            skip_legacy_holdout=skip_legacy_holdout,
            repo_root=repo_root,
            run_root=run_root,
        )
        run_dirs.append(run_dir)

    eval_reports: list[dict] = []
    for run_dir, seed in zip(run_dirs, seeds):
        try:
            report = load_eval_report_from_run(run_dir, loto_type)
            eval_reports.append(report)
            print(f"  Loaded eval_report for seed {seed}: {run_dir.name}")
        except FileNotFoundError as exc:
            print(f"  WARNING: {exc} — skipping seed {seed} from summary.")

    if not eval_reports:
        raise SystemExit(
            f"No eval_reports could be loaded for {loto_type}. Cannot build comparison summary."
        )

    summary = build_comparison_summary(
        eval_reports=eval_reports,
        loto_type=loto_type,
        preset=preset,
        seeds=seeds,
        alpha=alpha,
    )
    output_path = get_comparison_summary_path(loto_type, data_dir=data_dir)
    save_comparison_summary(summary, output_path)
    print(f"  Per-loto summary saved: {output_path}")
    return summary


def load_existing_comparison_summary(loto_type: str, data_dir: str) -> dict | None:
    """Load an existing comparison_summary for a loto_type, or return None."""
    path = get_comparison_summary_path(loto_type, data_dir=data_dir)
    if not path.exists():
        return None
    try:
        return load_comparison_summary(path)
    except Exception as exc:
        print(f"  WARNING: Failed to load {path}: {exc}")
        return None


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    loto_types = parse_loto_types(args.loto_types)
    repo_root = str(Path(args.repo_root).resolve())
    data_dir = str(Path(repo_root) / args.data_dir) if not Path(args.data_dir).is_absolute() else args.data_dir

    print(f"\n=== Cross-Loto Architecture Comparison ===")
    print(f"loto_types: {loto_types}")
    print(f"preset:     {args.preset}")
    print(f"seeds:      {seeds}")
    print(f"variants:   {args.evaluation_model_variants}")
    print(f"run_root:   {args.run_root}")
    print(f"skip_training: {args.skip_training}")
    print()

    per_loto_summaries: dict[str, dict] = {}

    if args.skip_training:
        # Only aggregate from existing comparison_summary files
        print("--- Skipping training: loading existing comparison_summary files ---")
        for loto_type in loto_types:
            summary = load_existing_comparison_summary(loto_type, data_dir)
            if summary is not None:
                per_loto_summaries[loto_type] = summary
                print(f"  Loaded existing summary for {loto_type}: run_count={summary.get('run_count')}")
            else:
                print(f"  WARNING: No comparison_summary found for {loto_type} — skipping.")
    else:
        # Run per-loto comparisons
        for loto_type in loto_types:
            try:
                summary = run_loto_type_comparison(
                    loto_type=loto_type,
                    preset=args.preset,
                    seeds=seeds,
                    model_variant=args.model_variant,
                    evaluation_model_variants=args.evaluation_model_variants,
                    saved_calibration_method=args.saved_calibration_method,
                    evaluation_calibration_methods=args.evaluation_calibration_methods,
                    skip_final_train=args.skip_final_train,
                    skip_legacy_holdout=args.skip_legacy_holdout,
                    repo_root=repo_root,
                    run_root=args.run_root,
                    alpha=args.alpha,
                    data_dir=data_dir,
                )
                per_loto_summaries[loto_type] = summary
            except SystemExit as exc:
                print(f"\nERROR for {loto_type}: {exc}. Continuing with remaining loto_types.")

    if not per_loto_summaries:
        raise SystemExit("No per-loto summaries available. Cannot build cross-loto summary.")

    # Build cross-loto summary
    print("\n--- Building cross-loto summary ---")
    cross_summary = build_cross_loto_summary(
        per_loto_summaries=per_loto_summaries,
        loto_types=list(per_loto_summaries.keys()),
        preset=args.preset,
        seeds=seeds,
        alpha=args.alpha,
    )
    cross_summary_path = get_cross_loto_summary_path(data_dir=data_dir)
    save_json(cross_summary, cross_summary_path)
    print(f"Cross-loto summary saved: {cross_summary_path}")

    # Build recommendation artifact
    recommendation = build_recommendation(cross_summary)
    rec_path = get_recommendation_path(data_dir=data_dir)
    save_json(recommendation, rec_path)
    print(f"Recommendation artifact saved: {rec_path}")

    # Human-readable summary
    print(f"\n=== Summary ({len(per_loto_summaries)} loto_type(s), seeds={seeds}) ===")
    overall_variants = (cross_summary.get("overall_summary") or {}).get("variants") or {}
    for variant_name in sorted(overall_variants.keys()):
        data = overall_variants[variant_name]
        ll = data.get("logloss") or {}
        mean_str = f"{ll['mean']:.4f}" if ll.get("mean") is not None else "N/A"
        std_str = f"±{ll['std']:.4f}" if ll.get("std") is not None else ""
        promote = data.get("promote_count_total", 0)
        hold = data.get("hold_count_total", 0)
        print(f"  {variant_name}: logloss={mean_str}{std_str}  promote={promote}/{promote+hold}")

    ranking = cross_summary.get("variant_ranking") or {}
    print("\n  Ranking by logloss:")
    for entry in ranking.get("by_logloss") or []:
        mean = f"{entry['mean']:.4f}" if entry.get("mean") is not None else "N/A"
        print(f"    #{entry['rank']} {entry['variant']}  mean={mean}")

    pairwise = cross_summary.get("pairwise_comparison_summary") or {}
    keys_of_interest = ["settransformer_vs_deepsets", "settransformer_vs_multihot", "deepsets_vs_multihot"]
    for key in keys_of_interest:
        if key in pairwise:
            overall = pairwise[key].get("overall") or {}
            print(
                f"\n  {key} (overall):"
                f" ci_wins={overall.get('ci_wins')}/{overall.get('run_count')},"
                f" perm_wins={overall.get('permutation_wins')}/{overall.get('run_count')},"
                f" both_pass={overall.get('both_pass_count')}/{overall.get('run_count')}"
            )

    rec = recommendation
    print(f"\n=== Recommendation ===")
    print(f"  recommended_next_action: {rec.get('recommended_next_action')}")
    print(f"  recommended_challenger: {rec.get('recommended_challenger')}")
    print(f"  keep_production_as_is: {rec.get('keep_production_as_is')}")
    print(f"  whether_to_try_pma_or_isab_next: {rec.get('whether_to_try_pma_or_isab_next')}")
    print("  blockers_to_promotion:")
    for b in rec.get("blockers_to_promotion") or []:
        print(f"    - {b}")
    print("  next_experiment_recommendations:")
    for r in rec.get("next_experiment_recommendations") or []:
        print(f"    - {r}")
    print()


if __name__ == "__main__":
    main()
