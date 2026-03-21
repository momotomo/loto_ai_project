"""scripts/run_multi_seed.py

Run train_prob_model.py for multiple seeds and aggregate the results into a
cross-seed comparison summary artifact (data/comparison_summary_{loto_type}.json).

Typical usage for a deepsets vs settransformer architecture comparison:

    python scripts/run_multi_seed.py \\
        --loto_type loto6 \\
        --preset archcomp \\
        --seeds 42,123,456 \\
        --model_variant legacy \\
        --evaluation_model_variants legacy,multihot,deepsets,settransformer \\
        --saved_calibration_method none \\
        --evaluation_calibration_methods none,temperature,isotonic \\
        --skip_final_train \\
        --run_root runs

The script:
1. Runs a single experiment per seed via experiment_runner.execute_experiment.
2. Loads the eval_report artifact from each run directory.
3. Aggregates results with comparison_summary.build_comparison_summary.
4. Saves the summary to data/comparison_summary_{loto_type}.json.

Notes
-----
- Each run produces its own run directory under --run_root.
- Production artifacts are NOT updated unless --model_variant is set and
  --skip_final_train is omitted.  The default is --skip_final_train to keep
  multi-seed comparison runs separate from the production save path.
- The comparison summary is written to data/ (the standard artifact location);
  it is NOT uploaded to Kaggle automatically.
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
from experiment_runner import execute_experiment, resolve_experiment_config  # noqa: E402
from model_variants import MODEL_VARIANT_CHOICES  # noqa: E402
from config import LOTO_CONFIG  # noqa: E402

VALID_PRESETS = ["default", "fast", "smoke", "archcomp"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run train_prob_model.py for multiple seeds and aggregate results.",
    )
    parser.add_argument("--loto_type", choices=sorted(LOTO_CONFIG.keys()), required=True)
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
        help="Comma-separated variants to evaluate (default: legacy,multihot,deepsets,settransformer)",
    )
    parser.add_argument(
        "--saved_calibration_method",
        default="none",
        help="Calibration method to save in the production artifact (default: none)",
    )
    parser.add_argument(
        "--evaluation_calibration_methods",
        default="none,temperature,isotonic",
        help="Comma-separated calibration methods to evaluate (default: none,temperature,isotonic)",
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
        help="Statistical significance threshold for pairwise comparison summary (default: 0.05)",
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
        raise SystemExit(f"Seed {seed} run failed: {summary['status']}")
    print(f"  Seed {seed}: run_dir={run_dir}")
    return run_dir


def load_eval_report_from_run(run_dir: Path, loto_type: str) -> dict:
    """Load the eval_report artifact copied into a run directory."""
    report_path = run_dir / "artifacts" / "data" / f"eval_report_{loto_type}.json"
    if not report_path.exists():
        raise FileNotFoundError(f"eval_report not found in run directory: {report_path}")
    return load_eval_report(report_path)


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    repo_root = str(Path(args.repo_root).resolve())

    print(f"\n=== Architecture Comparison Multi-Seed Run ===")
    print(f"loto_type: {args.loto_type}")
    print(f"preset:    {args.preset}")
    print(f"seeds:     {seeds}")
    print(f"variants:  {args.evaluation_model_variants}")
    print(f"calibration eval: {args.evaluation_calibration_methods}")
    print(f"run_root:  {args.run_root}")
    print()

    run_dirs: list[Path] = []
    for seed in seeds:
        print(f"--- Running seed {seed} ---")
        run_dir = run_single_seed(
            loto_type=args.loto_type,
            preset=args.preset,
            seed=seed,
            model_variant=args.model_variant,
            evaluation_model_variants=args.evaluation_model_variants,
            saved_calibration_method=args.saved_calibration_method,
            evaluation_calibration_methods=args.evaluation_calibration_methods,
            skip_final_train=args.skip_final_train,
            skip_legacy_holdout=args.skip_legacy_holdout,
            repo_root=repo_root,
            run_root=args.run_root,
        )
        run_dirs.append(run_dir)

    print("\n--- Aggregating results ---")
    eval_reports: list[dict] = []
    for run_dir, seed in zip(run_dirs, seeds):
        try:
            report = load_eval_report_from_run(run_dir, args.loto_type)
            eval_reports.append(report)
            print(f"  Loaded eval_report for seed {seed}: {run_dir.name}")
        except FileNotFoundError as exc:
            print(f"  WARNING: {exc} — skipping seed {seed} from summary.")

    if not eval_reports:
        raise SystemExit("No eval_reports could be loaded. Cannot build comparison summary.")

    summary = build_comparison_summary(
        eval_reports=eval_reports,
        loto_type=args.loto_type,
        preset=args.preset,
        seeds=seeds,
        alpha=args.alpha,
    )

    output_path = get_comparison_summary_path(args.loto_type)
    save_comparison_summary(summary, output_path)
    print(f"\n✅ Comparison summary saved to {output_path}")

    # Print a brief human-readable summary
    print(f"\n=== Summary ({len(eval_reports)} runs, seeds={seeds}) ===")
    for variant_name, data in summary["variants"].items():
        logloss = data["logloss"]
        mean_str = f"{logloss['mean']:.4f}" if logloss["mean"] is not None else "N/A"
        std_str = f"±{logloss['std']:.4f}" if logloss["std"] is not None else ""
        print(f"  {variant_name}: logloss={mean_str}{std_str}  promote={data['promote_count']}/{data['run_count']}")

    pairwise = summary.get("pairwise_comparisons") or {}
    keys_of_interest = ["settransformer_vs_deepsets", "settransformer_vs_multihot", "deepsets_vs_multihot"]
    for key in keys_of_interest:
        if key in pairwise:
            data = pairwise[key]
            print(
                f"  {key}: ci_wins={data['ci_wins']}/{data['run_count']}, "
                f"perm_wins={data['permutation_wins']}/{data['run_count']}, "
                f"both_pass={data['both_pass_count']}/{data['run_count']}"
            )

    print()


if __name__ == "__main__":
    main()
