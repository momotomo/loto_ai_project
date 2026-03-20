import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment_runner import run_from_inputs


def parse_args():
    parser = argparse.ArgumentParser(description="Run a tracked experiment and snapshot its artifacts into runs/.")
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument("--config", help="JSON config file path")
    config_group.add_argument("--config-json", help="Inline JSON config")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--run-name", help="Optional explicit run directory name")
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir, summary = run_from_inputs(
        config_path=args.config,
        config_json=args.config_json,
        repo_root=args.repo_root,
        run_root=args.run_root,
        run_name=args.run_name,
    )
    print(f"run_dir={run_dir}")
    print(f"status={summary['status']}")
    if summary.get("manifest_summary"):
        print(f"bundle_id={summary['manifest_summary'].get('bundle_id')}")


if __name__ == "__main__":
    main()
