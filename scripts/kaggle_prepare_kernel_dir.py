import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT_FILES = [
    "config.py",
    "data_collector.py",
    "train_prob_model.py",
    "predict.py",
    "update_system.py",
]
ROOT_DIRS = [
    "data",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare a Kaggle kernel build directory from the repository.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--kernel-id", required=True, help="username/kernel-slug")
    parser.add_argument("--kernel-title", help="Optional Kaggle kernel title")
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--train-preset", default="fast")
    parser.add_argument("--skip-legacy-holdout", action="store_true")
    return parser.parse_args()


def slug_to_title(kernel_id):
    slug = kernel_id.split("/", 1)[-1]
    return slug.replace("-", " ").replace("_", " ").title()


def reset_build_dir(build_dir):
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)


def copy_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_data_dir(source_dir, destination_dir):
    destination_dir.mkdir(parents=True, exist_ok=True)
    for csv_path in sorted(source_dir.glob("*.csv")):
        copy_file(csv_path, destination_dir / csv_path.name)


def build_metadata(kernel_id, kernel_title):
    return {
        "id": kernel_id,
        "title": kernel_title,
        "code_file": "kaggle_entry.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": True,
    }


def build_run_config(args):
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "targets": args.targets,
        "train_preset": args.train_preset,
        "skip_legacy_holdout": args.skip_legacy_holdout,
    }


def main():
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    build_dir = Path(args.build_dir).resolve()
    entry_script = repo_root / "scripts" / "kaggle_entry.py"

    reset_build_dir(build_dir)

    for relative_path in ROOT_FILES:
        source = repo_root / relative_path
        if not source.exists():
            raise SystemExit(f"Missing required source file: {source}")
        copy_file(source, build_dir / relative_path)

    for relative_path in ROOT_DIRS:
        source = repo_root / relative_path
        if source.exists():
            copy_data_dir(source, build_dir / relative_path)

    copy_file(entry_script, build_dir / "kaggle_entry.py")

    metadata = build_metadata(args.kernel_id, args.kernel_title or slug_to_title(args.kernel_id))
    with open(build_dir / "kernel-metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    with open(build_dir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(build_run_config(args), handle, indent=2, ensure_ascii=False)

    print(f"Prepared Kaggle kernel directory: {build_dir}")


if __name__ == "__main__":
    main()
