import argparse
import json
import shutil
from pathlib import Path


ROOT_FILES = [
    "AGENT.md",
    "README.md",
    "config.py",
    "data_collector.py",
    "predict.py",
    "train_prob_model.py",
    "update_system.py",
    "requirements.txt",
]
ROOT_DIRS = [
    "data",
    "docs",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare a Kaggle kernel build directory from the repository.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--kernel-id", required=True, help="username/kernel-slug")
    parser.add_argument("--kernel-title", help="Optional Kaggle kernel title")
    return parser.parse_args()


def slug_to_title(kernel_id):
    slug = kernel_id.split("/", 1)[-1]
    return slug.replace("-", " ").replace("_", " ").title()


def reset_build_dir(build_dir):
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)


def copy_path(source, destination):
    if source.is_dir():
        shutil.copytree(
            source,
            destination,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
        )
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


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


def main():
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    build_dir = Path(args.build_dir).resolve()
    entry_script = repo_root / "scripts" / "kaggle_entry.py"

    reset_build_dir(build_dir)

    for relative_path in ROOT_FILES:
        source = repo_root / relative_path
        if source.exists():
            copy_path(source, build_dir / relative_path)

    for relative_path in ROOT_DIRS:
        source = repo_root / relative_path
        if source.exists():
            copy_path(source, build_dir / relative_path)

    copy_path(entry_script, build_dir / "kaggle_entry.py")

    metadata = build_metadata(args.kernel_id, args.kernel_title or slug_to_title(args.kernel_id))
    with open(build_dir / "kernel-metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    print(f"Prepared Kaggle kernel directory: {build_dir}")


if __name__ == "__main__":
    main()
