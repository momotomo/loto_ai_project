import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_SCRIPTS = ("data_collector.py", "train_prob_model.py")
DEBUG_LISTING_LIMIT = 40


def iter_candidate_roots():
    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()
    seen = set()
    for base in [script_dir, cwd, *script_dir.parents, *cwd.parents]:
        for candidate in (base, base / "src"):
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield resolved


def locate_repo_root():
    for candidate in iter_candidate_roots():
        if all((candidate / script_name).exists() for script_name in REQUIRED_SCRIPTS):
            return candidate
    return None


def print_directory_listing(path):
    print(f"[kaggle-entry] debug listing for {path}", flush=True)
    if not path.exists():
        print("[kaggle-entry] path does not exist", flush=True)
        return
    for index, entry in enumerate(sorted(path.iterdir(), key=lambda item: item.name)):
        if index >= DEBUG_LISTING_LIMIT:
            print("[kaggle-entry] ... listing truncated ...", flush=True)
            break
        entry_type = "dir" if entry.is_dir() else "file"
        print(f"[kaggle-entry]   {entry_type}: {entry.name}", flush=True)


def load_run_config(repo_root):
    config_path = repo_root / "run_config.json"
    if not config_path.exists():
        raise SystemExit(f"Missing run_config.json in {repo_root}")
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def run_command(label, repo_root, command, allow_failure=False):
    print(f"[kaggle-entry] {label}: {' '.join(command)}", flush=True)
    result = subprocess.run(command, cwd=repo_root)
    if result.returncode != 0 and not allow_failure:
        raise SystemExit(result.returncode)
    return result.returncode


def main():
    repo_root = locate_repo_root()
    if repo_root is None:
        script_dir = Path(__file__).resolve().parent
        print("[kaggle-entry] failed to locate repo root containing data_collector.py and train_prob_model.py", flush=True)
        print_directory_listing(script_dir)
        print_directory_listing(Path.cwd().resolve())
        raise SystemExit(1)

    os.chdir(repo_root)
    run_config = load_run_config(repo_root)
    targets = run_config.get("targets", [])
    if not targets:
        print("[kaggle-entry] run_config.json did not contain any targets; nothing to do.", flush=True)
        return

    train_preset = run_config.get("train_preset", "fast")
    skip_legacy_holdout = bool(run_config.get("skip_legacy_holdout", False))

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "run_config": run_config,
        "targets": {},
    }

    for loto_type in targets:
        target_summary = {}
        data_command = [sys.executable, str(repo_root / "data_collector.py"), "--loto_type", loto_type]
        data_returncode = run_command(f"data_collector[{loto_type}]", repo_root, data_command, allow_failure=True)
        target_summary["data_collection_returncode"] = data_returncode
        target_summary["used_bundled_data_fallback"] = data_returncode != 0
        if data_returncode != 0:
            print(f"[kaggle-entry] data collection failed for {loto_type}, continuing with bundled data/ fallback.", flush=True)

        train_command = [
            sys.executable,
            str(repo_root / "train_prob_model.py"),
            "--loto_type",
            loto_type,
            "--preset",
            train_preset,
        ]
        if skip_legacy_holdout:
            train_command.append("--skip_legacy_holdout")
        target_summary["train_returncode"] = run_command(f"train_prob_model[{loto_type}]", repo_root, train_command)
        summary["targets"][loto_type] = target_summary

    summary["data_files"] = sorted(path.name for path in (repo_root / "data").glob("*"))
    summary["model_files"] = sorted(path.name for path in (repo_root / "models").glob("*"))

    with open(repo_root / "kaggle_run_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
