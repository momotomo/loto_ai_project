import base64
import json
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path


PAYLOAD_BASE64 = """__KAGGLE_PAYLOAD_BASE64__"""
UNSET_PAYLOAD = "__KAGGLE_PAYLOAD_BASE64__"
DEFAULT_APP_DIR = Path(os.environ.get("KAGGLE_APP_DIR", "/kaggle/working/app"))
ROOT_OUTPUT_DIR = Path("/kaggle/working")
SUMMARY_PATH = Path("/kaggle/working/kaggle_run_summary.json")
REQUIRED_PAYLOAD_FILES = (
    "artifact_utils.py",
    "calibration_utils.py",
    "evaluation_statistics.py",
    "config.py",
    "data_collector.py",
    "model_variants.py",
    "report_utils.py",
    "train_prob_model.py",
    "predict.py",
    "update_system.py",
    "run_config.json",
)
DEBUG_LISTING_LIMIT = 80
DATA_EXPORT_PATTERNS = (
    "*_processed.csv",
    "*_raw.csv",
    "manifest_*.json",
    "eval_report_*.json",
    "prediction_history_*.json",
    "*_feature_cols.json",
    "*_calibrator.json",
)
MODEL_EXPORT_PATTERNS = (
    "*_prob.keras",
    "*_scaler.pkl",
    "*_feature_cols.json",
    "*_calibrator.json",
)


def print_directory_listing(path):
    print(f"[kaggle-entry] listing for {path}", flush=True)
    if not path.exists():
        print("[kaggle-entry] path does not exist", flush=True)
        return

    entries = sorted(path.rglob("*"), key=lambda item: item.as_posix())
    for index, entry in enumerate(entries):
        if index >= DEBUG_LISTING_LIMIT:
            print("[kaggle-entry] ... listing truncated ...", flush=True)
            break
        relative_path = entry.relative_to(path).as_posix()
        entry_type = "dir" if entry.is_dir() else "file"
        print(f"[kaggle-entry]   {entry_type}: {relative_path}", flush=True)


def extract_payload(app_dir):
    if PAYLOAD_BASE64 == UNSET_PAYLOAD:
        raise SystemExit("Embedded payload is missing. Rebuild script.py via scripts/kaggle_prepare_kernel_dir.py.")

    if app_dir.exists():
        shutil.rmtree(app_dir)
    app_dir.mkdir(parents=True, exist_ok=True)

    payload_bytes = base64.b64decode(PAYLOAD_BASE64.encode("ascii"))
    with zipfile.ZipFile(BytesIO(payload_bytes)) as archive:
        archive.extractall(app_dir)

    print(f"[kaggle-entry] extracted payload to {app_dir}", flush=True)
    print_directory_listing(app_dir)
    return app_dir


def ensure_required_files(app_dir):
    missing = [relative_path for relative_path in REQUIRED_PAYLOAD_FILES if not (app_dir / relative_path).exists()]
    if missing:
        print_directory_listing(app_dir)
        raise SystemExit(f"Missing extracted payload files: {', '.join(missing)}")


def load_run_config(app_dir):
    config_path = app_dir / "run_config.json"
    with open(config_path, "r", encoding="utf-8") as handle:
        run_config = json.load(handle)
    print("[kaggle-entry] run_config:", flush=True)
    print(json.dumps(run_config, ensure_ascii=False, indent=2), flush=True)
    return run_config


def run_command(label, app_dir, command):
    print(f"[kaggle-entry] {label}: {' '.join(command)}", flush=True)
    result = subprocess.run(command, cwd=app_dir)
    print(f"[kaggle-entry] {label} returncode={result.returncode}", flush=True)
    return result.returncode


def list_relative_files(path):
    if not path.exists():
        return []
    return sorted(item.relative_to(path).as_posix() for item in path.rglob("*") if item.is_file())


def collect_matching_files(base_dir, patterns):
    if not base_dir.exists():
        return []

    matched = {}
    for pattern in patterns:
        for path in sorted(base_dir.glob(pattern)):
            if path.is_file():
                matched[path.name] = path
    return [matched[name] for name in sorted(matched)]


def export_directory_subset(source_dir, destination_dir, patterns, label):
    exported = []
    destination_dir.mkdir(parents=True, exist_ok=True)

    for source_path in collect_matching_files(source_dir, patterns):
        destination_path = destination_dir / source_path.name
        shutil.copy2(source_path, destination_path)
        exported.append(source_path.name)
        print(f"[kaggle-entry] export {label}: {source_path} -> {destination_path}", flush=True)

    if not exported:
        print(f"[kaggle-entry] export {label}: no matching files under {source_dir}", flush=True)
    return exported


def export_artifacts_to_output_root(app_dir, output_root=ROOT_OUTPUT_DIR):
    source_data_dir = app_dir / "data"
    source_model_dir = app_dir / "models"
    output_data_dir = output_root / "data"
    output_model_dir = output_root / "models"

    exported_data_files = export_directory_subset(source_data_dir, output_data_dir, DATA_EXPORT_PATTERNS, "data")
    exported_model_files = export_directory_subset(source_model_dir, output_model_dir, MODEL_EXPORT_PATTERNS, "models")

    print_directory_listing(output_data_dir)
    print_directory_listing(output_model_dir)
    return {
        "exported_to_root": True,
        "exported_data_files": exported_data_files,
        "exported_model_files": exported_model_files,
        "root_data_files": list_relative_files(output_data_dir),
        "root_model_files": list_relative_files(output_model_dir),
    }


def write_summary(summary):
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"[kaggle-entry] wrote summary to {SUMMARY_PATH}", flush=True)


def main():
    app_dir = extract_payload(DEFAULT_APP_DIR)
    ensure_required_files(app_dir)
    os.chdir(app_dir)

    run_config = load_run_config(app_dir)
    targets = run_config.get("targets") or []
    if not targets:
        print("[kaggle-entry] run_config.json did not contain any targets; nothing to do.", flush=True)
        return

    train_preset = run_config.get("train_preset", "fast")
    model_variant = str(run_config.get("model_variant", "legacy"))
    evaluation_model_variants = str(run_config.get("evaluation_model_variants", "legacy,multihot"))
    saved_calibration_method = str(run_config.get("saved_calibration_method", "none"))
    evaluation_calibration_methods = str(
        run_config.get("evaluation_calibration_methods", "none,temperature,isotonic")
    )
    seed = int(run_config.get("seed", 42))
    skip_legacy_holdout = bool(run_config.get("skip_legacy_holdout", False))

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "app_dir": str(app_dir),
        "run_config": run_config,
        "targets": {},
    }
    exit_code = 0

    try:
        for loto_type in targets:
            target_summary = {}

            data_command = [sys.executable, "data_collector.py", "--loto_type", loto_type]
            data_returncode = run_command(f"data_collector[{loto_type}]", app_dir, data_command)
            target_summary["data_collection_returncode"] = data_returncode
            target_summary["used_bundled_data_fallback"] = data_returncode != 0
            if data_returncode != 0:
                print(
                    f"[kaggle-entry] data collection failed for {loto_type}; continuing with bundled data/*.csv fallback.",
                    flush=True,
                )

            train_command = [
                sys.executable,
                "train_prob_model.py",
                "--loto_type",
                loto_type,
                "--preset",
                train_preset,
                "--model_variant",
                model_variant,
                "--evaluation_model_variants",
                evaluation_model_variants,
                "--saved_calibration_method",
                saved_calibration_method,
                "--evaluation_calibration_methods",
                evaluation_calibration_methods,
                "--seed",
                str(seed),
            ]
            if skip_legacy_holdout:
                train_command.append("--skip_legacy_holdout")

            train_returncode = run_command(f"train_prob_model[{loto_type}]", app_dir, train_command)
            target_summary["train_returncode"] = train_returncode
            summary["targets"][loto_type] = target_summary

            if train_returncode != 0:
                exit_code = train_returncode
                break
    finally:
        export_summary = export_artifacts_to_output_root(app_dir)
        summary["data_files"] = list_relative_files(app_dir / "data")
        summary["model_files"] = list_relative_files(app_dir / "models")
        summary.update(export_summary)
        write_summary(summary)

    if exit_code != 0:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
