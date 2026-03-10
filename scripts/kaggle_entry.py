import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PRESET = "fast"


def build_loto_args():
    loto_type = os.environ.get("LOTO_TYPE", "").strip()
    if not loto_type:
        return []
    return ["--loto_type", loto_type]


def run_command(label, command, allow_failure=False):
    print(f"[kaggle-entry] {label}: {' '.join(command)}", flush=True)
    result = subprocess.run(command, cwd=PROJECT_ROOT)
    if result.returncode != 0 and not allow_failure:
        raise SystemExit(result.returncode)
    return result.returncode


def main():
    os.chdir(PROJECT_ROOT)
    loto_args = build_loto_args()
    train_preset = os.environ.get("LOTO_TRAIN_PRESET", DEFAULT_PRESET).strip() or DEFAULT_PRESET
    skip_final_train = os.environ.get("LOTO_SKIP_FINAL_TRAIN", "").lower() in {"1", "true", "yes"}

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "train_preset": train_preset,
        "loto_type": os.environ.get("LOTO_TYPE") or None,
    }

    data_command = [sys.executable, "data_collector.py", *loto_args]
    data_returncode = run_command("data_collector", data_command, allow_failure=True)
    summary["data_collection_returncode"] = data_returncode
    summary["used_bundled_data_fallback"] = data_returncode != 0

    if data_returncode != 0:
        print("[kaggle-entry] data collection failed, continuing with bundled data/.", flush=True)

    train_command = [sys.executable, "train_prob_model.py", *loto_args, "--preset", train_preset]
    if skip_final_train:
        train_command.append("--skip_final_train")
    train_returncode = run_command("train_prob_model", train_command)
    summary["train_returncode"] = train_returncode

    predict_command = [sys.executable, "predict.py", *loto_args]
    predict_returncode = run_command("predict", predict_command)
    summary["predict_returncode"] = predict_returncode

    summary["data_files"] = sorted(str(path.relative_to(PROJECT_ROOT)) for path in (PROJECT_ROOT / "data").glob("*"))
    summary["model_files"] = sorted(str(path.relative_to(PROJECT_ROOT)) for path in (PROJECT_ROOT / "models").glob("*"))

    with open(PROJECT_ROOT / "kaggle_run_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
