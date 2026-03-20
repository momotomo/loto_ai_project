import json
import shutil
import subprocess
import sys
from pathlib import Path

from config import PREPROCESSING_VERSION


ROOT_FILES = [
    "artifact_utils.py",
    "calibration_utils.py",
    "config.py",
    "data_collector.py",
    "evaluation_statistics.py",
    "model_variants.py",
    "predict.py",
    "report_utils.py",
    "train_prob_model.py",
    "update_system.py",
]
LEGACY_ARTIFACT_FILES = [
    "data/loto6_feature_cols.json",
    "models/loto6_feature_cols.json",
    "models/loto6_prob.keras",
    "models/loto6_scaler.pkl",
]


def prepare_workspace(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    for relative_path in ROOT_FILES:
        source = repo_root / relative_path
        destination = workspace / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    (workspace / "data").mkdir()
    for relative_path in ["data/loto6_processed.csv", "data/loto6_raw.csv"]:
        source = repo_root / relative_path
        destination = workspace / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    for relative_path in LEGACY_ARTIFACT_FILES:
        source = repo_root / relative_path
        if source.exists():
            destination = workspace / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    return workspace


def run_command(workspace, *args):
    result = subprocess.run(
        [sys.executable, *args],
        cwd=workspace,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return result


def test_train_prob_model_multihot_smoke_writes_variant_stats_and_calibration_artifacts(tmp_path):
    workspace = prepare_workspace(tmp_path)
    run_command(
        workspace,
        "train_prob_model.py",
        "--loto_type",
        "loto6",
        "--preset",
        "smoke",
        "--model_variant",
        "multihot",
        "--evaluation_model_variants",
        "legacy,multihot",
        "--saved_calibration_method",
        "temperature",
        "--evaluation_calibration_methods",
        "none,temperature",
        "--seed",
        "123",
    )
    predict_result = run_command(workspace, "predict.py", "--loto_type", "loto6")

    manifest = json.loads((workspace / "data" / "manifest_loto6.json").read_text(encoding="utf-8"))
    report = json.loads((workspace / "data" / "eval_report_loto6.json").read_text(encoding="utf-8"))

    assert manifest["training_context"]["model_variant"] == "multihot"
    assert manifest["training_context"]["feature_strategy"] == "derived_multihot"
    assert manifest["data_fingerprint"]["preprocessing_version"] == PREPROCESSING_VERSION
    assert manifest["training_context"]["saved_calibration_method"] == "temperature"
    assert manifest["calibration"]["saved_method"] == "temperature"
    assert manifest["metrics_summary"]["post_calibration_logloss"] is not None
    assert "legacy" in report["model_variants"]
    assert "multihot" in report["model_variants"]
    assert "comparisons" in report["statistical_tests"]
    assert "rule" in report["decision_summary"]
    assert "variant: multihot" in predict_result.stdout
    assert "calibration: Temperature Scaling (temperature)" in predict_result.stdout
    assert (workspace / "models" / "loto6_calibrator.json").exists()


def test_update_system_and_predict_smoke_support_variant_aware_artifacts(tmp_path):
    workspace = prepare_workspace(tmp_path)
    run_command(
        workspace,
        "update_system.py",
        "--loto_type",
        "loto6",
        "--train_preset",
        "smoke",
        "--model_variant",
        "legacy",
        "--evaluation_model_variants",
        "legacy,multihot",
        "--skip_final_train",
        "--skip_data_refresh",
        "--seed",
        "321",
    )
    predict_result = run_command(workspace, "predict.py", "--loto_type", "loto6")

    manifest = json.loads((workspace / "data" / "manifest_loto6.json").read_text(encoding="utf-8"))
    assert manifest["training_context"]["model_variant"] == "legacy"
    assert "variant: legacy" in predict_result.stdout
    assert "calibration: No Calibration (none)" in predict_result.stdout


def test_predict_smoke_falls_back_when_calibrator_artifact_is_missing(tmp_path):
    workspace = prepare_workspace(tmp_path)
    run_command(
        workspace,
        "train_prob_model.py",
        "--loto_type",
        "loto6",
        "--preset",
        "smoke",
        "--model_variant",
        "legacy",
        "--evaluation_model_variants",
        "legacy,multihot",
        "--saved_calibration_method",
        "temperature",
        "--evaluation_calibration_methods",
        "none,temperature",
        "--seed",
        "222",
    )
    for relative_path in ["data/loto6_calibrator.json", "models/loto6_calibrator.json"]:
        target = workspace / relative_path
        if target.exists():
            target.unlink()

    predict_result = run_command(workspace, "predict.py", "--loto_type", "loto6")
    assert "saved calibration requested in manifest but no fitted calibrator artifact was applied" in predict_result.stdout
