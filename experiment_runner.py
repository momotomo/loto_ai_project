import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from artifact_utils import collect_file_metadata, collect_runtime_environment
from config import RUN_TRACKING_SCHEMA_VERSION


DEFAULT_EXPERIMENT_CONFIG = {
    "loto_type": None,
    "preset": "default",
    "seed": 42,
    "model_variant": "legacy",
    "evaluation_model_variants": "legacy,multihot",
    "saved_calibration_method": "none",
    "evaluation_calibration_methods": "none,temperature,isotonic",
    "refresh_data": False,
    "skip_final_train": False,
    "skip_legacy_holdout": False,
    "skip_walk_forward": False,
    "initial_train_fraction": None,
    "walk_forward_test_window": None,
    "walk_forward_folds": None,
    "eval_epochs": None,
    "final_epochs": None,
    "batch_size": None,
    "patience": None,
}

# Valid preset names (kept in sync with train_prob_model.PRESET_CONFIGS)
VALID_PRESET_NAMES = ("default", "fast", "smoke", "archcomp")

TRACKED_SOURCE_FILES = [
    "AGENT.md",
    "README.md",
    "docs/ARCHITECTURE.md",
    "docs/ARTIFACTS.md",
    "docs/EVALUATION.md",
    "requirements.txt",
    "requirements.lock",
    "artifact_utils.py",
    "calibration_utils.py",
    "campaign_profiles.py",
    "campaign_manager.py",
    "benchmark_registry.py",
    "comparability_checker.py",
    "governance_layer.py",
    "comparison_summary.py",
    "cross_loto_summary.py",
    "cross_loto_report.py",
    "evaluation_statistics.py",
    "config.py",
    "data_collector.py",
    "model_variants.py",
    "report_utils.py",
    "predict.py",
    "train_prob_model.py",
    "update_system.py",
    "app.py",
    "scripts/kaggle_entry.py",
    "scripts/kaggle_prepare_kernel_dir.py",
    "scripts/run_campaign.py",
]

TRACKED_ARTIFACT_FILES = [
    "data/{loto_type}_raw.csv",
    "data/{loto_type}_processed.csv",
    "data/eval_report_{loto_type}.json",
    "data/manifest_{loto_type}.json",
    "data/prediction_history_{loto_type}.json",
    "data/{loto_type}_feature_cols.json",
    "data/{loto_type}_calibrator.json",
    "models/{loto_type}_prob.keras",
    "models/{loto_type}_scaler.pkl",
    "models/{loto_type}_feature_cols.json",
    "models/{loto_type}_calibrator.json",
]


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def load_experiment_config(config_path=None, config_json=None):
    if config_path:
        with open(config_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    if config_json:
        return json.loads(config_json)
    raise SystemExit("`--config` または `--config-json` のいずれかが必要です。")


def resolve_experiment_config(user_config):
    unknown_keys = sorted(set(user_config) - set(DEFAULT_EXPERIMENT_CONFIG))
    if unknown_keys:
        raise SystemExit(f"未対応の config key があります: {unknown_keys}")

    resolved = dict(DEFAULT_EXPERIMENT_CONFIG)
    resolved.update(user_config)

    if not resolved.get("loto_type"):
        raise SystemExit("config には `loto_type` が必要です。")
    if resolved["skip_legacy_holdout"] and resolved["skip_walk_forward"]:
        raise SystemExit("skip_legacy_holdout と skip_walk_forward を同時に true にできません。")

    return resolved


def resolve_path(repo_root, candidate):
    candidate_path = Path(candidate)
    if candidate_path.is_absolute():
        return candidate_path
    return repo_root / candidate_path


def build_run_id(resolved_config, now=None):
    timestamp = now or datetime.now(timezone.utc)
    compact = timestamp.strftime("%Y%m%dT%H%M%SZ")
    return (
        f"{compact}_{resolved_config['loto_type']}_{resolved_config['preset']}"
        f"_{resolved_config['model_variant']}"
        f"_seed{resolved_config['seed']}"
    )


def build_train_command(resolved_config):
    command = [
        sys.executable,
        "train_prob_model.py",
        "--loto_type",
        resolved_config["loto_type"],
        "--preset",
        resolved_config["preset"],
        "--model_variant",
        resolved_config["model_variant"],
        "--evaluation_model_variants",
        resolved_config["evaluation_model_variants"],
        "--saved_calibration_method",
        resolved_config["saved_calibration_method"],
        "--evaluation_calibration_methods",
        resolved_config["evaluation_calibration_methods"],
        "--seed",
        str(resolved_config["seed"]),
    ]
    boolean_flags = [
        "skip_final_train",
        "skip_legacy_holdout",
        "skip_walk_forward",
    ]
    value_options = [
        "initial_train_fraction",
        "walk_forward_test_window",
        "walk_forward_folds",
        "eval_epochs",
        "final_epochs",
        "batch_size",
        "patience",
    ]

    for flag_name in boolean_flags:
        if resolved_config.get(flag_name):
            command.append(f"--{flag_name}")

    for option_name in value_options:
        option_value = resolved_config.get(option_name)
        if option_value is not None:
            command.extend([f"--{option_name}", str(option_value)])

    return command


def build_refresh_command(resolved_config):
    return [sys.executable, "data_collector.py", "--loto_type", resolved_config["loto_type"]]


def default_command_runner(label, command, cwd):
    result = subprocess.run(command, cwd=cwd)
    return {
        "label": label,
        "command": command,
        "returncode": result.returncode,
    }


def collect_source_hashes(repo_root):
    source_map = {}
    for relative_path in TRACKED_SOURCE_FILES:
        source_path = repo_root / relative_path
        if source_path.exists():
            source_map[relative_path] = str(source_path)
    return collect_file_metadata(source_map)


def collect_artifact_paths(repo_root, loto_type):
    artifact_map = {}
    for template in TRACKED_ARTIFACT_FILES:
        relative_path = template.format(loto_type=loto_type)
        source_path = repo_root / relative_path
        if source_path.exists():
            artifact_map[relative_path] = str(source_path)
    return artifact_map


def copy_artifacts(repo_root, run_dir, loto_type):
    artifact_map = collect_artifact_paths(repo_root, loto_type)
    copied = {}
    for relative_path, source_path in artifact_map.items():
        source = Path(source_path)
        destination = run_dir / "artifacts" / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied[relative_path] = str(destination)
    return copied


def load_manifest_summary(repo_root, loto_type):
    manifest_path = repo_root / "data" / f"manifest_{loto_type}.json"
    if not manifest_path.exists():
        return {}

    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    return {
        "bundle_id": manifest.get("bundle_id"),
        "generated_at": manifest.get("generated_at"),
        "data_hash": ((manifest.get("data_fingerprint") or {}).get("data_hash")),
        "saved_model_variant": ((manifest.get("training_context") or {}).get("model_variant")),
        "saved_calibration_method": ((manifest.get("training_context") or {}).get("saved_calibration_method")),
        "recommended_model_variant": ((manifest.get("decision_summary") or {}).get("recommended_variant")),
        "recommended_calibration_method": ((manifest.get("decision_summary") or {}).get("recommended_calibration_method")),
        "final_artifact_status": ((manifest.get("metrics_summary") or {}).get("final_artifact_status")),
    }


def execute_experiment(
    resolved_config,
    repo_root=".",
    run_root="runs",
    run_name=None,
    command_runner=None,
    now=None,
    requested_config=None,
):
    repo_root = Path(repo_root).resolve()
    run_root = resolve_path(repo_root, run_root)
    command_runner = command_runner or default_command_runner
    started_at = now or datetime.now(timezone.utc)
    run_id = run_name or build_run_id(resolved_config, started_at)
    run_dir = run_root / run_id
    if run_dir.exists():
        raise SystemExit(f"run directory already exists: {run_dir}")

    summary = {
        "schema_version": RUN_TRACKING_SCHEMA_VERSION,
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "status": "running",
        "repo_root": str(repo_root),
        "runtime_environment": collect_runtime_environment(),
        "requested_config": requested_config,
        "resolved_config": resolved_config,
        "steps": [],
    }

    if requested_config is not None:
        save_json(run_dir / "config" / "requested_config.json", requested_config)
    save_json(run_dir / "config" / "resolved_config.json", resolved_config)
    save_json(run_dir / "source_hashes.json", collect_source_hashes(repo_root))

    try:
        if resolved_config.get("refresh_data"):
            refresh_step = command_runner("refresh_data", build_refresh_command(resolved_config), repo_root)
            summary["steps"].append(refresh_step)
            if refresh_step["returncode"] != 0:
                raise SystemExit(refresh_step["returncode"])

        train_step = command_runner("train", build_train_command(resolved_config), repo_root)
        summary["steps"].append(train_step)
        if train_step["returncode"] != 0:
            raise SystemExit(train_step["returncode"])

        copied_artifacts = copy_artifacts(repo_root, run_dir, resolved_config["loto_type"])
        summary["copied_artifacts"] = copied_artifacts
        summary["artifact_hashes"] = collect_file_metadata(collect_artifact_paths(repo_root, resolved_config["loto_type"]))
        summary["manifest_summary"] = load_manifest_summary(repo_root, resolved_config["loto_type"])
        summary["status"] = "succeeded"
        return run_dir, summary
    except SystemExit:
        summary["status"] = "failed"
        raise
    except Exception:
        summary["status"] = "failed"
        raise
    finally:
        finished_at = datetime.now(timezone.utc)
        summary["finished_at"] = finished_at.isoformat()
        save_json(run_dir / "run_summary.json", summary)


def run_from_inputs(config_path=None, config_json=None, repo_root=".", run_root="runs", run_name=None):
    requested_config = load_experiment_config(config_path=config_path, config_json=config_json)
    resolved_config = resolve_experiment_config(requested_config)
    run_dir, summary = execute_experiment(
        resolved_config=resolved_config,
        repo_root=repo_root,
        run_root=run_root,
        run_name=run_name,
        requested_config=requested_config,
    )
    return run_dir, summary
