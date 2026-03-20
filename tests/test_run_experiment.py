import json
from pathlib import Path

from experiment_runner import TRACKED_SOURCE_FILES, execute_experiment, resolve_experiment_config


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_fake_repo(repo_root):
    for relative_path in TRACKED_SOURCE_FILES:
        write_text(repo_root / relative_path, f"# placeholder for {relative_path}\n")


def build_fake_runner(repo_root):
    def fake_runner(label, command, cwd):
        loto_type = command[command.index("--loto_type") + 1]
        if label == "refresh_data":
            write_text(repo_root / "data" / f"{loto_type}_raw.csv", "draw_id,date,num1\n1,2024/01/01,1\n")
            write_text(repo_root / "data" / f"{loto_type}_processed.csv", "draw_id,date,num1,sum_val\n1,2024/01/01,1,1\n")
        elif label == "train":
            write_text(repo_root / "data" / f"{loto_type}_feature_cols.json", json.dumps(["num1", "sum_val"]))
            write_text(repo_root / "models" / f"{loto_type}_feature_cols.json", json.dumps(["num1", "sum_val"]))
            write_text(repo_root / "models" / f"{loto_type}_prob.keras", "dummy model\n")
            write_text(repo_root / "models" / f"{loto_type}_scaler.pkl", "dummy scaler\n")
            write_text(
                repo_root / "data" / f"eval_report_{loto_type}.json",
                json.dumps({"bundle_id": "bundle-demo", "artifact_schema_version": 3}),
            )
            write_text(
                repo_root / "data" / f"prediction_history_{loto_type}.json",
                json.dumps({"bundle_id": "bundle-demo", "artifact_schema_version": 3}),
            )
            write_text(
                repo_root / "data" / f"manifest_{loto_type}.json",
                json.dumps(
                    {
                        "bundle_id": "bundle-demo",
                        "generated_at": "2026-03-20T00:00:00+00:00",
                        "artifact_schema_version": 3,
                        "data_fingerprint": {"data_hash": "abc123"},
                        "metrics_summary": {"final_artifact_status": "reused_existing_artifacts"},
                    }
                ),
            )
        return {"label": label, "command": command, "returncode": 0}

    return fake_runner


def test_execute_experiment_creates_run_directory_and_snapshots_artifacts(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    build_fake_repo(repo_root)

    resolved_config = resolve_experiment_config(
        {
            "loto_type": "loto6",
            "preset": "smoke",
            "seed": 7,
            "refresh_data": True,
            "skip_final_train": True,
        }
    )
    run_dir, summary = execute_experiment(
        resolved_config=resolved_config,
        repo_root=repo_root,
        run_root="runs",
        run_name="demo-run",
        command_runner=build_fake_runner(repo_root),
        requested_config={"loto_type": "loto6", "preset": "smoke"},
    )

    assert summary["status"] == "succeeded"
    assert [step["label"] for step in summary["steps"]] == ["refresh_data", "train"]
    assert (run_dir / "config" / "requested_config.json").exists()
    assert (run_dir / "config" / "resolved_config.json").exists()
    assert (run_dir / "source_hashes.json").exists()
    assert (run_dir / "run_summary.json").exists()
    assert (run_dir / "artifacts" / "data" / "manifest_loto6.json").exists()
    assert (run_dir / "artifacts" / "models" / "loto6_prob.keras").exists()

    saved_summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert saved_summary["manifest_summary"]["bundle_id"] == "bundle-demo"
    assert saved_summary["manifest_summary"]["data_hash"] == "abc123"
