import argparse
import base64
import json
import shutil
import textwrap
import zipfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path


PAYLOAD_PLACEHOLDER = "__KAGGLE_PAYLOAD_BASE64__"
ROOT_FILES = [
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
]
DATA_GLOB = "data/*.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare a self-contained Kaggle kernel build directory.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--kernel-id", required=True, help="username/kernel-slug")
    parser.add_argument("--kernel-title", help="Optional Kaggle kernel title")
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--train-preset", default="fast")
    parser.add_argument("--model-variant", default="legacy")
    parser.add_argument("--evaluation-model-variants", default="legacy,multihot", help="例: legacy,multihot,deepsets")
    parser.add_argument("--saved-calibration-method", default="none")
    parser.add_argument("--evaluation-calibration-methods", default="none,temperature,isotonic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-legacy-holdout", action="store_true")
    return parser.parse_args()


def slug_to_title(kernel_id):
    slug = kernel_id.split("/", 1)[-1]
    return slug.replace("-", " ").replace("_", " ").title()


def reset_build_dir(build_dir):
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)


def build_metadata(kernel_id, kernel_title):
    return {
        "id": kernel_id,
        "title": kernel_title,
        "code_file": "script.py",
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
        "model_variant": args.model_variant,
        "evaluation_model_variants": args.evaluation_model_variants,
        "saved_calibration_method": args.saved_calibration_method,
        "evaluation_calibration_methods": args.evaluation_calibration_methods,
        "seed": args.seed,
        "skip_legacy_holdout": args.skip_legacy_holdout,
    }


def collect_payload_sources(repo_root):
    sources = []
    for relative_path in ROOT_FILES:
        source = repo_root / relative_path
        if not source.exists():
            raise SystemExit(f"Missing required source file: {source}")
        sources.append(source)

    for csv_path in sorted(repo_root.glob(DATA_GLOB)):
        sources.append(csv_path)

    return sources


def build_payload_bytes(repo_root, run_config):
    payload_sources = collect_payload_sources(repo_root)
    buffer = BytesIO()

    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in payload_sources:
            archive.write(source, arcname=source.relative_to(repo_root).as_posix())
        archive.writestr("run_config.json", json.dumps(run_config, ensure_ascii=False, indent=2) + "\n")

    return buffer.getvalue(), payload_sources


def render_payload_text(payload_bytes):
    encoded = base64.b64encode(payload_bytes).decode("ascii")
    return "\n".join(textwrap.wrap(encoded, 120))


def render_script(repo_root, payload_text):
    template_path = repo_root / "scripts" / "kaggle_entry.py"
    template = template_path.read_text(encoding="utf-8")
    if PAYLOAD_PLACEHOLDER not in template:
        raise SystemExit(f"Payload placeholder not found in template: {template_path}")
    return template.replace(PAYLOAD_PLACEHOLDER, payload_text, 1)


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main():
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    build_dir = Path(args.build_dir).resolve()

    reset_build_dir(build_dir)

    run_config = build_run_config(args)
    payload_bytes, payload_sources = build_payload_bytes(repo_root, run_config)
    script_text = render_script(repo_root, render_payload_text(payload_bytes))
    metadata = build_metadata(args.kernel_id, args.kernel_title or slug_to_title(args.kernel_id))

    (build_dir / "script.py").write_text(script_text, encoding="utf-8")
    write_json(build_dir / "kernel-metadata.json", metadata)
    write_json(build_dir / "run_config.json", run_config)

    print(f"Prepared Kaggle kernel directory: {build_dir}")
    print(f"Embedded payload files: {[path.relative_to(repo_root).as_posix() for path in payload_sources]}")
    print(f"Embedded payload size_bytes={len(payload_bytes)}")
    print(f"Generated script size_bytes={(build_dir / 'script.py').stat().st_size}")


if __name__ == "__main__":
    main()
