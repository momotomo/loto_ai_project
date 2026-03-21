import argparse
import json
import os
import subprocess
import sys

from calibration_utils import (
    CALIBRATION_METHOD_CHOICES,
    DEFAULT_EVALUATION_CALIBRATION_METHODS,
    DEFAULT_SAVED_CALIBRATION_METHOD,
    NO_CALIBRATION_METHOD,
)
from config import LOTO_CONFIG
from model_variants import DEFAULT_MODEL_VARIANT, MODEL_VARIANT_CHOICES

# --- Macフリーズ対策 ---
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def parse_args():
    parser = argparse.ArgumentParser(description="Run data refresh, training, and a quick prediction smoke test.")
    parser.add_argument("--loto_type", choices=sorted(LOTO_CONFIG.keys()), help="対象の宝くじ種類を1つに絞る")
    parser.add_argument("--train_preset", choices=["default", "fast", "smoke", "archcomp"], default="default", help="train_prob_model.py に渡す preset")
    parser.add_argument("--model_variant", choices=sorted(MODEL_VARIANT_CHOICES), default=DEFAULT_MODEL_VARIANT, help="保存する本番 artifact の model variant")
    parser.add_argument("--evaluation_model_variants", help="評価対象 variant をカンマ区切りで指定 (例: legacy,multihot,deepsets)")
    parser.add_argument(
        "--saved_calibration_method",
        choices=sorted(CALIBRATION_METHOD_CHOICES),
        default=DEFAULT_SAVED_CALIBRATION_METHOD,
        help="保存する本番 artifact の calibration method",
    )
    parser.add_argument(
        "--evaluation_calibration_methods",
        default=DEFAULT_EVALUATION_CALIBRATION_METHODS,
        help="評価対象 calibration method をカンマ区切りで指定 (例: none,temperature,isotonic)",
    )
    parser.add_argument("--skip_final_train", action="store_true", help="train_prob_model.py の final 学習を省略する")
    parser.add_argument("--skip_data_refresh", action="store_true", help="data_collector.py を実行せず、既存 data/*.csv を使う")
    parser.add_argument("--seed", type=int, default=42, help="train_prob_model.py に渡す乱数 seed")
    return parser.parse_args()


def build_command(script_name, loto_type=None):
    command = [sys.executable, script_name]
    if loto_type:
        command.extend(["--loto_type", loto_type])
    return command


def build_train_command(args):
    command = build_command("train_prob_model.py", args.loto_type)
    command.extend(["--preset", args.train_preset])
    command.extend(["--model_variant", args.model_variant])
    command.extend(["--saved_calibration_method", args.saved_calibration_method])
    command.extend(["--evaluation_calibration_methods", args.evaluation_calibration_methods])
    command.extend(["--seed", str(args.seed)])
    if args.evaluation_model_variants:
        command.extend(["--evaluation_model_variants", args.evaluation_model_variants])
    if args.skip_final_train:
        command.append("--skip_final_train")
    return command


def run_step(step_name, command):
    print(step_name)
    result = subprocess.run(command)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def verify_artifacts(loto_type=None):
    loto_types = [loto_type] if loto_type else list(LOTO_CONFIG.keys())
    missing = []

    for current_loto_type in loto_types:
        eval_report_path = os.path.join("data", f"eval_report_{current_loto_type}.json")
        manifest_path = os.path.join("data", f"manifest_{current_loto_type}.json")
        prediction_history_path = os.path.join("data", f"prediction_history_{current_loto_type}.json")
        if not os.path.exists(eval_report_path):
            missing.append(eval_report_path)
        if not os.path.exists(manifest_path):
            missing.append(manifest_path)
        if not os.path.exists(prediction_history_path):
            missing.append(prediction_history_path)
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            calibration_payload = manifest.get("calibration") or {}
            training_context = manifest.get("training_context") or {}
            saved_calibration_method = (
                calibration_payload.get("saved_method")
                or training_context.get("saved_calibration_method")
                or NO_CALIBRATION_METHOD
            )
            if saved_calibration_method != NO_CALIBRATION_METHOD:
                calibrator_candidates = [
                    os.path.join("data", f"{current_loto_type}_calibrator.json"),
                    os.path.join("models", f"{current_loto_type}_calibrator.json"),
                ]
                if not any(os.path.exists(path) for path in calibrator_candidates):
                    missing.append(" or ".join(calibrator_candidates))

    if missing:
        print("❌ 学習完了後の成果物が不足しています。")
        for path in missing:
            print(f" - {path}")
        raise SystemExit(1)


def main():
    args = parse_args()
    os.makedirs("data", exist_ok=True)

    print("\n=======================================================")
    print(" 🚀 宝くじAI 全自動アップデートシステム開始 (確率モデル版)")
    print("=======================================================\n")

    if args.skip_data_refresh:
        print("📡 [1/3] データ更新をスキップし、既存 data/*.csv を使用します...")
    else:
        run_step(
            "📡 [1/3] Webから最新の過去データをダウンロード中...",
            build_command("data_collector.py", args.loto_type),
        )

    run_step(
        "\n🧠 [2/3] 確率モデル(LSTM)の評価と本学習を実行中... (数分かかります)",
        build_train_command(args),
    )
    verify_artifacts(args.loto_type)

    run_step(
        "\n🔮 [3/3] 最新のAIモデルで推論テストを実行中...",
        build_command("predict.py", args.loto_type),
    )

    print("\n=======================================================")
    print(" ✅ 全ての処理が正常に完了しました。")
    print(" 評価結果は data/eval_report_*.json、manifest は data/manifest_*.json を確認してください。")
    print(" 回別照合用の履歴は data/prediction_history_*.json を確認してください。")
    print(" UI を使う場合は `streamlit run app.py` を実行してください。")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
