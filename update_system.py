import argparse
import os
import subprocess
import sys

from config import LOTO_CONFIG

# --- Macフリーズ対策 ---
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def parse_args():
    parser = argparse.ArgumentParser(description="Run data refresh, training, and a quick prediction smoke test.")
    parser.add_argument("--loto_type", choices=sorted(LOTO_CONFIG.keys()), help="対象の宝くじ種類を1つに絞る")
    parser.add_argument("--train_preset", choices=["default", "fast", "smoke"], default="default", help="train_prob_model.py に渡す preset")
    parser.add_argument("--skip_final_train", action="store_true", help="train_prob_model.py の final 学習を省略する")
    return parser.parse_args()


def build_command(script_name, loto_type=None):
    command = [sys.executable, script_name]
    if loto_type:
        command.extend(["--loto_type", loto_type])
    return command


def build_train_command(args):
    command = build_command("train_prob_model.py", args.loto_type)
    command.extend(["--preset", args.train_preset])
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
        if not os.path.exists(eval_report_path):
            missing.append(eval_report_path)
        if not os.path.exists(manifest_path):
            missing.append(manifest_path)

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
    print(" UI を使う場合は `streamlit run app.py` を実行してください。")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
