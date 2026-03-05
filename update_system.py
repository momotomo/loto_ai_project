import os
import sys
import subprocess
from datetime import datetime

# --- Macフリーズ対策 ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == "__main__":
    print("\n=======================================================")
    print(" 🚀 宝くじAI 全自動アップデートシステム開始 (確率モデル版)")
    print("=======================================================\n")
    
    os.makedirs("data", exist_ok=True)
    python_exe = sys.executable

    # 1. データ収集
    print("📡 [1/3] Webから最新の過去データをダウンロード中...")
    subprocess.run([python_exe, "data_collector.py"])

    # 2. 確率モデルの学習とWalk-forward評価
    print("\n🧠 [2/3] 確率モデル(LSTM)の評価と本学習を実行中... (数分かかります)")
    subprocess.run([python_exe, "train_prob_model.py"])

    # 3. 予測結果の出力確認
    print("\n🔮 [3/3] 最新のAIモデルで推論テストを実行中...")
    subprocess.run([python_exe, "predict.py"])

    print("\n=======================================================")
    print(" ✅ 全ての処理が正常に完了しました！")
    print(" 評価結果を見たい場合は 'python backtest.py' を、")
    print(" UIを利用する場合は 'streamlit run app.py' を実行してください。")
    print("=======================================================\n")