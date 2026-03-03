import os
import sys
import json
import subprocess
from datetime import datetime
import pandas as pd
import numpy as np

# --- Macフリーズ対策 ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

LOTO_CONFIG = {
    "miniloto": {"name": "ミニロト", "max_num": 31, "pick_count": 5, "sum_range": [60, 100], "max_parity_diff": 3},
    "loto6": {"name": "ロト6", "max_num": 43, "pick_count": 6, "sum_range": [115, 150], "max_parity_diff": 2},
    "loto7": {"name": "ロト7", "max_num": 37, "pick_count": 7, "sum_range": [110, 155], "max_parity_diff": 3}
}
LOOKBACK_WINDOW = 10
PREDICTION_FILE = "data/latest_predictions.json"

def generate_candidates(base_pred, config, num_candidates=500):
    candidates = []
    for _ in range(num_candidates):
        noise = np.random.uniform(-2.5, 2.5, size=config["pick_count"])
        candidate = np.round(base_pred + noise).astype(int)
        candidate = np.clip(candidate, 1, config["max_num"])
        unique_candidate = []
        for num in candidate:
            while num in unique_candidate or num < 1 or num > config["max_num"]:
                num = np.random.randint(1, config["max_num"] + 1)
            unique_candidate.append(num)
        unique_candidate.sort()
        candidates.append(unique_candidate)
    return candidates

def apply_statistical_filters(candidates, config):
    valid_candidates = []
    for nums in candidates:
        if not (config["sum_range"][0] <= sum(nums) <= config["sum_range"][1]):
            continue
        odd_count = sum(1 for n in nums if n % 2 != 0)
        even_count = config["pick_count"] - odd_count
        if abs(odd_count - even_count) > config["max_parity_diff"]:
            continue
        valid_candidates.append(nums)
    return valid_candidates

if __name__ == "__main__":
    print("\n=======================================================")
    print(" 🚀 宝くじAI 全自動アップデートシステム開始")
    print("=======================================================\n")
    
    os.makedirs("data", exist_ok=True)
    python_exe = sys.executable

    # 1. データ収集の実行
    print("📡 [1/3] Webから最新の過去データをダウンロード中...")
    subprocess.run([python_exe, "data_collector.py"])

    # 2. AIモデルの再学習の実行
    print("\n🧠 [2/3] AIモデルを再学習中... (数分かかります)")
    subprocess.run([python_exe, "train_model.py"])

    # 3. 最新モデルを使った予測の生成と保存
    print("\n🔮 [3/3] 最新のAIモデルで全種類の次回予測を計算中...")
    all_predictions = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    
    for loto_type in ["miniloto", "loto6", "loto7"]:
        data_file = f"data/{loto_type}_processed.csv"
        model_path = f"models/{loto_type}_lstm_best.keras"
        
        if not os.path.exists(data_file) or not os.path.exists(model_path):
            continue
            
        df = pd.read_csv(data_file)
        model = load_model(model_path)
        config = LOTO_CONFIG[loto_type]
        
        target_cols = [f"num{i+1}" for i in range(config["pick_count"])]
        features_df = df.drop(['draw_id', 'date'], axis=1)
        other_cols = [c for c in features_df.columns if c not in target_cols]
        features_df = features_df[target_cols + other_cols]
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(features_df)
        
        recent_data = scaled_data[-LOOKBACK_WINDOW:]
        X_input = np.array([recent_data], dtype=np.float32)
        
        X_input_tf = tf.convert_to_tensor(X_input, dtype=tf.float32)
        scaled_pred = model(X_input_tf, training=False).numpy()
        
        dummy_array = np.zeros((1, len(features_df.columns)))
        dummy_array[0, :config["pick_count"]] = scaled_pred[0]
        unscaled_pred = scaler.inverse_transform(dummy_array)[0, :config["pick_count"]]
        
        candidates = generate_candidates(unscaled_pred, config)
        valid_candidates = apply_statistical_filters(candidates, config)
        
        top_matches_int = []
        if valid_candidates:
            seen = set()
            unique_cands = []
            for c in valid_candidates:
                t = tuple(c)
                if t not in seen:
                    seen.add(t)
                    unique_cands.append(c)
            sorted_cands = sorted(unique_cands, key=lambda x: np.sum(np.abs(np.array(x) - unscaled_pred)))
            top_matches_int = [[int(n) for n in match] for match in sorted_cands[:5]]
            
        all_predictions[loto_type] = {"candidates": top_matches_int}
        print(f"  └ {config['name']} の予測生成完了")
        
    with open(PREDICTION_FILE, "w", encoding="utf-8") as f:
        json.dump(all_predictions, f, ensure_ascii=False, indent=2)

    print("\n=======================================================")
    print(" ✅ 全ての処理が正常に完了しました！")
    print(" Web画面 (app.py) を開いて、最新の予測結果を確認してください。")
    print("=======================================================\n")