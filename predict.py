import os
import json
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import warnings
from config import LOTO_CONFIG, LOOKBACK_WINDOW, generate_valid_sample

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.set_visible_devices([], 'GPU')

DATA_DIR = "data"
MODEL_DIR = "models"

if __name__ == "__main__":
    print("\n" + "="*60)
    print(" 🎯 宝くじ AI確率予測システム - CLI")
    print("="*60 + "\n")

    for ltype, config in LOTO_CONFIG.items():
        model_path = os.path.join(MODEL_DIR, f"{ltype}_prob.keras")
        scaler_path = os.path.join(MODEL_DIR, f"{ltype}_scaler.pkl")
        cols_path = os.path.join(MODEL_DIR, f"{ltype}_feature_cols.json")
        data_path = os.path.join(DATA_DIR, f"{ltype}_processed.csv")
        
        if not all(os.path.exists(p) for p in [model_path, scaler_path, cols_path, data_path]):
            print(f"[{ltype.upper()}] 必要なファイルが不足しています。スキップします。")
            continue

        df = pd.read_csv(data_path)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        with open(cols_path, "r") as f:
            feature_cols = json.load(f)
            
        # データの準備
        features_df = df[feature_cols] # 学習時と同じ列順序を保証
        scaled_data = scaler.transform(features_df)
        recent_input = np.array([scaled_data[-LOOKBACK_WINDOW:]], dtype=np.float32)

        # モデル推論
        model = load_model(model_path, compile=False)
        probs = model(tf.convert_to_tensor(recent_input), training=False).numpy()[0]
        
        print(f"\n--- {config['name']} の予測結果 ---")
        
        # 1. 確率の高い上位数字の表示
        top_idx = np.argsort(probs)[::-1]
        print("🔝 確率上位ランキング:")
        for i in range(5):
            print(f"   第{i+1}位: {top_idx[i]+1:02d} ({probs[top_idx[i]]*100:.1f}%)")
            
        # 2. サンプリングによる買い目生成 (心理逆張り＆統計フィルタ適用)
        print("\n🎰 推奨買い目 (重み付き抽出 + 期待値最大化フィルタ適用):")
        for i in range(3): # 3口生成
            cand = generate_valid_sample(probs, config, use_psychological=True, use_statistical=True, sampling_mode="weighted")
            cand_str = ", ".join(f"{n:02d}" for n in cand)
            print(f"   口{i+1}: [{cand_str}] (合計:{sum(cand)})")
            
    print("\n" + "="*60)