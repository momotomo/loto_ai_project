import argparse
import os
import json
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import warnings
from config import LOTO_CONFIG, LOOKBACK_WINDOW, generate_valid_sample
from model_variants import build_recent_model_input, resolve_model_variant_from_manifest

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.set_visible_devices([], 'GPU')

DATA_DIR = "data"
MODEL_DIR = "models"


def parse_args():
    parser = argparse.ArgumentParser(description="Run a quick lottery prediction smoke test.")
    parser.add_argument("--loto_type", choices=sorted(LOTO_CONFIG.keys()), help="対象の宝くじ種類を1つに絞る")
    parser.add_argument("--model_variant", choices=["legacy", "multihot"], help="manifest が無い場合の variant 上書き")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    loto_types = [args.loto_type] if args.loto_type else list(LOTO_CONFIG.keys())

    print("\n" + "="*60)
    print(" 🎯 宝くじ AI確率予測システム - CLI")
    print("="*60 + "\n")

    for ltype in loto_types:
        config = LOTO_CONFIG[ltype]
        model_path = os.path.join(MODEL_DIR, f"{ltype}_prob.keras")
        scaler_path = os.path.join(MODEL_DIR, f"{ltype}_scaler.pkl")
        cols_paths = [
            os.path.join(MODEL_DIR, f"{ltype}_feature_cols.json"),
            os.path.join(DATA_DIR, f"{ltype}_feature_cols.json"),
        ]
        data_path = os.path.join(DATA_DIR, f"{ltype}_processed.csv")
        manifest_path = os.path.join(DATA_DIR, f"manifest_{ltype}.json")
        
        if not all(os.path.exists(p) for p in [model_path, scaler_path, data_path]):
            print(f"[{ltype.upper()}] 必要なファイルが不足しています。スキップします。")
            continue

        df = pd.read_csv(data_path)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        feature_cols = None
        for cols_path in cols_paths:
            if os.path.exists(cols_path):
                with open(cols_path, "r") as f:
                    feature_cols = json.load(f)
                break
        if feature_cols is None:
            print(f"[{ltype.upper()}] feature_cols.json が見つかりません。スキップします。")
            continue

        manifest = None
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        model_variant = args.model_variant or resolve_model_variant_from_manifest(manifest)
        recent_input, prepared_dataset = build_recent_model_input(df, ltype, model_variant, scaler)

        # モデル推論
        model = load_model(model_path, compile=False)
        probs = model(tf.convert_to_tensor(recent_input), training=False).numpy()[0]
        
        print(f"\n--- {config['name']} の予測結果 ---")
        print(f"variant: {model_variant} / feature_count: {len(prepared_dataset['feature_cols'])}")
        
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
