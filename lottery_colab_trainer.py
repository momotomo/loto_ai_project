import os
import json
import pickle
import pandas as pd
import numpy as np
import requests
import io
import time
from datetime import datetime

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ==========================================
# 1. 設定・ディレクトリ準備
# ==========================================
LOTO_TYPES = {
    "miniloto": {"max_num": 31, "pick_count": 5},
    "loto6": {"max_num": 43, "pick_count": 6},
    "loto7": {"max_num": 37, "pick_count": 7}
}

# Kaggle/Colab両対応の出力先
WORK_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
DATA_DIR = os.path.join(WORK_DIR, "data")
MODEL_DIR = os.path.join(WORK_DIR, "models")

LOOKBACK_WINDOW = 10
EPOCHS = 80
BATCH_SIZE = 32

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43}

# ==========================================
# 2. 関数群
# ==========================================
def calculate_advanced_features(df, num_cols):
    df["sum_val"] = df[num_cols].sum(axis=1)
    df["odd_count"] = df[num_cols].apply(lambda row: sum(1 for x in row if x % 2 != 0), axis=1)
    df["even_count"] = len(num_cols) - df["odd_count"]
    
    def count_consecutive(row):
        nums = sorted(row)
        return sum(1 for i in range(len(nums)-1) if nums[i+1] - nums[i] == 1)
    df["consecutive_pairs"] = df[num_cols].apply(count_consecutive, axis=1)
    df["max_min_diff"] = df[num_cols].max(axis=1) - df[num_cols].min(axis=1)
    df["last_digit_sum"] = df[num_cols].apply(lambda row: sum(x % 10 for x in row), axis=1)
    df["sum_ma5"] = df["sum_val"].rolling(window=5).mean().fillna(df["sum_val"])
    return df

def fetch_and_process(loto_type):
    print(f"\n🌐 [{loto_type.upper()}] データをWebから取得中...")
    url = f"https://{loto_type}.thekyo.jp/data/{loto_type}.csv"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers)
        csv_text = response.content.decode('shift_jis', errors='ignore')
        df_raw = pd.read_csv(io.StringIO(csv_text), header=None, on_bad_lines='skip')
        df_raw = df_raw[pd.to_numeric(df_raw[0], errors='coerce').notnull()]
        
        pick_count = LOTO_TYPES[loto_type]["pick_count"]
        df = df_raw.iloc[:, :pick_count+2].copy()
        cols = ["draw_id", "date"] + [f"num{i+1}" for i in range(pick_count)]
        df.columns = cols
            
        num_cols = [c for c in df.columns if c.startswith('num')]
        df[num_cols] = df[num_cols].astype(int)
        df["draw_id"] = df["draw_id"].astype(int)
        df = df.sort_values("draw_id").reset_index(drop=True)
        
        df = calculate_advanced_features(df, num_cols)
        path = os.path.join(DATA_DIR, f"{loto_type}_processed.csv")
        df.to_csv(path, index=False)
        return df, num_cols
    except Exception as e:
        print(f"❌ エラー: {e}")
        return None, None

def create_multi_hot(targets, max_num):
    vectors = []
    for nums in targets:
        vec = np.zeros(max_num)
        for n in nums:
            if 1 <= n <= max_num:
                vec[int(n)-1] = 1.0
        vectors.append(vec)
    return np.array(vectors, dtype=np.float32)

def calculate_metrics(preds, targets, k):
    preds_clip = np.clip(preds, 1e-9, 1 - 1e-9)
    logloss = -np.mean(targets * np.log(preds_clip) + (1 - targets) * np.log(1 - preds_clip))
    brier = np.mean((preds - targets) ** 2)
    
    top_k_preds = np.argsort(preds, axis=1)[:, -k:]
    overlaps = []
    for p_idx, t_vec in zip(top_k_preds, targets):
        t_idx = np.where(t_vec == 1.0)[0]
        overlaps.append(len(set(p_idx) & set(t_idx)))
        
    overlap_dist = {str(i): overlaps.count(i) for i in range(k+1)}
    mean_overlap = np.mean(overlaps)
    
    calib_bins = []
    bins = np.linspace(0, 1, 11)
    for i in range(10):
        mask = (preds >= bins[i]) & (preds < bins[i+1])
        if np.sum(mask) > 0:
            calib_bins.append({
                "bin_range": f"{bins[i]:.1f}-{bins[i+1]:.1f}",
                "pred_prob": float(np.mean(preds[mask])),
                "true_prob": float(np.mean(targets[mask])),
                "count": int(np.sum(mask))
            })
    return {"logloss": float(logloss), "brier": float(brier), "mean_overlap_top_k": float(mean_overlap), "overlap_dist": overlap_dist, "calibration": calib_bins}

def get_baselines(train_targets, test_targets, max_num, pick_count):
    N_test = len(test_targets)
    # Uniform
    uniform_preds = np.ones((N_test, max_num)) * (pick_count / max_num)
    uniform_metrics = calculate_metrics(uniform_preds, test_targets, pick_count)
    # Frequency
    freq_prob = np.mean(train_targets, axis=0)
    freq_preds = np.tile(freq_prob, (N_test, 1))
    freq_metrics = calculate_metrics(freq_preds, test_targets, pick_count)
    
    return {"Uniform": uniform_metrics, "Frequency": freq_metrics}

def build_prob_model(input_shape, max_num):
    # 確率ベクトル出力モデル (Sigmoid & Binary Crossentropy)
    model = Sequential([
        Input(shape=input_shape),
        LSTM(128, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(64),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(max_num, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

# ==========================================
# 3. メイン処理 (学習と評価)
# ==========================================
for ltype, config in LOTO_TYPES.items():
    df, target_cols = fetch_and_process(ltype)
    if df is None: continue
    
    features_df = df.drop(['draw_id', 'date'], axis=1)
    other_cols = [c for c in features_df.columns if c not in target_cols]
    feature_cols = target_cols + other_cols
    features_df = features_df[feature_cols]
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(features_df)
    
    X, y_raw = [], []
    for i in range(len(scaled_data) - LOOKBACK_WINDOW):
        X.append(scaled_data[i : i + LOOKBACK_WINDOW])
        y_raw.append(df.iloc[i + LOOKBACK_WINDOW][target_cols].values)
        
    X = np.array(X, dtype=np.float32)
    y = create_multi_hot(y_raw, config["max_num"])
    
    # --- Walk-forward 評価 (Train 80% / Test 20%) ---
    print(f"🧠 [{ltype.upper()}] モデル評価(Walk-forward)を開始...")
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    val_model = build_prob_model((X.shape[1], X.shape[2]), config["max_num"])
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    val_model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, callbacks=[early_stop], verbose=0)
    
    preds_test = val_model.predict(X_test, verbose=0)
    model_metrics = calculate_metrics(preds_test, y_test, config["pick_count"])
    baselines = get_baselines(y_train, y_test, config["max_num"], config["pick_count"])
    
    report = {
        "test_samples": len(y_test),
        "Model (LSTM)": model_metrics,
        "Baselines": baselines
    }
    with open(os.path.join(DATA_DIR, f"eval_report_{ltype}.json"), "w") as f:
        json.dump(report, f, indent=2)
        
    # --- 本番用モデルの構築 (全データ使用) ---
    print(f"⚙️ [{ltype.upper()}] 本番用モデルを学習中...")
    final_model = build_prob_model((X.shape[1], X.shape[2]), config["max_num"])
    final_model.fit(X, y, epochs=30, batch_size=BATCH_SIZE, verbose=0)
    
    # 成果物の保存
    final_model.save(os.path.join(MODEL_DIR, f"{ltype}_prob.keras"))
    with open(os.path.join(MODEL_DIR, f"{ltype}_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODEL_DIR, f"{ltype}_feature_cols.json"), "w") as f:
        json.dump(feature_cols, f)

print(f"\n--- 全工程完了。ファイルは {WORK_DIR} に保存されました ---")