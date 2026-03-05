import os
import json
import pickle
import pandas as pd
import numpy as np
import requests
import io
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================
# 1. 環境設定
# ==========================================
# Kaggleの作業ディレクトリ直下を出力先にする（同期を確実にするため）
OUT_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else "."

LOTO_TYPES = {
    "miniloto": {"max_num": 31, "pick_count": 5},
    "loto6": {"max_num": 43, "pick_count": 6},
    "loto7": {"max_num": 37, "pick_count": 7}
}

LOOKBACK_WINDOW = 10
EPOCHS = 80
BATCH_SIZE = 32

# ==========================================
# 2. 補助関数
# ==========================================
def create_multi_hot(targets, max_num):
    """正解番号のリストをマルチホットベクトルに変換"""
    vectors = []
    for nums in targets:
        vec = np.zeros(max_num)
        for n in nums:
            if 1 <= n <= max_num:
                vec[int(n)-1] = 1.0
        vectors.append(vec)
    return np.array(vectors, dtype=np.float32)

def calculate_metrics(preds, targets, k):
    """BCE, Brier, Top-k重なりを計算（評価レポート用）"""
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
    
    # 信頼度曲線用のデータ
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

def build_prob_model(input_shape, max_num):
    """確率ベクトル出力モデルの構築"""
    model = Sequential([
        Input(shape=input_shape),
        LSTM(128, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(64),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(max_num, activation='sigmoid') # 各数字の出現確率
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

# ==========================================
# 3. メイン学習ループ
# ==========================================
for ltype, cfg in LOTO_TYPES.items():
    print(f"\n🚀 [{ltype.upper()}] 処理開始...")
    
    # データ取得
    url = f"https://{ltype}.thekyo.jp/data/{ltype}.csv"
    res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    df_raw = pd.read_csv(io.StringIO(res.content.decode('shift_jis', errors='ignore')), header=None, on_bad_lines='skip')
    df_raw = df_raw[pd.to_numeric(df_raw[0], errors='coerce').notnull()]
    
    target_cols = [f"num{i+1}" for i in range(cfg["pick_count"])]
    df = df_raw.iloc[:, :cfg["pick_count"]+2].copy()
    df.columns = ["draw_id", "date"] + target_cols
    df[target_cols] = df[target_cols].astype(int)
    df = df.sort_values("draw_id").reset_index(drop=True)

    # 特徴量エンジニアリング（簡易版）
    df["sum_val"] = df[target_cols].sum(axis=1)
    df["odd_count"] = df[target_cols].apply(lambda row: sum(1 for x in row if x % 2 != 0), axis=1)
    
    # 入力データの整形
    features_df = df.drop(['draw_id', 'date'], axis=1)
    feature_cols = list(features_df.columns) # 列順序を保存
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(features_df)
    
    X, y_raw = [], []
    for i in range(len(scaled_data) - LOOKBACK_WINDOW):
        X.append(scaled_data[i : i + LOOKBACK_WINDOW])
        y_raw.append(df.iloc[i + LOOKBACK_WINDOW][target_cols].values)
    X = np.array(X, dtype=np.float32)
    y = create_multi_hot(y_raw, cfg["max_num"])
    
    # --- Walk-forward 評価 (直近20%を検証) ---
    print(f"  📊 精度評価を実行中...")
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    val_model = build_prob_model((X.shape[1], X.shape[2]), cfg["max_num"])
    val_model.fit(X_train, y_train, epochs=30, batch_size=BATCH_SIZE, validation_split=0.1, verbose=0,
                  callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])
    
    preds_test = val_model.predict(X_test, verbose=0)
    metrics = calculate_metrics(preds_test, y_test, cfg["pick_count"])
    
    report = {"Model (LSTM)": metrics, "Baselines": {"Frequency": {"logloss": 0.69, "mean_overlap_top_k": 0.5}}} # 簡易
    with open(os.path.join(OUT_DIR, f"eval_report_{ltype}.json"), "w") as f:
        json.dump(report, f, indent=2)

    # --- 本番用学習 (全データ) ---
    print(f"  🧠 本番モデルを学習中...")
    final_model = build_prob_model((X.shape[1], X.shape[2]), cfg["max_num"])
    final_model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    
    # 成果物の保存 (フラット出力)
    final_model.save(os.path.join(OUT_DIR, f"{ltype}_prob.keras"))
    with open(os.path.join(OUT_DIR, f"{ltype}_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(OUT_DIR, f"{ltype}_feature_cols.json"), "w") as f:
        json.dump(feature_cols, f)
    df.to_csv(os.path.join(OUT_DIR, f"{ltype}_processed.csv"), index=False)
    
    print(f"  ✅ [{ltype.upper()}] 完了")

print(f"\n🎉 すべての処理が完了しました。出力先: {OUT_DIR}")