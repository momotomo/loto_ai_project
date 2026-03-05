import os
import json
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from config import LOTO_CONFIG, LOOKBACK_WINDOW

# フリーズ回避
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.set_visible_devices([], 'GPU')

DATA_DIR = "data"
MODEL_DIR = "models"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def create_multi_hot(targets, max_num):
    """実数のリストをマルチホットベクトルに変換"""
    vectors = []
    for nums in targets:
        vec = np.zeros(max_num)
        for n in nums:
            if 1 <= n <= max_num:
                vec[int(n)-1] = 1.0
        vectors.append(vec)
    return np.array(vectors, dtype=np.float32)

def calculate_metrics(preds, targets, k):
    """BCE, Brier, Top-k重なりを計算"""
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
    
    # 簡易 Calibration (10 bins)
    calib_bins = []
    bins = np.linspace(0, 1, 11)
    for i in range(10):
        mask = (preds >= bins[i]) & (preds < bins[i+1])
        if np.sum(mask) > 0:
            true_prob = np.mean(targets[mask])
            pred_prob = np.mean(preds[mask])
            calib_bins.append({"bin_range": f"{bins[i]:.1f}-{bins[i+1]:.1f}", "pred_prob": float(pred_prob), "true_prob": float(true_prob), "count": int(np.sum(mask))})
            
    return {"logloss": float(logloss), "brier": float(brier), "mean_overlap_top_k": float(mean_overlap), "overlap_dist": overlap_dist, "calibration": calib_bins}

def get_baselines(train_targets, test_targets, max_num):
    N_test = len(test_targets)
    
    # 1. Uniform
    uniform_preds = np.ones((N_test, max_num)) * (sum(train_targets[0]) / max_num)
    uniform_metrics = calculate_metrics(uniform_preds, test_targets, int(sum(train_targets[0])))
    
    # 2. Frequency (train区間)
    freq_prob = np.mean(train_targets, axis=0)
    freq_preds = np.tile(freq_prob, (N_test, 1))
    freq_metrics = calculate_metrics(freq_preds, test_targets, int(sum(train_targets[0])))
    
    # 3. Gap-based (最後に出てからの期間に基づく簡易スコア)
    gap_preds = np.zeros((N_test, max_num))
    last_seen = np.zeros(max_num)
    for i in range(max_num):
        idx = np.where(train_targets[:, i] == 1)[0]
        last_seen[i] = len(train_targets) - idx[-1] if len(idx) > 0 else len(train_targets)
        
    for i in range(N_test):
        gap_score = 1.0 / (last_seen + 1)
        # 確率スケールに簡易正規化
        gap_preds[i] = gap_score * (sum(train_targets[0]) / np.sum(gap_score))
        
        # 次のステップのために更新
        last_seen += 1
        last_seen[test_targets[i] == 1] = 0
        
    gap_metrics = calculate_metrics(gap_preds, test_targets, int(sum(train_targets[0])))
    
    return {"Uniform": uniform_metrics, "Frequency": freq_metrics, "Gap-based": gap_metrics}

def build_prob_model(input_shape, max_num):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(128, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(64),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(max_num, activation='sigmoid') # 確率ベクトル出力
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

if __name__ == "__main__":
    for ltype, config in LOTO_CONFIG.items():
        print(f"\n--- {ltype.upper()} 確率モデル学習 ＆ Walk-Forward 評価 ---")
        df_path = os.path.join(DATA_DIR, f"{ltype}_processed.csv")
        if not os.path.exists(df_path):
            print("データなし。スキップします。")
            continue
            
        df = pd.read_csv(df_path)
        target_cols = [f"num{i+1}" for i in range(config["pick_count"])]
        
        # 特徴量の整理
        features_df = df.drop(['draw_id', 'date'], axis=1)
        other_cols = [c for c in features_df.columns if c not in target_cols]
        # target_cols も入力特徴量として使うが、リストの順序を記録する
        feature_cols = target_cols + other_cols
        features_df = features_df[feature_cols]
        
        # 正規化
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(features_df)
        
        # X (LOOKBACK_WINDOW), y (Multi-hot)
        X, y_raw = [], []
        for i in range(len(scaled_data) - LOOKBACK_WINDOW):
            X.append(scaled_data[i : i + LOOKBACK_WINDOW])
            y_raw.append(df.iloc[i + LOOKBACK_WINDOW][target_cols].values)
            
        X = np.array(X, dtype=np.float32)
        y = create_multi_hot(y_raw, config["max_num"])
        
        # 時系列分割 (Train 80%, Test 20%)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # --- Walk-forward評価用モデル学習 ---
        print("  [1/3] 検証区間(Test)に対する評価用モデルの学習...")
        val_model = build_prob_model((X.shape[1], X.shape[2]), config["max_num"])
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        val_model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.1, callbacks=[early_stop], verbose=0)
        
        # 評価
        print("  [2/3] メトリクス計算とベースライン比較...")
        preds_test = val_model.predict(X_test, verbose=0)
        model_metrics = calculate_metrics(preds_test, y_test, config["pick_count"])
        baselines = get_baselines(y_train, y_test, config["max_num"])
        
        report = {
            "test_samples": len(y_test),
            "Model (LSTM)": model_metrics,
            "Baselines": baselines
        }
        with open(os.path.join(DATA_DIR, f"eval_report_{ltype}.json"), "w") as f:
            json.dump(report, f, indent=2)
            
        # --- 推論用最終モデル学習 (全データ使用) ---
        print("  [3/3] 全データを用いた本番用モデルの構築と保存...")
        final_model = build_prob_model((X.shape[1], X.shape[2]), config["max_num"])
        # 全データなのでvalidation_splitを切り、Trainのベストエポック数程度回す
        final_model.fit(X, y, epochs=25, batch_size=32, verbose=0)
        
        # 成果物の保存
        final_model.save(os.path.join(MODEL_DIR, f"{ltype}_prob.keras"))
        with open(os.path.join(MODEL_DIR, f"{ltype}_scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)
        with open(os.path.join(MODEL_DIR, f"{ltype}_feature_cols.json"), "w") as f:
            json.dump(feature_cols, f)
            
        print(f"✅ 完了 (LogLoss: {model_metrics['logloss']:.4f}, Top-k重なり平均: {model_metrics['mean_overlap_top_k']:.2f})")