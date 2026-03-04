import os
import pandas as pd
import numpy as np
import requests
import io
import time
from datetime import datetime

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import warnings

# CPUでの実行を最適化・安定化させる設定
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.set_visible_devices([], 'GPU') # 明示的にGPUを無効化（CPUのみを使用）

# ==========================================
# 1. 設定・Kaggle用ディレクトリ準備
# ==========================================
LOTO_TYPES = ["miniloto", "loto6", "loto7"]

# Kaggleの作業用ディレクトリ (実行完了後にOutputとして自動保存されます)
WORK_DIR = "/kaggle/working"
DATA_DIR = os.path.join(WORK_DIR, "data")
MODEL_DIR = os.path.join(WORK_DIR, "models")

LOOKBACK_WINDOW = 10
EPOCHS = 80
BATCH_SIZE = 32

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# 素数リスト
PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43}

# ==========================================
# 2. 特徴量エンジニアリング (ロト特有指標)
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
    df["mean_deviation"] = df["sum_val"] / len(num_cols)
    df["sum_ma5"] = df["sum_val"].rolling(window=5).mean().fillna(df["sum_val"])
    df["sum_diff"] = df["sum_val"].diff().fillna(0)
    df["std_dev"] = df[num_cols].std(axis=1).fillna(0)
    
    df["zone1_cnt"] = df[num_cols].apply(lambda row: sum(1 for x in row if 1 <= x <= 10), axis=1)
    df["zone2_cnt"] = df[num_cols].apply(lambda row: sum(1 for x in row if 11 <= x <= 20), axis=1)
    df["zone3_cnt"] = df[num_cols].apply(lambda row: sum(1 for x in row if 21 <= x <= 30), axis=1)
    df["zone4_cnt"] = df[num_cols].apply(lambda row: sum(1 for x in row if 31 <= x <= 43), axis=1)

    df["prime_count"] = df[num_cols].apply(lambda row: sum(1 for x in row if x in PRIMES), axis=1)
    
    def count_same_last_digit(row):
        last_digits = [x % 10 for x in row]
        unique_digits = len(set(last_digits))
        return len(last_digits) - unique_digits
    df["same_last_digit"] = df[num_cols].apply(count_same_last_digit, axis=1)

    overlap_prev = [0]
    slide_prev = [0]
    
    nums_array = df[num_cols].values
    for i in range(1, len(df)):
        prev_nums = set(nums_array[i-1])
        curr_nums = set(nums_array[i])
        
        overlap_cnt = len(prev_nums & curr_nums)
        overlap_prev.append(overlap_cnt)
        
        slide_nums = set()
        for n in prev_nums:
            slide_nums.add(n - 1)
            slide_nums.add(n + 1)
        slide_cnt = len(slide_nums & curr_nums)
        slide_prev.append(slide_cnt)
        
    df["overlap_prev"] = overlap_prev
    df["slide_prev"] = slide_prev

    hot_scores = []
    window = 10
    past_draws = df[num_cols].values
    for i in range(len(df)):
        if i < window:
            hot_scores.append(0)
        else:
            past_window_nums = past_draws[i-window:i].flatten()
            current_nums = past_draws[i]
            score = sum((past_window_nums == n).sum() for n in current_nums)
            hot_scores.append(score)
    df["hot_score"] = hot_scores

    return df

# ==========================================
# 3. データ収集・前処理
# ==========================================
def fetch_and_process(loto_type):
    print(f"\n🌐 [{loto_type.upper()}] データをWebから取得中...")
    url = f"https://{loto_type}.thekyo.jp/data/{loto_type}.csv"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        response = requests.get(url, headers=headers)
        csv_text = response.content.decode('shift_jis', errors='ignore')
        df_raw = pd.read_csv(io.StringIO(csv_text), header=None, on_bad_lines='skip')
        df_raw = df_raw[pd.to_numeric(df_raw[0], errors='coerce').notnull()]
        
        if loto_type == "miniloto":
            df = df_raw.iloc[:, :7].copy()
            df.columns = ["draw_id", "date", "num1", "num2", "num3", "num4", "num5"]
        elif loto_type == "loto6":
            df = df_raw.iloc[:, :8].copy()
            df.columns = ["draw_id", "date", "num1", "num2", "num3", "num4", "num5", "num6"]
        elif loto_type == "loto7":
            df = df_raw.iloc[:, :9].copy()
            df.columns = ["draw_id", "date", "num1", "num2", "num3", "num4", "num5", "num6", "num7"]
            
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

def create_dataset(data, lookback, target_count):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:(i + lookback), :])
        y.append(data[i + lookback, :target_count])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# ==========================================
# 4. DAE + LSTM ハイブリッドモデルの構築
# ==========================================
def train_dae_and_lstm(X, y, output_units, model_path):
    features = X.shape[2]
    lookback = X.shape[1]

    # --- [STEP 1] DAE (ノイズ除去) ---
    print("   -> [STEP 1] DAE (ノイズ除去) 事前学習を開始...")
    start_dae = time.time()
    
    dae_inputs = Input(shape=(lookback, features))
    
    encoded = TimeDistributed(Dense(128, activation='relu'))(dae_inputs)
    encoded = TimeDistributed(BatchNormalization())(encoded)
    encoded_compressed = TimeDistributed(Dense(64, activation='relu'))(encoded)
    
    decoded = TimeDistributed(Dense(128, activation='relu'))(encoded_compressed)
    decoded = TimeDistributed(Dense(features, activation='linear'))(decoded)

    dae = Model(dae_inputs, decoded)
    dae.compile(optimizer='adam', loss='mse')

    noise_factor = 0.1
    X_noisy = X + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X.shape)
    X_noisy = np.clip(X_noisy, 0., 1.)

    dae_early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    dae.fit(X_noisy, X, epochs=30, batch_size=32, validation_split=0.2, callbacks=[dae_early_stop], verbose=0)
    
    elapsed_dae = time.time() - start_dae
    print(f"   -> [STEP 1] 完了 ({elapsed_dae:.1f} 秒)")

    # --- [STEP 2] LSTM 本学習 ---
    print("   -> [STEP 2] LSTM 本学習を開始...")
    start_lstm = time.time()
    
    encoder = Model(dae_inputs, encoded_compressed)
    encoder.trainable = True

    lstm_inputs = Input(shape=(lookback, features))
    clean_features = encoder(lstm_inputs)

    x = LSTM(128, return_sequences=True)(clean_features)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = LSTM(64)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    outputs = Dense(output_units)(x)

    full_model = Model(lstm_inputs, outputs)
    full_model.compile(optimizer='adam', loss='mae')

    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=0)
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    
    full_model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2, 
                   callbacks=[checkpoint, early_stop], verbose=0)
    
    elapsed_lstm = time.time() - start_lstm
    print(f"   -> [STEP 2] 完了 ({elapsed_lstm:.1f} 秒)")
    
    return full_model

# ==========================================
# 5. メイン実行ループ
# ==========================================
total_start_time = time.time()
print(f"🚀 スクリプト実行開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

for ltype in LOTO_TYPES:
    df, target_cols = fetch_and_process(ltype)
    if df is None: continue
    
    features_df = df.drop(['draw_id', 'date'], axis=1)
    other_cols = [c for c in features_df.columns if c not in target_cols]
    features_df = features_df[target_cols + other_cols]
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(features_df)
    
    X, y = create_dataset(scaled_data, LOOKBACK_WINDOW, len(target_cols))
    
    print(f"🧠 [{ltype.upper()}] DAE+LSTMモデルの構築と学習を開始します... (データ件数: {len(X)})")
    model_path = os.path.join(MODEL_DIR, f"{ltype}_lstm_best.keras")
    
    model = train_dae_and_lstm(X, y, len(target_cols), model_path)
    print(f"✅ [{ltype.upper()}] 全学習プロセス完了")

total_elapsed = (time.time() - total_start_time) / 60
print(f"\n=======================================================")
print(f"🎉 全工程完了！ (総所要時間: 約 {total_elapsed:.1f} 分)")
print(f"保存先: {WORK_DIR}")
print(f"=======================================================")