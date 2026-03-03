import os
import pandas as pd
import numpy as np

# --- Mac環境でのフリーズを物理的に回避する設定 ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# --- 設定 ---
LOTO_TYPES = ["miniloto", "loto6", "loto7"]
MODEL_DIR = "models"
LOOKBACK_WINDOW = 10 
EPOCHS = 50          
BATCH_SIZE = 32

def setup_directories():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

def create_dataset(data, lookback, target_count):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:(i + lookback), :])
        y.append(data[i + lookback, :target_count])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def build_safe_model(input_shape, output_units):
    """
    フリーズの原因となるLSTMを使わず、安全なDense(全結合)ネットワークを構築します。
    """
    model = Sequential([
        Input(shape=input_shape),
        Flatten(), # 3次元データを2次元に変換
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(output_units)
    ])
    model.compile(optimizer='adam', loss='mae')
    return model

if __name__ == "__main__":
    setup_directories()

    for loto_type in LOTO_TYPES:
        print(f"\n--- {loto_type.upper()} 学習開始 ---")
        data_file = f"data/{loto_type}_processed.csv"
        
        if loto_type == "miniloto":
            target_cols = ["num1", "num2", "num3", "num4", "num5"]
        elif loto_type == "loto6":
            target_cols = ["num1", "num2", "num3", "num4", "num5", "num6"]
        elif loto_type == "loto7":
            target_cols = ["num1", "num2", "num3", "num4", "num5", "num6", "num7"]

        if not os.path.exists(data_file):
            print(f"データが見つかりません: {data_file}")
            continue
            
        df = pd.read_csv(data_file)
        features_df = df.drop(['draw_id', 'date'], axis=1)
        other_cols = [c for c in features_df.columns if c not in target_cols]
        features_df = features_df[target_cols + other_cols]
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(features_df)

        X, y = create_dataset(scaled_data, LOOKBACK_WINDOW, len(target_cols))
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # 安全なモデルを構築
        model = build_safe_model((X_train.shape[1], X_train.shape[2]), len(target_cols))

        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model_path = os.path.join(MODEL_DIR, f"{loto_type}_lstm_best.keras")
        checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)

        model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_val, y_val),
            callbacks=[early_stop, checkpoint],
            verbose=1
        )