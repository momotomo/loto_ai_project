import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import warnings

# --- Mac環境での安定動作設定 ---
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # GPU干渉を防止

# TensorFlowの初期化設定（Macでのフリーズ防止）
tf.config.set_visible_devices([], 'GPU')

# --- 定数・検証設定 ---
LOTO_TYPES = ["miniloto", "loto6", "loto7"]
LOOKBACK_WINDOW = 10
TEST_COUNT = 5  # 直近何回分を検証するか

LOTO_CONFIG = {
    "miniloto": {"name": "ミニロト", "max_num": 31, "pick_count": 5},
    "loto6": {"name": "ロト6", "max_num": 43, "pick_count": 6},
    "loto7": {"name": "ロト7", "max_num": 37, "pick_count": 7}
}

if __name__ == "__main__":
    print("\n" + "="*60)
    print(" 🕵️ 宝くじ AI過去検証（バックテスト）シミュレーター")
    print(" (Google Colab 学習済みモデルを使用)")
    print("="*60 + "\n")
    print(f"直近 {TEST_COUNT} 回の抽選結果に対して、当時のデータのみを使用して")
    print("AIが事前に予測していた場合の的中率を確認します。\n")

    for loto_type in LOTO_TYPES:
        data_file = f"data/{loto_type}_processed.csv"
        model_path = f"models/{loto_type}_lstm_best.keras"
        
        if not os.path.exists(model_path):
            print(f"[{loto_type.upper()}] モデルが見つかりません。スキップします。")
            continue
        
        if not os.path.exists(data_file):
            print(f"[{loto_type.upper()}] 過去データが見つかりません。")
            continue

        # --- データの準備 ---
        df = pd.read_csv(data_file)
        config = LOTO_CONFIG[loto_type]
        target_cols = [f"num{i+1}" for i in range(config["pick_count"])]
        
        features_df = df.drop(['draw_id', 'date'], axis=1)
        other_cols = [c for c in features_df.columns if c not in target_cols]
        features_df = features_df[target_cols + other_cols]
        
        # モデルの読み込み
        model = load_model(model_path, compile=False)
        
        # 正規化の準備（全データでFitさせて学習時と同じスケールを再現）
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(features_df)
        
        print(f"--- {config['name']} の検証結果 ---")

        # 直近5回分を1回ずつ検証
        for i in range(TEST_COUNT, 0, -1):
            target_idx = len(df) - i
            actual_draw_id = df.iloc[target_idx]['draw_id']
            actual_date = df.iloc[target_idx]['date']
            actual_nums = df.iloc[target_idx][target_cols].astype(int).tolist()
            
            # 予測に使うのは「その回の前」までのデータ10回分
            start_idx = target_idx - LOOKBACK_WINDOW
            X_input = np.array([scaled_data[start_idx:target_idx]], dtype=np.float32)
            
            # 推論
            input_tensor = tf.convert_to_tensor(X_input)
            scaled_pred = model(input_tensor, training=False).numpy()
            
            # 逆正規化
            dummy = np.zeros((1, len(features_df.columns)))
            dummy[0, :config["pick_count"]] = scaled_pred[0]
            raw_pred = scaler.inverse_transform(dummy)[0, :config["pick_count"]]
            
            # 予測値を整数に丸めてソート
            ai_prediction = sorted([int(n) for n in np.round(np.clip(raw_pred, 1, config["max_num"]))])
            
            # 的中確認
            hit_nums = sorted(list(set(ai_prediction) & set(actual_nums)))
            hit_count = len(hit_nums)
            
            # 出力の整形（np.int64問題を回避）
            actual_str = ", ".join(map(str, actual_nums))
            ai_str = ", ".join(map(str, ai_prediction))
            hit_str = ", ".join(map(str, hit_nums)) if hit_nums else "なし"
            
            print(f"第{actual_draw_id}回 ({actual_date}):")
            print(f"  🎯 正解: [{actual_str}]")
            print(f"  🤖 AI  : [{ai_str}]")
            
            result_label = f"✨ {hit_count}個的中！" if hit_count >= 3 else f"{hit_count}個的中"
            print(f"  => 結果: {result_label} (的中数字: {hit_str})")
            print("")

        print("-" * 50)

    print("\nすべての検証が完了しました。")