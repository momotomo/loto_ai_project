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

# --- 定数・統計フィルタ設定 ---
LOTO_TYPES = ["miniloto", "loto6", "loto7"]
LOOKBACK_WINDOW = 10

LOTO_CONFIG = {
    "miniloto": {
        "name": "ミニロト", 
        "max_num": 31, 
        "pick_count": 5, 
        "sum_range": [60, 100], 
        "max_parity_diff": 3
    },
    "loto6": {
        "name": "ロト6", 
        "max_num": 43, 
        "pick_count": 6, 
        "sum_range": [115, 150], 
        "max_parity_diff": 2
    },
    "loto7": {
        "name": "ロト7", 
        "max_num": 37, 
        "pick_count": 7, 
        "sum_range": [110, 155], 
        "max_parity_diff": 3
    }
}

def generate_candidates(base_pred, config, num_candidates=1000):
    """AIの予測値をベースに、統計的フィルタをクリアする候補を生成します。"""
    candidates = []
    for _ in range(num_candidates):
        # 予測値の周辺にノイズを加えてバリエーションを作る
        noise = np.random.uniform(-2.5, 2.5, size=config["pick_count"])
        candidate = np.round(base_pred + noise).astype(float)
        candidate = np.clip(candidate, 1, config["max_num"])
        
        # 重複除去
        unique_res = []
        for n in candidate:
            n_int = int(n)
            while n_int in unique_res:
                n_int = np.random.randint(1, config["max_num"] + 1)
            unique_res.append(n_int)
        
        unique_res.sort()
        candidates.append(unique_res)
    return candidates

def apply_filters(candidates, config):
    """合計値と奇数・偶数のバランスで候補を絞り込みます。"""
    valid_list = []
    for nums in candidates:
        # 1. 合計値チェック
        if not (config["sum_range"][0] <= sum(nums) <= config["sum_range"][1]):
            continue
        
        # 2. 奇数・偶数バランスチェック
        odd_cnt = sum(1 for n in nums if n % 2 != 0)
        even_cnt = config["pick_count"] - odd_cnt
        if abs(odd_cnt - even_cnt) > config["max_parity_diff"]:
            continue
            
        valid_list.append(nums)
    return valid_list

if __name__ == "__main__":
    print("\n" + "="*60)
    print(" 🎯 宝くじ AI予測システム - ターミナル出力版")
    print(" (Google Colab 学習済みモデル使用)")
    print("="*60 + "\n")

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
        
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(features_df)
        
        # 直近10回分のデータ
        recent_input = np.array([scaled_data[-LOOKBACK_WINDOW:]], dtype=np.float32)

        # --- 推論の実行 ---
        # compile=Falseで読み込み、関数として呼び出すことでフリーズを回避
        model = load_model(model_path, compile=False)
        input_tensor = tf.convert_to_tensor(recent_input)
        scaled_pred = model(input_tensor, training=False).numpy()
        
        # 逆正規化（スケールを元に戻す）
        dummy = np.zeros((1, len(features_df.columns)))
        dummy[0, :config["pick_count"]] = scaled_pred[0]
        raw_pred = scaler.inverse_transform(dummy)[0, :config["pick_count"]]
        
        # --- 候補生成とフィルタリング ---
        print(f"--- {config['name']} の予測結果 ---")
        candidates = generate_candidates(raw_pred, config)
        valid_candidates = apply_filters(candidates, config)
        
        if valid_candidates:
            # 重複を排除
            seen = set()
            unique_valid = []
            for c in valid_candidates:
                t = tuple(c)
                if t not in seen:
                    seen.add(t)
                    unique_valid.append(c)
            
            # AIの生予測値に近い順にソートして上位5件を表示
            sorted_matches = sorted(unique_valid, key=lambda x: np.sum(np.abs(np.array(x) - raw_pred)))
            top_5 = sorted_matches[:5]
            
            for i, match in enumerate(top_5):
                # 整数に変換してカンマ区切りの文字列に整形（np.int64表示を回避）
                match_ints = [int(n) for n in match]
                match_str = ", ".join(map(str, match_ints))
                
                sum_val = sum(match_ints)
                odd_cnt = sum(1 for n in match_ints if n % 2 != 0)
                even_cnt = config["pick_count"] - odd_cnt
                
                print(f" 🌟 第{i+1}候補: [{match_str}]")
                print(f"    (合計:{sum_val} | 奇数{odd_cnt}:偶数{even_cnt})")
        else:
            # フィルタを通るものがなかった場合のバックアップ表示
            fallback = sorted([int(n) for n in np.round(np.clip(raw_pred, 1, config["max_num"]))])
            print(f" ⚠️ 推奨(統計外): {', '.join(map(str, fallback))}")
        
        print("-" * 40)

    print("\nすべての予測が完了しました。")