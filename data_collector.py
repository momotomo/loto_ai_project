import os
import pandas as pd
import requests
import io
import time
import warnings

# pandasの不要な警告を非表示にする
warnings.filterwarnings('ignore')

# プロジェクトのディレクトリ設定
DATA_DIR = "data"

def setup_directories():
    """データを保存するためのディレクトリを作成します。"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"ディレクトリ作成: {DATA_DIR}/")

def auto_download_lottery_data(loto_type):
    """
    Web上に公開されている過去データのCSVファイルに直接アクセスし、
    自動でダウンロードと整形を行います。
    """
    print(f"🌐 [{loto_type.upper()}] Webから過去データを自動ダウンロード中...")
    
    # 宝くじ過去データを提供している代表的なアーカイブサイトのCSV URL
    url = f"https://{loto_type}.thekyo.jp/data/{loto_type}.csv"
    
    try:
        # サーバーに負荷をかけないよう少し待機
        time.sleep(1) 
        
        # データのダウンロード (プログラムからのアクセスと明示しつつ弾かれないように設定)
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # 日本のサイトのCSVはShift-JISでエンコードされていることが多いので考慮
        try:
            csv_text = response.content.decode('shift_jis')
        except UnicodeDecodeError:
            csv_text = response.content.decode('utf-8')
            
        # ヘッダーなしで一旦全データを読み込む（不要な行があってもエラーを無視）
        df_raw = pd.read_csv(io.StringIO(csv_text), header=None, on_bad_lines='skip')
        
        # 1列目が数字（回号）である行だけを抽出（サイトのヘッダーや説明文などの不要な行を除去）
        df_raw = df_raw[pd.to_numeric(df_raw[0], errors='coerce').notnull()]
        
        # --- データの整形（AIが必要とする列だけを左から順番に抽出） ---
        # サイトのCSV構造: 0:回号, 1:日付, 2〜:本数字, その後:ボーナスや賞金情報など
        if loto_type == "miniloto":
            df = df_raw.iloc[:, :7].copy()
            df.columns = ["draw_id", "date", "num1", "num2", "num3", "num4", "num5"]
        elif loto_type == "loto6":
            df = df_raw.iloc[:, :8].copy()
            df.columns = ["draw_id", "date", "num1", "num2", "num3", "num4", "num5", "num6"]
        elif loto_type == "loto7":
            df = df_raw.iloc[:, :9].copy()
            df.columns = ["draw_id", "date", "num1", "num2", "num3", "num4", "num5", "num6", "num7"]
            
        # 本数字の部分を確実に整数型(int)に変換
        num_cols = [c for c in df.columns if c.startswith('num')]
        df[num_cols] = df[num_cols].astype(int)
        
        # 古いデータが上、新しいデータが下になるようにソート（回号で昇順）
        df["draw_id"] = df["draw_id"].astype(int)
        df = df.sort_values("draw_id").reset_index(drop=True)
        
        # ローカルに「生データ(raw)」として保存（確認用）
        raw_filepath = os.path.join(DATA_DIR, f"{loto_type}_raw.csv")
        df.to_csv(raw_filepath, index=False, encoding="utf-8")
        
        print(f"✅ ダウンロード成功: 全 {len(df)} 回分の実データを取得しました。")
        return df

    except Exception as e:
        print(f"❌ 自動ダウンロードに失敗しました: {e}")
        print(f"※対象のサイト({url})がアクセス制限を行っている可能性があります。")
        return None

def feature_engineering(df, loto_type):
    """
    AIに学習させるための付加情報（合計値や奇数偶数の比率など）を計算します。
    """
    if df is None or len(df) == 0:
        return None
        
    print(f"⚙️ {loto_type} の特徴量（AI用の追加データ）を計算中...")
    
    num_cols = [c for c in df.columns if c.startswith('num')]
    pick_count = len(num_cols)
    
    # 各回の合計値（ガウス和制約の基礎）
    df["sum_val"] = df[num_cols].sum(axis=1)
    
    # 奇数の数（パリティ分散の基礎）
    df["odd_count"] = df[num_cols].apply(lambda row: sum(1 for x in row if x % 2 != 0), axis=1)
    df["even_count"] = pick_count - df["odd_count"]
    
    # 簡単な移動平均（トレンドの確認用）
    df["sum_moving_avg_5"] = df["sum_val"].rolling(window=5).mean().fillna(df["sum_val"])

    return df

if __name__ == "__main__":
    print("=================================================")
    print(" 📡 宝くじ AI予測システム: 完全自動データ収集パイプライン")
    print("=================================================\n")
    
    setup_directories()
    loto_types = ["miniloto", "loto6", "loto7"]
    success_count = 0
    
    for ltype in loto_types:
        print(f"\n--- {ltype.upper()} の処理を開始 ---")
        
        # 1. ネットワーク経由で実データを自動ダウンロード
        history_df = auto_download_lottery_data(loto_type=ltype)
        
        if history_df is not None:
            # 2. 特徴量の追加
            processed_df = feature_engineering(history_df, loto_type=ltype)
            
            # 3. 処理済みデータを保存
            processed_filepath = os.path.join(DATA_DIR, f"{ltype}_processed.csv")
            processed_df.to_csv(processed_filepath, index=False, encoding="utf-8")
            print(f"💾 特徴量追加済みのデータを保存しました: {processed_filepath}")
            success_count += 1
    
    print("\n=================================================")
    if success_count == len(loto_types):
        print("🎉 すべての宝くじのデータ収集が【完全自動】で完了しました！")
        print("次は `python train_model.py` を実行して、この最新の実データでAIを再学習させてください。")
    else:
        print("⚠️ 一部のデータの自動収集に失敗しました。")