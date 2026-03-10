import argparse
import io
import os
import time
import warnings

import numpy as np
import pandas as pd
import requests

from config import LOTO_CONFIG

warnings.filterwarnings("ignore")

DATA_DIR = "data"
REQUEST_TIMEOUT_SECONDS = 10
MAX_RETRIES = 3
BACKOFF_SECONDS = [1, 2, 4]


class DataCollectionError(RuntimeError):
    """収集パイプラインから扱いやすくするための明示的な例外。"""


def setup_directories():
    os.makedirs(DATA_DIR, exist_ok=True)


def get_target_columns(loto_type):
    pick_count = LOTO_CONFIG[loto_type]["pick_count"]
    return [f"num{i + 1}" for i in range(pick_count)]


def decode_csv_bytes(content):
    for encoding in ("shift_jis", "cp932", "utf-8"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise DataCollectionError("CSV の文字コードを判定できませんでした。")


def download_csv_text(url):
    headers = {"User-Agent": "Mozilla/5.0 (compatible; loto-ai-bot/1.0)"}
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            return decode_csv_bytes(response.content)
        except requests.RequestException as exc:
            last_error = exc
        except DataCollectionError as exc:
            last_error = exc

        if attempt < MAX_RETRIES:
            time.sleep(BACKOFF_SECONDS[attempt - 1])

    raise DataCollectionError(f"CSV 取得に失敗しました: {url} ({last_error})")


def parse_downloaded_csv(csv_text, loto_type):
    num_cols = get_target_columns(loto_type)
    expected_columns = 2 + len(num_cols)
    df_raw = pd.read_csv(io.StringIO(csv_text), header=None, on_bad_lines="skip")
    df_raw = df_raw[pd.to_numeric(df_raw[0], errors="coerce").notnull()]

    if df_raw.empty:
        raise DataCollectionError("CSV から抽選データ行を抽出できませんでした。")
    if df_raw.shape[1] < expected_columns:
        raise DataCollectionError(
            f"CSV 列数が不足しています。expected>={expected_columns}, actual={df_raw.shape[1]}"
        )

    df = df_raw.iloc[:, :expected_columns].copy()
    df.columns = ["draw_id", "date", *num_cols]
    return df


def validate_history_dataframe(df, loto_type):
    config = LOTO_CONFIG[loto_type]
    num_cols = get_target_columns(loto_type)
    required_columns = ["draw_id", "date", *num_cols]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise DataCollectionError(f"必要カラムが不足しています: {missing_columns}")

    validated = df.copy()
    validated["draw_id"] = pd.to_numeric(validated["draw_id"], errors="coerce")
    if validated["draw_id"].isna().any():
        raise DataCollectionError("draw_id を数値として解釈できない行があります。")

    for column in num_cols:
        validated[column] = pd.to_numeric(validated[column], errors="coerce")
        if validated[column].isna().any():
            raise DataCollectionError(f"{column} を数値として解釈できない行があります。")

    validated["draw_id"] = validated["draw_id"].astype(int)
    validated[num_cols] = validated[num_cols].astype(int)
    deduped_count = len(validated)
    validated = validated.drop_duplicates(subset=["draw_id"], keep="last").sort_values("draw_id").reset_index(drop=True)
    removed_duplicates = deduped_count - len(validated)

    diffs = validated["draw_id"].diff().fillna(1)
    if not (diffs > 0).all():
        raise DataCollectionError("draw_id が単調増加になっていません。")

    invalid_range_mask = (validated[num_cols] < 1) | (validated[num_cols] > config["max_num"])
    if invalid_range_mask.any(axis=None):
        first_invalid = int(np.where(invalid_range_mask.any(axis=1))[0][0])
        raise DataCollectionError(
            f"数字範囲外の行があります。row={first_invalid}, max_num={config['max_num']}"
        )

    for row_index, values in enumerate(validated[num_cols].to_numpy(dtype=int)):
        if len(set(values.tolist())) != len(values):
            draw_id = int(validated.iloc[row_index]["draw_id"])
            raise DataCollectionError(f"重複数字を含む行があります。draw_id={draw_id}")

    return validated, removed_duplicates


def read_latest_draw_id(path):
    if not os.path.exists(path):
        return None

    try:
        previous = pd.read_csv(path)
        if previous.empty or "draw_id" not in previous.columns:
            return None
        return int(previous["draw_id"].iloc[-1])
    except Exception:
        return None


def auto_download_lottery_data(loto_type):
    url = f"https://{loto_type}.thekyo.jp/data/{loto_type}.csv"
    csv_text = download_csv_text(url)
    parsed = parse_downloaded_csv(csv_text, loto_type)
    validated, removed_duplicates = validate_history_dataframe(parsed, loto_type)
    if removed_duplicates:
        print(f"ℹ️ [{loto_type.upper()}] 重複 draw_id を {removed_duplicates} 件除去しました。")
    return validated


def feature_engineering(df, loto_type):
    if df is None or len(df) == 0:
        raise DataCollectionError(f"[{loto_type}] 特徴量化対象のデータが空です。")

    print(f"⚙️ [{loto_type.upper()}] 特徴量を計算中...")
    num_cols = get_target_columns(loto_type)
    pick_count = len(num_cols)
    engineered = df.copy()
    engineered["sum_val"] = engineered[num_cols].sum(axis=1)
    engineered["odd_count"] = engineered[num_cols].apply(lambda row: sum(1 for value in row if value % 2 != 0), axis=1)
    engineered["even_count"] = pick_count - engineered["odd_count"]
    engineered["sum_moving_avg_5"] = engineered["sum_val"].rolling(window=5).mean().fillna(engineered["sum_val"])
    return engineered


def collect_lottery_data(loto_type):
    print(f"🌐 [{loto_type.upper()}] Web から過去データを取得中...")
    raw_path = os.path.join(DATA_DIR, f"{loto_type}_raw.csv")
    processed_path = os.path.join(DATA_DIR, f"{loto_type}_processed.csv")
    previous_latest_draw_id = read_latest_draw_id(raw_path)

    history_df = auto_download_lottery_data(loto_type)
    processed_df = feature_engineering(history_df, loto_type)
    history_df.to_csv(raw_path, index=False, encoding="utf-8")
    processed_df.to_csv(processed_path, index=False, encoding="utf-8")

    latest_draw_id = int(history_df.iloc[-1]["draw_id"])
    if previous_latest_draw_id is None:
        print(f"✅ [{loto_type.upper()}] 初回保存完了: latest_draw_id={latest_draw_id}")
    elif latest_draw_id == previous_latest_draw_id:
        print(f"ℹ️ [{loto_type.upper()}] 更新なし: latest_draw_id={latest_draw_id}")
    else:
        print(
            f"✅ [{loto_type.upper()}] 更新検知: "
            f"{previous_latest_draw_id} -> {latest_draw_id}"
        )

    print(f"💾 [{loto_type.upper()}] raw={raw_path}, processed={processed_path}")
    return {
        "loto_type": loto_type,
        "rows": len(history_df),
        "latest_draw_id": latest_draw_id,
        "updated": previous_latest_draw_id != latest_draw_id if previous_latest_draw_id is not None else True,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Download and validate lottery history data.")
    parser.add_argument("--loto_type", choices=sorted(LOTO_CONFIG.keys()), help="対象の宝くじ種類を1つに絞る")
    return parser.parse_args()


def main():
    print("=================================================")
    print(" 📡 宝くじ AI予測システム: データ収集パイプライン")
    print("=================================================\n")

    setup_directories()
    args = parse_args()
    loto_types = [args.loto_type] if args.loto_type else list(LOTO_CONFIG.keys())
    success_count = 0
    failed = []

    for loto_type in loto_types:
        print(f"\n--- {loto_type.upper()} の処理を開始 ---")
        try:
            result = collect_lottery_data(loto_type)
            print(
                f"📌 [{loto_type.upper()}] rows={result['rows']}, "
                f"latest_draw_id={result['latest_draw_id']}, updated={result['updated']}"
            )
            success_count += 1
        except DataCollectionError as exc:
            failed.append((loto_type, str(exc)))
            print(f"❌ [{loto_type.upper()}] 収集に失敗しました: {exc}")

    print("\n=================================================")
    if success_count == len(loto_types):
        print("🎉 指定した宝くじデータの収集が完了しました。")
    else:
        print("⚠️ 一部のデータ収集に失敗しました。")
        for loto_type, message in failed:
            print(f" - {loto_type}: {message}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
