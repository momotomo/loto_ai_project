import ast
import json
import os
import pickle
import shutil
import warnings
from datetime import datetime

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model

from config import LOOKBACK_WINDOW, LOTO_CONFIG, generate_valid_sample

# --- Mac環境安定化設定 ---
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

st.set_page_config(page_title="宝くじAI 確率予測ダッシュボード", page_icon="🎲", layout="wide")

BASELINE_LABELS = {
    "uniform": "Uniform",
    "frequency": "Frequency (static)",
    "gap": "Gap (static)",
    "frequency_online": "Frequency (online)",
    "gap_online": "Gap (online)",
}
HISTORY_LIST_COLUMNS = [
    "actual_numbers",
    "predicted_top_k",
    "predicted_top_k_hit_numbers",
    "top_probability_numbers",
    "top_probability_scores",
]


def replace_file(source_path, target_path, copy_only=False):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    if os.path.exists(target_path):
        os.remove(target_path)
    if copy_only:
        shutil.copy2(source_path, target_path)
    else:
        shutil.move(source_path, target_path)


def sync_from_kaggle(slug):
    try:
        if not os.getenv("KAGGLE_USERNAME") or not os.getenv("KAGGLE_KEY"):
            if "KAGGLE_USERNAME" in st.secrets and "KAGGLE_KEY" in st.secrets:
                os.environ["KAGGLE_USERNAME"] = st.secrets["KAGGLE_USERNAME"]
                os.environ["KAGGLE_KEY"] = st.secrets["KAGGLE_KEY"]
            else:
                return False, "❌ Kaggleの認証情報が見つかりません。"

        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()

        temp_dir = "kaggle_temp"
        os.makedirs(temp_dir, exist_ok=True)
        api.kernels_output(slug, path=temp_dir)

        os.makedirs("data", exist_ok=True)
        os.makedirs("models", exist_ok=True)

        for root, _, files in os.walk(temp_dir):
            for file_name in files:
                source_path = os.path.join(root, file_name)
                if file_name.endswith(".csv"):
                    replace_file(source_path, os.path.join("data", file_name))
                elif file_name.endswith(".keras") or file_name.endswith(".pkl"):
                    replace_file(source_path, os.path.join("models", file_name))
                elif file_name.endswith(".json"):
                    replace_file(source_path, os.path.join("data", file_name), copy_only=True)
                    if file_name.endswith("_feature_cols.json"):
                        replace_file(source_path, os.path.join("models", file_name))
                    else:
                        os.remove(source_path)

        shutil.rmtree(temp_dir)
        return True, "✅ Kaggleからの同期が完了しました。"
    except Exception as exc:
        return False, f"❌ 同期エラー: {str(exc)}"


@st.cache_data(ttl=3600)
def load_tabular_assets(ltype):
    data_path = f"data/{ltype}_processed.csv"
    cols_paths = [f"data/{ltype}_feature_cols.json", f"models/{ltype}_feature_cols.json"]

    if not os.path.exists(data_path):
        return None, None

    df = pd.read_csv(data_path)

    feature_cols = None
    for cols_path in cols_paths:
        if os.path.exists(cols_path):
            with open(cols_path, "r", encoding="utf-8") as handle:
                feature_cols = json.load(handle)
            break

    return df, feature_cols


@st.cache_resource(ttl=3600)
def load_model_assets(ltype):
    model_path = f"models/{ltype}_prob.keras"
    scaler_path = f"models/{ltype}_scaler.pkl"
    if not all(os.path.exists(path) for path in [model_path, scaler_path]):
        return None, None

    model = load_model(model_path, compile=False)
    with open(scaler_path, "rb") as handle:
        scaler = pickle.load(handle)
    return model, scaler


def load_assets(ltype):
    df, feature_cols = load_tabular_assets(ltype)
    model, scaler = load_model_assets(ltype)
    return df, model, scaler, feature_cols


@st.cache_data(ttl=3600)
def load_json_candidates(file_name):
    for path in [os.path.join("data", file_name), os.path.join("models", file_name)]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
    return None


def load_eval_report(ltype):
    return load_json_candidates(f"eval_report_{ltype}.json")


def load_manifest(ltype):
    return load_json_candidates(f"manifest_{ltype}.json")


def parse_history_list_cell(value):
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
            except Exception:
                continue
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
        return [text]
    return [value]


def parse_history_bool_series(series):
    if series is None:
        return pd.Series(dtype=bool)
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna(False).astype(str).str.lower().isin(["1", "true", "yes", "y"])


def normalize_prediction_history_df(df):
    if df is None or df.empty:
        return pd.DataFrame()

    normalized = df.copy()
    for column in HISTORY_LIST_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = [[] for _ in range(len(normalized))]
        else:
            normalized[column] = normalized[column].apply(parse_history_list_cell)

    for column in ["draw_id", "pick_count", "max_num", "predicted_top_k_hit_count"]:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    if "fold_index" in normalized.columns:
        normalized["fold_index"] = pd.to_numeric(normalized["fold_index"], errors="coerce")
    else:
        normalized["fold_index"] = np.nan

    if "evaluation_mode" not in normalized.columns:
        normalized["evaluation_mode"] = "unknown"
    normalized["evaluation_mode"] = normalized["evaluation_mode"].fillna("unknown").astype(str)

    if "hit_rate_any" not in normalized.columns:
        normalized["hit_rate_any"] = normalized["predicted_top_k_hit_count"].fillna(0) >= 1
    else:
        normalized["hit_rate_any"] = parse_history_bool_series(normalized["hit_rate_any"])

    if "hit_rate_two_plus" not in normalized.columns:
        normalized["hit_rate_two_plus"] = normalized["predicted_top_k_hit_count"].fillna(0) >= 2
    else:
        normalized["hit_rate_two_plus"] = parse_history_bool_series(normalized["hit_rate_two_plus"])

    if "date" in normalized.columns:
        normalized["date"] = normalized["date"].astype(str)

    normalized = normalized.sort_values(
        ["draw_id", "evaluation_mode", "fold_index"],
        ascending=[True, True, True],
        na_position="first",
    ).reset_index(drop=True)
    return normalized


@st.cache_data(ttl=3600)
def load_prediction_history(ltype):
    json_path = os.path.join("data", f"prediction_history_{ltype}.json")
    csv_path = os.path.join("data", f"prediction_history_{ltype}.csv")

    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        records = payload.get("records", payload) if isinstance(payload, dict) else payload
        return normalize_prediction_history_df(pd.DataFrame(records)), {"format": "json", "path": json_path}

    if os.path.exists(csv_path):
        return normalize_prediction_history_df(pd.read_csv(csv_path)), {"format": "csv", "path": csv_path}

    return None, None


def calculate_next_draw_date(loto_type, last_date_str):
    last_date = pd.to_datetime(last_date_str.replace("/", "-"))
    today = pd.to_datetime(datetime.now().date())
    start_date = last_date + pd.Timedelta(days=1) if last_date >= today else today

    if loto_type == "miniloto":
        draw_weekdays = [1]
    elif loto_type == "loto6":
        draw_weekdays = [0, 3]
    else:
        draw_weekdays = [4]

    next_date = start_date
    while next_date.weekday() not in draw_weekdays:
        next_date += pd.Timedelta(days=1)

    weekdays_ja = ["月", "火", "水", "木", "金", "土", "日"]
    return next_date.strftime("%Y/%m/%d"), weekdays_ja[next_date.weekday()]


def format_metric(value, digits=4):
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def format_number_list(values):
    if not values:
        return "-"
    return ", ".join(f"{int(value):02d}" for value in values)


def format_score_list(values):
    if not values:
        return "-"
    return ", ".join(f"{float(value):.3f}" for value in values)


def summary_entry_to_row(label, summary_entry):
    metric_summary = summary_entry["metric_summary"]
    return {
        "モデル": label,
        "LogLoss mean↓": metric_summary["logloss"]["mean"],
        "LogLoss var": metric_summary["logloss"]["variance"],
        "Brier mean↓": metric_summary["brier"]["mean"],
        "Brier var": metric_summary["brier"]["variance"],
        "Top-K mean↑": metric_summary["mean_overlap_top_k"]["mean"],
        "Top-K var": metric_summary["mean_overlap_top_k"]["variance"],
    }


def legacy_metrics_to_row(label, metrics):
    return {
        "モデル": label,
        "LogLoss (BCE)↓": metrics["logloss"],
        "Brier Score↓": metrics["brier"],
        "Top-K 正解重なり↑": metrics["mean_overlap_top_k"],
    }


def render_calibration_chart(calibration_rows, color):
    calibration_df = pd.DataFrame(calibration_rows)
    calibration_df = calibration_df.dropna(subset=["pred_prob", "true_prob"], how="all")
    if calibration_df.empty:
        st.info("Calibration の表示に十分なデータがありません。")
        return

    bars = alt.Chart(calibration_df).mark_bar(opacity=0.75, color=color).encode(
        x=alt.X("bin_range:O", title="予測確率 bin"),
        y=alt.Y("pred_prob:Q", title="平均予測確率"),
        tooltip=["bin_range", "count", alt.Tooltip("pred_prob:Q", format=".4f"), alt.Tooltip("true_prob:Q", format=".4f")],
    )
    line = alt.Chart(calibration_df).mark_line(color="red", point=True).encode(
        x="bin_range:O",
        y=alt.Y("true_prob:Q", title="実測率"),
    )
    st.altair_chart(bars + line, use_container_width=True)


def render_manifest_section(manifest):
    if not manifest:
        return

    st.subheader("🧾 Artifact Manifest")
    metrics_summary = manifest.get("metrics_summary", {})
    primary_model = metrics_summary.get("primary_model") or metrics_summary.get("walk_forward_model", {})
    best_static = metrics_summary.get("best_static_baseline", {})
    evaluation_source = metrics_summary.get("evaluation_source", "walk_forward")
    final_artifact_status = metrics_summary.get("final_artifact_status", "-")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("latest_draw_id", manifest.get("latest_draw_id"))
    col2.metric("Primary LogLoss", format_metric(primary_model.get("logloss_mean")))
    col3.metric("Primary Brier", format_metric(primary_model.get("brier_mean")))
    col4.metric("Best static", BASELINE_LABELS.get(best_static.get("name"), best_static.get("name", "-")))

    train_range = manifest.get("train_range", {})
    delta = best_static.get("delta_model_minus_baseline", {})
    st.caption(
        f"学習レンジ: 第{train_range.get('start_draw_id', '-')}回 ({train_range.get('start_date', '-')}) "
        f"〜 第{train_range.get('end_draw_id', '-')}回 ({train_range.get('end_date', '-')})"
    )
    if delta:
        st.caption(
            "model - best static: "
            f"logloss={format_metric(delta.get('logloss'))}, "
            f"brier={format_metric(delta.get('brier'))}, "
            f"top-k={format_metric(delta.get('mean_overlap_top_k'))}"
        )
    if manifest.get("prediction_history_rows") is not None:
        st.caption(
            "prediction_history: "
            f"{manifest.get('prediction_history_rows')} rows "
            f"({manifest.get('prediction_history_path', '-')})"
        )
    st.caption(f"evaluation_source={evaluation_source}, final_artifact_status={final_artifact_status}")

    with st.expander("Manifest 詳細", expanded=False):
        st.json(manifest)


def render_legacy_holdout_section(report):
    legacy = report.get("legacy_holdout")
    if not legacy:
        return

    rows = [legacy_metrics_to_row("★ AI Model (holdout)", legacy["model"])]
    rows.extend(
        legacy_metrics_to_row(BASELINE_LABELS.get(name, name), metrics)
        for name, metrics in legacy.get("static_baselines", {}).items()
    )
    rows.extend(
        legacy_metrics_to_row(BASELINE_LABELS.get(name, name), metrics)
        for name, metrics in legacy.get("online_baselines", {}).items()
    )

    st.dataframe(pd.DataFrame(rows).set_index("モデル"), use_container_width=True)
    st.write("##### Holdout Calibration")
    render_calibration_chart(legacy["model"]["calibration"], "#2563eb")


def render_walk_forward_section(report):
    walk_forward = report.get("walk_forward")
    if not walk_forward:
        st.subheader("🕵️ 評価レポート")
        rows = [legacy_metrics_to_row("★ AI Model (LSTM)", report["Model (LSTM)"])]
        rows.extend(
            legacy_metrics_to_row(BASELINE_LABELS.get(name, name), metrics)
            for name, metrics in report.get("Baselines", {}).items()
        )
        st.dataframe(pd.DataFrame(rows).set_index("モデル"), use_container_width=True)
        st.write("##### Calibration")
        render_calibration_chart(report["Model (LSTM)"]["calibration"], "#2563eb")
        return

    aggregate = walk_forward["aggregate"]
    settings = walk_forward.get("settings", {})

    st.subheader("🕵️ Walk-Forward 評価レポート")
    st.caption(
        "主比較対象は static baselines です。online baselines はテスト中に状態更新する参考値として別枠で表示します。"
    )
    st.caption(
        f"initial_train_fraction={settings.get('initial_train_fraction', '-')}, "
        f"test_window={settings.get('test_window', '-')}, "
        f"folds={len(walk_forward.get('folds', []))}"
    )

    main_rows = [summary_entry_to_row("★ AI Model", aggregate["model"])]
    main_rows.extend(
        summary_entry_to_row(BASELINE_LABELS.get(name, name), summary)
        for name, summary in aggregate.get("static_baselines", {}).items()
    )
    st.write("##### Walk-Forward Summary")
    st.dataframe(pd.DataFrame(main_rows).set_index("モデル"), use_container_width=True)

    online_rows = [
        summary_entry_to_row(BASELINE_LABELS.get(name, name), summary)
        for name, summary in aggregate.get("online_baselines", {}).items()
    ]
    if online_rows:
        st.write("##### Online Baselines (参考)")
        st.dataframe(pd.DataFrame(online_rows).set_index("モデル"), use_container_width=True)

    fold_rows = []
    for fold in walk_forward.get("folds", []):
        fold_rows.append(
            {
                "fold": fold["fold"],
                "train_end_draw_id": fold["train_range"]["end_draw_id"],
                "test_start_draw_id": fold["test_range"]["start_draw_id"],
                "test_end_draw_id": fold["test_range"]["end_draw_id"],
                "LogLoss": fold["model"]["logloss"],
                "Brier": fold["model"]["brier"],
                "Top-K": fold["model"]["mean_overlap_top_k"],
            }
        )

    if fold_rows:
        st.write("##### Fold 別モデル成績")
        st.dataframe(pd.DataFrame(fold_rows), use_container_width=True)

    st.write("##### Walk-Forward Calibration")
    render_calibration_chart(aggregate["model"]["calibration"], "#1d4ed8")

    with st.expander("Legacy Holdout", expanded=False):
        render_legacy_holdout_section(report)


def render_prediction_history_section(loto_type):
    history_df, history_meta = load_prediction_history(loto_type)
    st.subheader("✅ 実績との照合")

    if history_df is None or history_df.empty:
        st.info("prediction history が未生成です")
        st.write("train を再実行するか Kaggle 同期を実施してください")
        return

    st.caption(f"source={history_meta.get('format', '-')}, path={history_meta.get('path', '-')}")

    available_modes = ["all"] + sorted(history_df["evaluation_mode"].dropna().unique().tolist())
    default_min_draw = int(history_df["draw_id"].min())
    default_max_draw = int(history_df["draw_id"].max())

    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
    with filter_col1:
        selected_mode = st.selectbox("evaluation_mode", available_modes, index=0)
    with filter_col2:
        draw_range = st.slider("draw_id 範囲", default_min_draw, default_max_draw, (default_min_draw, default_max_draw))
    with filter_col3:
        recent_option = st.selectbox("直近N件表示", ["10", "20", "50", "全件"], index=3)
    with filter_col4:
        one_plus_only = st.checkbox("1個以上一致のみ表示", value=False)

    filtered = history_df.copy()
    if selected_mode != "all":
        filtered = filtered[filtered["evaluation_mode"] == selected_mode]
    filtered = filtered[filtered["draw_id"].between(draw_range[0], draw_range[1])]
    if one_plus_only:
        filtered = filtered[filtered["hit_rate_any"]]

    filtered = filtered.sort_values(["draw_id", "evaluation_mode", "fold_index"], ascending=[False, True, True])
    if recent_option != "全件":
        filtered = filtered.head(int(recent_option))

    if filtered.empty:
        st.warning("条件に一致する prediction history がありません。")
        return

    distribution = (
        filtered["predicted_top_k_hit_count"]
        .fillna(0)
        .astype(int)
        .value_counts()
        .sort_index()
        .rename_axis("一致数")
        .reset_index(name="件数")
    )

    total_rows = len(filtered)
    avg_hit = float(filtered["predicted_top_k_hit_count"].fillna(0).mean()) if total_rows else 0.0
    any_hits = int(filtered["hit_rate_any"].sum())
    two_plus_hits = int(filtered["hit_rate_two_plus"].sum())

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("対象件数", total_rows)
    metric_col2.metric("平均一致数", f"{avg_hit:.2f}")
    metric_col3.metric("1個以上一致", f"{any_hits} / {total_rows}", f"{(any_hits / total_rows):.1%}")
    metric_col4.metric("2個以上一致", f"{two_plus_hits} / {total_rows}", f"{(two_plus_hits / total_rows):.1%}")

    st.write("##### 一致数の分布")
    st.bar_chart(distribution.set_index("一致数"))

    display_df = filtered.copy()
    display_df["actual_numbers"] = display_df["actual_numbers"].apply(format_number_list)
    display_df["predicted_top_k"] = display_df["predicted_top_k"].apply(format_number_list)
    display_df["predicted_top_k_hit_numbers"] = display_df["predicted_top_k_hit_numbers"].apply(format_number_list)
    display_df["fold_index"] = display_df["fold_index"].apply(lambda value: "-" if pd.isna(value) else int(value))
    display_df = display_df[
        [
            "draw_id",
            "date",
            "actual_numbers",
            "predicted_top_k",
            "predicted_top_k_hit_numbers",
            "predicted_top_k_hit_count",
            "evaluation_mode",
            "fold_index",
        ]
    ].rename(
        columns={
            "draw_id": "draw_id",
            "date": "date",
            "actual_numbers": "actual_numbers",
            "predicted_top_k": "predicted_top_k",
            "predicted_top_k_hit_numbers": "predicted_top_k_hit_numbers",
            "predicted_top_k_hit_count": "hit_count",
            "evaluation_mode": "evaluation_mode",
            "fold_index": "fold_index",
        }
    )

    st.write("##### 回別一覧")
    st.dataframe(display_df, use_container_width=True)

    detail_source = filtered.reset_index(drop=True)
    selected_index = st.selectbox(
        "詳細表示",
        options=list(range(len(detail_source))),
        format_func=lambda index: (
            f"第{int(detail_source.iloc[index]['draw_id'])}回 "
            f"{detail_source.iloc[index]['date']} "
            f"{detail_source.iloc[index]['evaluation_mode']} "
            f"fold={('-' if pd.isna(detail_source.iloc[index]['fold_index']) else int(detail_source.iloc[index]['fold_index']))}"
        ),
    )
    detail_row = detail_source.iloc[selected_index]

    st.write("##### 詳細")
    detail_col1, detail_col2, detail_col3 = st.columns(3)
    detail_col1.metric("actual_numbers", format_number_list(detail_row["actual_numbers"]))
    detail_col2.metric("predicted_top_k", format_number_list(detail_row["predicted_top_k"]))
    detail_col3.metric("hit_numbers", format_number_list(detail_row["predicted_top_k_hit_numbers"]))

    probability_df = pd.DataFrame(
        {
            "number": detail_row["top_probability_numbers"],
            "probability": detail_row["top_probability_scores"],
        }
    )
    probability_df["actual_overlap"] = probability_df["number"].isin(detail_row["actual_numbers"])
    probability_df["probability"] = probability_df["probability"].map(lambda value: f"{float(value):.3f}")
    st.dataframe(probability_df, use_container_width=True)
    st.caption(f"top_probability_scores: {format_score_list(detail_row['top_probability_scores'])}")


with st.sidebar:
    st.header("☁️ Kaggle 同期設定")
    default_slug = os.getenv("KAGGLE_SLUG", "")
    if not default_slug and "KAGGLE_SLUG" in st.secrets:
        default_slug = st.secrets["KAGGLE_SLUG"]

    k_slug = st.text_input("Notebook Slug", value=default_slug)

    if st.button("🔄 最新AIモデルを同期", use_container_width=True):
        if k_slug:
            with st.spinner("同期中..."):
                success, message = sync_from_kaggle(k_slug)
                if success:
                    st.success(message)
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    st.rerun()
                else:
                    st.error(message)
        else:
            st.warning("Slugを入力してください。")


st.title("🎯 宝くじ AI確率予測システム")
st.markdown("LSTMから出力された**出現確率ベクトル**に基づき、重み付きサンプリングで買い目を生成します。")
st.markdown("---")

selected_loto = st.radio(
    "宝くじの種類",
    options=list(LOTO_CONFIG.keys()),
    format_func=lambda x: LOTO_CONFIG[x]["name"],
    horizontal=True,
)
config = LOTO_CONFIG[selected_loto]

df, model, scaler, feature_cols = load_assets(selected_loto)
manifest = load_manifest(selected_loto)

if df is None or feature_cols is None:
    st.error("⚠️ 必要なファイルが不足しています。")
    st.info("Kaggleの Output にモデルや JSON が存在するか確認し、サイドバーから同期し直してください。")
    st.stop()

tab1, tab2, tab3 = st.tabs(["🎲 確率サンプリング予測", "📊 モデル評価レポート (Walk-Forward)", "✅ 実績との照合"])

with tab1:
    last_draw_id = int(df.iloc[-1]["draw_id"])
    last_draw_date = df.iloc[-1]["date"]
    next_date, next_weekday = calculate_next_draw_date(selected_loto, last_draw_date)

    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.info(f"💾 **最新データ:** 第{last_draw_id}回 ({last_draw_date})")
    with col_info2:
        st.success(f"🗓️ **次回抽選:** {next_date} ({next_weekday})")

    st.markdown("---")
    st.subheader(f"✨ 次回（第{last_draw_id + 1}回）向けの確率予測と買い目生成")

    features_df = df[feature_cols]
    scaled_data = scaler.transform(features_df)
    recent_input = np.array([scaled_data[-LOOKBACK_WINDOW:]], dtype=np.float32)
    probs = model(tf.convert_to_tensor(recent_input), training=False).numpy()[0]

    prob_df = pd.DataFrame({"Number": np.arange(1, config["max_num"] + 1), "Probability": probs})
    chart = alt.Chart(prob_df).mark_bar(color=config["color"]).encode(
        x=alt.X("Number:O", title="数字"),
        y=alt.Y("Probability:Q", title="出現確率", axis=alt.Axis(format="%")),
        tooltip=["Number", alt.Tooltip("Probability:Q", format=".2%")],
    ).properties(height=250)
    st.altair_chart(chart, use_container_width=True)

    st.markdown("##### ⚙️ 買い目生成オプション")
    col1, col2, col3 = st.columns(3)
    with col1:
        sampling_mode = st.selectbox("抽出方式", ["weighted (確率重み付き抽出)", "top-k (最上位固定)"])
    with col2:
        use_psych = st.checkbox("期待値最大化フィルタ", value=True)
    with col3:
        use_stat = st.checkbox("基本統計フィルタ", value=True)

    num_tickets = st.slider("生成口数", 1, 10, 5)

    if st.button("買い目を生成する", type="primary", use_container_width=True):
        mode = "top-k" if "top-k" in sampling_mode else "weighted"
        cols = st.columns(min(num_tickets, 5))
        for index in range(num_tickets):
            candidate = generate_valid_sample(probs, config, use_psych, use_stat, sampling_mode=mode)
            candidate_str = ", ".join(f"{number:02d}" for number in candidate)
            with cols[index % 5]:
                st.markdown(
                    (
                        f'<div style="background-color:{config["color"]}; padding:10px; border-radius:8px; '
                        f'text-align:center; color:white; margin-bottom:10px;"><b>{candidate_str}</b></div>'
                    ),
                    unsafe_allow_html=True,
                )

with tab2:
    report = load_eval_report(selected_loto)
    render_manifest_section(manifest)

    if report:
        render_walk_forward_section(report)
    else:
        st.info("評価レポートが見つかりません。")
        st.write("デバッグ情報: data/ フォルダ内のファイル一覧")
        st.write(os.listdir("data") if os.path.exists("data") else "data フォルダが存在しません")

with tab3:
    render_prediction_history_section(selected_loto)
