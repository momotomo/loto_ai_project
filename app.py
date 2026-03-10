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
    walk_forward_model = metrics_summary.get("walk_forward_model", {})
    best_static = metrics_summary.get("best_static_baseline", {})

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("latest_draw_id", manifest.get("latest_draw_id"))
    col2.metric("WF LogLoss", format_metric(walk_forward_model.get("logloss_mean")))
    col3.metric("WF Brier", format_metric(walk_forward_model.get("brier_mean")))
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

tab1, tab2 = st.tabs(["🎲 確率サンプリング予測", "📊 モデル評価レポート (Walk-Forward)"])

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
