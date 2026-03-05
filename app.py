import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import pickle
import shutil
import tensorflow as tf
from tensorflow.keras.models import load_model
import warnings
import altair as alt

from config import LOTO_CONFIG, LOOKBACK_WINDOW, generate_valid_sample

# --- Mac環境安定化設定 ---
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
tf.config.set_visible_devices([], 'GPU')

st.set_page_config(page_title="宝くじAI 確率予測ダッシュボード", page_icon="🎲", layout="wide")

# ==========================================
# ☁️ Kaggle同期 (Sync) 機能
# ==========================================
def sync_from_kaggle(slug):
    try:
        if not os.getenv('KAGGLE_USERNAME') or not os.getenv('KAGGLE_KEY'):
            if 'KAGGLE_USERNAME' in st.secrets and 'KAGGLE_KEY' in st.secrets:
                os.environ['KAGGLE_USERNAME'] = st.secrets['KAGGLE_USERNAME']
                os.environ['KAGGLE_KEY'] = st.secrets['KAGGLE_KEY']
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
        
        # 確率モデルに必要なファイルをすべて抽出・移動
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                source_path = os.path.join(root, file)
                if file.endswith(".csv") or (file.endswith(".json") and "eval" in file):
                    shutil.move(source_path, os.path.join("data", file))
                elif file.endswith(".keras") or file.endswith(".pkl") or file.endswith(".json"):
                    shutil.move(source_path, os.path.join("models", file))
                    
        shutil.rmtree(temp_dir)
        return True, "✅ Kaggleからの最新確率モデル・データの同期が完了しました！"
    except Exception as e:
        return False, f"❌ 同期エラー: {str(e)}"

# ==========================================
# データ・モデル読み込み関数
# ==========================================
@st.cache_data(ttl=3600)
def load_assets(ltype):
    data_path = f"data/{ltype}_processed.csv"
    model_path = f"models/{ltype}_prob.keras"
    scaler_path = f"models/{ltype}_scaler.pkl"
    cols_path = f"models/{ltype}_feature_cols.json"
    
    if not all(os.path.exists(p) for p in [data_path, model_path, scaler_path, cols_path]):
        return None, None, None, None
        
    df = pd.read_csv(data_path)
    model = load_model(model_path, compile=False)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(cols_path, "r") as f:
        feature_cols = json.load(f)
        
    return df, model, scaler, feature_cols

def load_eval_report(ltype):
    p = f"data/eval_report_{ltype}.json"
    if os.path.exists(p):
        with open(p, "r") as f:
            return json.load(f)
    return None

def calculate_next_draw_date(loto_type, last_date_str):
    last_date = pd.to_datetime(last_date_str.replace('/', '-'))
    from datetime import datetime
    today = pd.to_datetime(datetime.now().date())
    start_date = last_date + pd.Timedelta(days=1) if last_date >= today else today
    
    draw_weekdays = []
    if loto_type == "miniloto": draw_weekdays = [1]
    elif loto_type == "loto6": draw_weekdays = [0, 3]
    elif loto_type == "loto7": draw_weekdays = [4]
    
    next_date = start_date
    while next_date.weekday() not in draw_weekdays:
        next_date += pd.Timedelta(days=1)
        
    weekdays_ja = ["月", "火", "水", "木", "金", "土", "日"]
    return next_date.strftime("%Y/%m/%d"), weekdays_ja[next_date.weekday()]

# ==========================================
# サイドバー
# ==========================================
with st.sidebar:
    st.header("☁️ Kaggle 同期設定")
    default_slug = os.getenv('KAGGLE_SLUG', "")
    if not default_slug and 'KAGGLE_SLUG' in st.secrets:
        default_slug = st.secrets['KAGGLE_SLUG']
        
    k_slug = st.text_input("Notebook Slug", value=default_slug)
    
    if st.button("🔄 最新AIモデルを同期", use_container_width=True):
        if k_slug:
            with st.spinner("同期中..."):
                success, msg = sync_from_kaggle(k_slug)
                if success:
                    st.success(msg)
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error(msg)
        else:
            st.warning("Slugを入力してください。")

# ==========================================
# メイン UI
# ==========================================
st.title("🎯 宝くじ AI確率予測システム (Multi-Label Classification)")
st.markdown("LSTMから出力された**出現確率ベクトル**に基づき、重み付きサンプリングで買い目を生成します。")
st.markdown("---")

selected_loto = st.radio("宝くじの種類", options=list(LOTO_CONFIG.keys()), format_func=lambda x: LOTO_CONFIG[x]["name"], horizontal=True)
config = LOTO_CONFIG[selected_loto]

df, model, scaler, feature_cols = load_assets(selected_loto)

if df is None:
    st.error("⚠️ 必要なファイルが不足しています。Kaggle同期またはローカルでの学習(train_prob_model.py)を実行してください。")
    st.stop()

tab1, tab2 = st.tabs(["🎲 確率サンプリング予測", "📊 モデル評価レポート (Walk-Forward)"])

with tab1:
    last_draw_id = int(df.iloc[-1]['draw_id'])
    last_draw_date = df.iloc[-1]['date']
    next_date, next_weekday = calculate_next_draw_date(selected_loto, last_draw_date)
    
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.info(f"💾 **最新データ:** 第{last_draw_id}回 ({last_draw_date})")
    with col_info2:
        st.success(f"🗓️ **次回抽選:** {next_date} ({next_weekday})")

    st.markdown("---")
    st.subheader(f"✨ 次回（第{last_draw_id + 1}回）向けの確率予測と買い目生成")
    
    # --- 推論処理 ---
    # 学習時と完全に同じ特徴量の順序を保証する
    features_df = df[feature_cols]
    scaled_data = scaler.transform(features_df)
    recent_input = np.array([scaled_data[-LOOKBACK_WINDOW:]], dtype=np.float32)
    probs = model(tf.convert_to_tensor(recent_input), training=False).numpy()[0]
    
    # 確率バーチャート
    prob_df = pd.DataFrame({"Number": np.arange(1, config["max_num"] + 1), "Probability": probs})
    chart = alt.Chart(prob_df).mark_bar(color=config["color"]).encode(
        x=alt.X('Number:O', title='数字'),
        y=alt.Y('Probability:Q', title='出現確率', axis=alt.Axis(format='%')),
        tooltip=['Number', alt.Tooltip('Probability:Q', format='.2%')]
    ).properties(height=250)
    st.altair_chart(chart, use_container_width=True)
    
    st.markdown("##### ⚙️ 買い目生成オプション")
    col1, col2, col3 = st.columns(3)
    with col1:
        sampling_mode = st.selectbox("抽出方式", ["weighted (確率重み付き抽出)", "top-k (最上位固定)"])
    with col2:
        use_psych = st.checkbox("期待値最大化フィルタ(人間心理の逆張り)", value=True, help="素数、連番、32以上などを含め、等差数列を除外")
    with col3:
        use_stat = st.checkbox("基本統計フィルタ", value=True, help="合計値のガウス和制約とパリティ分散制約")
        
    num_tickets = st.slider("生成口数", 1, 10, 5)
    
    if st.button("買い目を生成する", type="primary", use_container_width=True):
        st.success("🎯 生成結果")
        mode = "top-k" if "top-k" in sampling_mode else "weighted"
        
        cols = st.columns(min(num_tickets, 5))
        for i in range(num_tickets):
            cand = generate_valid_sample(probs, config, use_psych, use_stat, sampling_mode=mode)
            cand_str = ", ".join(f"{n:02d}" for n in cand)
            with cols[i % 5]:
                st.markdown(f"""
                <div style="background-color:{config['color']}; padding:10px; border-radius:8px; text-align:center; color:white; margin-bottom:10px;">
                    <div style="font-size:10px; opacity:0.8;">口数 {i+1}</div>
                    <div style="font-size:18px; font-weight:bold;">{cand_str}</div>
                </div>
                """, unsafe_allow_html=True)

with tab2:
    report = load_eval_report(selected_loto)
    if report:
        st.subheader("🕵️ Walk-Forward 評価レポート")
        st.caption("時系列の分割テストによる、過去データのリークを含まない厳密な評価指標です。")
        
        metrics_df = []
        # Model
        m = report["Model (LSTM)"]
        metrics_df.append({"モデル": "★ AI Model (LSTM)", "LogLoss (BCE)↓": m["logloss"], "Brier Score↓": m["brier"], "Top-K 正解重なり↑": m["mean_overlap_top_k"]})
        # Baselines
        for b_name, b_metrics in report["Baselines"].items():
            metrics_df.append({"モデル": b_name, "LogLoss (BCE)↓": b_metrics["logloss"], "Brier Score↓": b_metrics["brier"], "Top-K 正解重なり↑": b_metrics["mean_overlap_top_k"]})
            
        st.dataframe(pd.DataFrame(metrics_df).set_index("モデル"), use_container_width=True)
        
        # Calibration
        st.write("##### 信頼度曲線 (Calibration)")
        calib_df = pd.DataFrame(m["calibration"])
        if not calib_df.empty:
            calib_chart = alt.Chart(calib_df).mark_bar(opacity=0.7).encode(
                x='bin_range:O',
                y=alt.Y('pred_prob:Q', title='予測確率 vs 実測確率'),
                color=alt.value('blue')
            )
            calib_line = alt.Chart(calib_df).mark_line(color='red', point=True).encode(
                x='bin_range:O',
                y='true_prob:Q'
            )
            st.altair_chart(calib_chart + calib_line, use_container_width=True)
            st.caption("青棒: モデルの予測確率平均 / 赤線: 実際の当選確率。一致しているほど精度が高い。")
    else:
        st.info("評価レポートが見つかりません。")