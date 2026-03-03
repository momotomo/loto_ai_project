import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import warnings
import altair as alt

# --- Mac環境安定化設定 ---
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
tf.config.set_visible_devices([], 'GPU')

# --- ページ設定 ---
st.set_page_config(page_title="宝くじAI予測 (Advanced)", page_icon="🎯", layout="wide")

# --- 定数・設定 ---
LOTO_CONFIG = {
    "miniloto": {"name": "ミニロト", "max_num": 31, "pick_count": 5, "sum_range": [60, 100], "max_parity_diff": 3, "color": "#ec4899"},
    "loto6": {"name": "ロト6", "max_num": 43, "pick_count": 6, "sum_range": [115, 150], "max_parity_diff": 2, "color": "#3b82f6"},
    "loto7": {"name": "ロト7", "max_num": 37, "pick_count": 7, "sum_range": [110, 155], "max_parity_diff": 3, "color": "#f59e0b"}
}
LOOKBACK_WINDOW = 10

# --- 関数 ---
@st.cache_data
def load_data(loto_type):
    data_file = f"data/{loto_type}_processed.csv"
    if os.path.exists(data_file):
        df = pd.read_csv(data_file)
        return df
    return None

@st.cache_resource
def get_model(loto_type):
    model_path = f"models/{loto_type}_lstm_best.keras"
    if os.path.exists(model_path):
        return load_model(model_path, compile=False)
    return None

def prepare_data_for_ai(df, config):
    target_cols = [f"num{i+1}" for i in range(config["pick_count"])]
    features_df = df.drop(['draw_id', 'date'], axis=1, errors='ignore')
    
    other_cols = [c for c in features_df.columns if c not in target_cols]
    ordered_cols = target_cols + other_cols
    features_df = features_df[ordered_cols]
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(features_df)
    return scaled_data, scaler, ordered_cols

def calculate_next_draw_date(loto_type, last_date_str):
    """次回抽選日を計算する関数"""
    last_date = pd.to_datetime(last_date_str.replace('/', '-'))
    today = pd.to_datetime(datetime.now().date())
    
    # データが最新(今日以降)なら明日以降から探す。過去のデータなら今日から探す。
    start_date = last_date + pd.Timedelta(days=1) if last_date >= today else today
    
    # 抽選曜日設定 (月=0, 火=1, 水=2, 木=3, 金=4, 土=5, 日=6)
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
# メイン UI
# ==========================================
st.title("🎯 宝くじ AI予測 & 全期間分析ダッシュボード")
st.markdown("Colabで生成した「増強特徴量モデル」を用いた一貫した解析システムです。")

st.markdown("---")

# 1. 宝くじ選択
selected_loto = st.radio(
    "宝くじの種類を選択してください",
    options=["miniloto", "loto6", "loto7"],
    format_func=lambda x: LOTO_CONFIG[x]["name"],
    horizontal=True
)

config = LOTO_CONFIG[selected_loto]
df = load_data(selected_loto)
model = get_model(selected_loto)

if df is None or model is None:
    st.error("⚠️ モデルまたはデータが見つかりません。Colabで作成したファイルを data/ および models/ に配置してください。")
    st.stop()

# 共通データの準備
scaled_data, scaler, col_names = prepare_data_for_ai(df, config)
target_cols = [f"num{i+1}" for i in range(config["pick_count"])]

# --- タブ構成 ---
tab1, tab2, tab3 = st.tabs(["🔮 次回予測", "📈 強化トレンド・散布図分析", "🕵️ 精度検証 (バックテスト)"])

# ==========================================
# タブ1: 次回予測 (答え合わせ機能追加)
# ==========================================
with tab1:
    last_draw_id = int(df.iloc[-1]['draw_id'])
    last_draw_date = df.iloc[-1]['date']
    next_date, next_weekday = calculate_next_draw_date(selected_loto, last_draw_date)
    
    # --- ヘッダー情報（最新回と次回抽選日） ---
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.info(f"💾 **取得済みの最新データ:** 第{last_draw_id}回 ({last_draw_date})")
    with col_info2:
        st.success(f"🗓️ **次回抽選予定日:** {next_date} ({next_weekday})")

    # --- 直近の答え合わせ ---
    if len(df) > LOOKBACK_WINDOW:
        with st.expander(f"🔍 直近（第{last_draw_id}回）のAI予測はどうだった？（答え合わせ）"):
            # 最新回の実際の番号
            actual_nums = [int(n) for n in df.iloc[-1][target_cols].tolist()]
            
            # その1回前のデータを使って推論
            start_idx = len(df) - 1 - LOOKBACK_WINDOW
            end_idx = len(df) - 1
            test_input = np.array([scaled_data[start_idx:end_idx]], dtype=np.float32)
            p_scaled = model(tf.convert_to_tensor(test_input), training=False).numpy()
            
            dummy = np.zeros((1, len(col_names)))
            dummy[0, :config["pick_count"]] = p_scaled[0]
            p_raw = scaler.inverse_transform(dummy)[0, :config["pick_count"]]
            
            # AIの基本予測
            prediction = sorted([int(n) for n in np.round(np.clip(p_raw, 1, config["max_num"]))])
            hits = sorted(list(set(prediction) & set(actual_nums)))
            hit_count = len(hits)
            
            st.write(f"**実際の当選番号:** `{', '.join(map(str, actual_nums))}`")
            pred_display = [f"**{n}**" if n in hits else str(n) for n in prediction]
            st.write(f"**当時のAIベース予測:** {', '.join(pred_display)}")
            
            if hits:
                st.success(f"✨ **{hit_count}個的中！** (的中数字: {', '.join(map(str, hits))})")
            else:
                st.write("残念ながら的中数字はありませんでした。")
            st.caption("※期待値フィルタ適用前のAIの純粋な予測値による検証です。")

    st.markdown("---")
    st.subheader(f"✨ 第{last_draw_id + 1}回 ({next_date}) 向けのAI予測")
    
    # 【アプローチ2】期待値最大化モードのトグル
    st.markdown("##### 🧠 予測戦略の選択")
    use_expected_value_mode = st.checkbox(
        "💰 **期待値（配当）最大化モード** (他人が買いにくい組み合わせを狙い、当選時の独占を狙う)", 
        value=True,
        help="人間が好む「誕生日(1〜31)」や「散らばった数字」を避け、あえて「連続数字」や「32以上の数字」を組み込みます。"
    )
    
    if st.button(f"第{last_draw_id + 1}回の予測を実行する", type="primary", use_container_width=True):
        with st.spinner('最新データを解析中...'):
            try:
                recent_input = np.array([scaled_data[-LOOKBACK_WINDOW:]], dtype=np.float32)
                scaled_pred = model(tf.convert_to_tensor(recent_input), training=False).numpy()
                
                dummy = np.zeros((1, len(col_names)))
                dummy[0, :config["pick_count"]] = scaled_pred[0]
                raw_pred = scaler.inverse_transform(dummy)[0, :config["pick_count"]]
                
                results = []
                for _ in range(5000):
                    noise = np.random.uniform(-2.5, 2.5, size=config["pick_count"])
                    cand = np.round(np.clip(raw_pred + noise, 1, config["max_num"])).astype(int)
                    cand.sort()
                    
                    if len(set(cand)) == config["pick_count"]:
                        s_val = sum(cand)
                        odd = sum(1 for n in cand if n % 2 != 0)
                        
                        if config["sum_range"][0] <= s_val <= config["sum_range"][1] and \
                           abs(odd - (config["pick_count"] - odd)) <= config["max_parity_diff"]:
                            
                            if use_expected_value_mode:
                                if selected_loto in ["loto6", "loto7"]:
                                    if not any(n >= 32 for n in cand):
                                        continue
                                has_consecutive = any(cand[i+1] - cand[i] == 1 for i in range(len(cand)-1))
                                if not has_consecutive:
                                    continue
                            results.append(cand.tolist())
                
                unique_results = []
                for r in results:
                    if r not in unique_results: unique_results.append(r)
                
                if unique_results:
                    if use_expected_value_mode:
                        st.success(f"💰 **期待値重視！** 他人が敬遠しがちな、配当が高くなりやすい組み合わせ（上位{min(len(unique_results), 5)}件）です。")
                    else:
                        st.success(f"🎯 AIが算出した標準的な推奨買い目（上位{min(len(unique_results), 5)}件）です。")
                        
                    cols = st.columns(min(len(unique_results), 5))
                    for i, match in enumerate(unique_results[:5]):
                        with cols[i]:
                            match_str = ", ".join(map(str, [int(n) for n in match]))
                            st.markdown(f"""
                            <div style="background-color:{config['color']}; padding:15px; border-radius:12px; text-align:center; color:white;">
                                <div style="font-size:11px; opacity:0.8;">推奨 {i+1}</div>
                                <div style="font-size:22px; font-weight:bold; margin:8px 0;">{match_str}</div>
                                <div style="font-size:11px;">合計: {sum(match)}</div>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.warning("条件を満たす候補が見つかりませんでした。もう一度実行するか、モードをOFFにしてください。")
            except Exception as e:
                st.error(f"エラー: {e}")

# ==========================================
# タブ2: 強化トレンド分析・全期間散布図
# ==========================================
with tab2:
    st.header(f"📊 {config['name']} 統計解析")
    
    st.subheader("📍 全期間の番号出現散布図")
    st.markdown(f"**横軸: 1〜{config['max_num']}** 、 **縦軸: 上が第1回 〜 下が第{df['draw_id'].max()}回（最新）** で固定しています。")
    
    scatter_df = df[['draw_id'] + target_cols].melt(id_vars=['draw_id'], value_name='当選番号')
    
    chart = alt.Chart(scatter_df).mark_circle(size=40, opacity=0.5).encode(
        x=alt.X('当選番号:Q', 
                title='当選番号', 
                scale=alt.Scale(domain=[1, config["max_num"]]),
                axis=alt.Axis(tickCount=config["max_num"] // 2)),
        y=alt.Y('draw_id:Q', 
                title='抽選回 (第n回)', 
                scale=alt.Scale(domain=[1, df['draw_id'].max()], reverse=True)),
        color=alt.value(config["color"]),
        tooltip=['draw_id', '当選番号']
    ).properties(
        height=1200, 
        width='container'
    )

    st.altair_chart(chart, use_container_width=True)
    st.info(f"💡 縦軸は上が第1回、下が最新回です。横軸は1から最大数字までで固定されています。")
    st.markdown("---")
    
    st.subheader("🔢 番号出現頻度ランキング")
    all_numbers_series = df[target_cols].stack()
    distribution = all_numbers_series.value_counts().sort_index()
    
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        most_common = distribution.idxmax()
        st.success(f"🏆 最も多く出ている数字: **{most_common}** ({distribution.max()}回)")
    with col_info2:
        least_common = distribution.idxmin()
        st.warning(f"🧊 最も出ていない数字: **{least_common}** ({distribution.min()}回)")

    st.markdown("---")
    
    st.subheader("📈 直近60回の特徴量トレンド")
    recent_df = df.tail(60).copy()
    recent_df['draw_id'] = recent_df['draw_id'].astype(str)
    
    def safe_chart(cols, title, caption, chart_type='line'):
        available = [c for c in cols if c in recent_df.columns]
        if available:
            st.write(f"**{title}**")
            st.caption(caption)
            if chart_type == 'line': st.line_chart(recent_df.set_index('draw_id')[available])
            elif chart_type == 'bar': st.bar_chart(recent_df.set_index('draw_id')[available])
            elif chart_type == 'area': st.area_chart(recent_df.set_index('draw_id')[available])

    col_a, col_b = st.columns(2)
    with col_a:
        safe_chart(['sum_val', 'sum_ma5'], "1. 合計値とトレンド", "合計値の推移と直近5回平均")
        safe_chart(['consecutive_pairs'], "2. 連続数字ペア数", "隣接する数字の出現傾向", 'bar')
    with col_b:
        safe_chart(['last_digit_sum'], "3. 下一桁合計", "一の位の偏り分析")
        safe_chart(['max_min_diff'], "4. 数値レンジ", "数字の広がり（最大-最小）", 'area')

# ==========================================
# タブ3: 精度検証
# ==========================================
with tab3:
    st.subheader("🕵️ バックテスト・シミュレーター")
    st.markdown("過去の実際の当選番号に対して、その一回前の時点のAIモデルで「予測していたらどうなっていたか」を検証します。")
    
    num_test = st.slider("検証する過去の回数を選択", 1, 20, 5)
    
    if st.button("過去検証を開始", use_container_width=True):
        with st.spinner('過去の予測をシミュレーション中...'):
            for i in range(num_test, 0, -1):
                target_idx = len(df) - i
                if target_idx < LOOKBACK_WINDOW: continue
                
                actual_draw_id = int(df.iloc[target_idx]['draw_id'])
                actual_date = df.iloc[target_idx]['date']
                actual_nums = [int(n) for n in df.iloc[target_idx][target_cols].tolist()]
                
                start_idx = target_idx - LOOKBACK_WINDOW
                test_input = np.array([scaled_data[start_idx:target_idx]], dtype=np.float32)
                p_scaled = model(tf.convert_to_tensor(test_input), training=False).numpy()
                
                dummy = np.zeros((1, len(col_names)))
                dummy[0, :config["pick_count"]] = p_scaled[0]
                p_raw = scaler.inverse_transform(dummy)[0, :config["pick_count"]]
                
                prediction = sorted([int(n) for n in np.round(np.clip(p_raw, 1, config["max_num"]))])
                hits = sorted(list(set(prediction) & set(actual_nums)))
                hit_count = len(hits)
                
                with st.expander(f"第 {actual_draw_id} 回 ({actual_date}) - {hit_count}個 的中"):
                    st.write(f"**実際の当選番号:** `{', '.join(map(str, actual_nums))}`")
                    pred_display = []
                    for n in prediction:
                        if n in hits: pred_display.append(f"**{n}**")
                        else: pred_display.append(str(n))
                    st.write(f"**AIの当時の予測:** {', '.join(pred_display)}")
                    if hits: st.success(f"的中した数字: {', '.join(map(str, hits))}")
                    else: st.info("的中した数字はありませんでした。")

    st.markdown("---")
    st.caption("※この検証は、モデルが学習時に見ていなかった将来のデータを予測する形式（バックテスト）で行っています。")