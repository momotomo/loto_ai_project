import streamlit as st
import pandas as pd
import numpy as np
import os
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
st.set_page_config(page_title="宝くじAI予測 (Advanced)", page_icon="🧬", layout="wide")

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
    last_date = pd.to_datetime(last_date_str.replace('/', '-'))
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
# 🧬 遺伝的アルゴリズム (GA) エンジン
# ==========================================
def genetic_algorithm_search(raw_pred, config, use_expected_value_mode, pop_size=200, generations=60):
    """AIの予測値を初期集団とし、交叉と突然変異を繰り返して最適解を進化させる"""
    PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43}
    
    # 1. 初期集団の生成
    population = []
    for _ in range(pop_size):
        noise = np.random.uniform(-3.5, 3.5, size=config["pick_count"])
        cand = np.round(np.clip(raw_pred + noise, 1, config["max_num"])).astype(int)
        cand = list(set(cand))
        while len(cand) < config["pick_count"]:
            new_num = np.random.randint(1, config["max_num"] + 1)
            if new_num not in cand: cand.append(new_num)
        cand.sort()
        population.append(cand)

    # 2. 適応度 (Fitness) 評価関数
    def calculate_fitness(ind):
        score = 3000.0 # 初期スコア
        mae = np.mean(np.abs(np.array(ind) - raw_pred))
        score -= mae * 10 # AIの原案から離れるほど微小な減点
        
        # [必須フィルタ] ガウス和制約・パリティ分散
        s_val = sum(ind)
        if not (config["sum_range"][0] <= s_val <= config["sum_range"][1]):
            score -= 1000
        odd = sum(1 for n in ind if n % 2 != 0)
        if abs(odd - (config["pick_count"] - odd)) > config["max_parity_diff"]:
            score -= 1000
            
        # [高度心理バイアスフィルタ] 期待値最大化モード
        if use_expected_value_mode:
            # ① 連続数字の強制
            has_consecutive = any(ind[i+1] - ind[i] == 1 for i in range(len(ind)-1))
            if not has_consecutive: score -= 500
                
            # ② カレンダー外数字 (32以上) の強制
            if config["name"] in ["ロト6", "ロト7"]:
                if not any(n >= 32 for n in ind): score -= 500
                    
            # ③ 素数の強制 (人間が嫌う中途半端な数字)
            prime_count = sum(1 for n in ind if n in PRIMES)
            if prime_count == 0: score -= 500
                
            # ④ 等間隔数列 (等差数列) の排除 (人間が塗りやすい規則的なパターン)
            has_arithmetic = False
            for i in range(len(ind)-2):
                for j in range(i+1, len(ind)-1):
                    diff = ind[j] - ind[i]
                    if ind[j] + diff in ind:
                        has_arithmetic = True
                        break
                if has_arithmetic: break
            if has_arithmetic: score -= 500
                
        return score

    # 3. 進化ループ
    for gen in range(generations):
        fitnesses = [calculate_fitness(ind) for ind in population]
        sorted_indices = np.argsort(fitnesses)[::-1]
        
        # エリート保存 (上位10%はそのまま次世代へ)
        new_population = [population[i] for i in sorted_indices[:int(pop_size * 0.1)]]
        
        # ルーレット選択のための重み付け
        weights = np.array(fitnesses)
        weights = np.where(weights < 0, 1e-5, weights) 
        weights = weights / sum(weights) if sum(weights) > 0 else np.ones(pop_size) / pop_size
        
        # 交叉と突然変異
        while len(new_population) < pop_size:
            p1 = population[np.random.choice(pop_size, p=weights)]
            p2 = population[np.random.choice(pop_size, p=weights)]
            
            # 交叉 (親の遺伝子を半分ずつ受け継ぐ)
            split_idx = config["pick_count"] // 2
            child = list(set(p1[:split_idx] + p2[split_idx:]))
            
            # 遺伝子の修復 (重複排除・数合わせ)
            while len(child) < config["pick_count"]:
                new_num = np.random.randint(1, config["max_num"] + 1)
                if new_num not in child: child.append(new_num)
            while len(child) > config["pick_count"]:
                child.pop(np.random.randint(len(child)))
                
            # 突然変異 (15%の確率で遺伝子の一部がランダム変化)
            if np.random.rand() < 0.15:
                idx = np.random.randint(config["pick_count"])
                child.pop(idx)
                while len(child) < config["pick_count"]:
                    new_num = np.random.randint(1, config["max_num"] + 1)
                    if new_num not in child: child.append(new_num)
                        
            child.sort()
            new_population.append(child)
            
        population = new_population
        
    # 4. 最終世代から優秀な個体(条件クリア)を抽出
    fitnesses = [calculate_fitness(ind) for ind in population]
    sorted_indices = np.argsort(fitnesses)[::-1]
    
    best_results = []
    seen = set()
    for i in sorted_indices:
        cand = tuple(population[i])
        # スコアが2000以上 (ペナルティを1つも受けていない完全体) のみを抽出
        if cand not in seen and fitnesses[i] >= 2000:
            seen.add(cand)
            best_results.append(list(cand))
            
    return best_results

# ==========================================
# メイン UI
# ==========================================
st.title("🎯 宝くじ AI予測 & 全期間分析ダッシュボード")
st.markdown("Colabで生成した「DAE(ノイズ除去) + LSTMモデル」と「遺伝的アルゴリズム(GA)」を組み合わせた最先端の解析システムです。")

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

scaled_data, scaler, col_names = prepare_data_for_ai(df, config)
target_cols = [f"num{i+1}" for i in range(config["pick_count"])]

# --- タブ構成 ---
tab1, tab2, tab3 = st.tabs(["🧬 遺伝的アルゴリズム予測", "📈 強化トレンド・散布図分析", "🕵️ 精度検証 (バックテスト)"])

# ==========================================
# タブ1: 遺伝的アルゴリズム予測
# ==========================================
with tab1:
    last_draw_id = int(df.iloc[-1]['draw_id'])
    last_draw_date = df.iloc[-1]['date']
    next_date, next_weekday = calculate_next_draw_date(selected_loto, last_draw_date)
    
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.info(f"💾 **取得済みの最新データ:** 第{last_draw_id}回 ({last_draw_date})")
    with col_info2:
        st.success(f"🗓️ **次回抽選予定日:** {next_date} ({next_weekday})")

    # --- 直近の答え合わせ ---
    if len(df) > LOOKBACK_WINDOW:
        with st.expander(f"🔍 直近（第{last_draw_id}回）のAI純粋予測はどうだった？"):
            actual_nums = [int(n) for n in df.iloc[-1][target_cols].tolist()]
            
            start_idx = len(df) - 1 - LOOKBACK_WINDOW
            end_idx = len(df) - 1
            test_input = np.array([scaled_data[start_idx:end_idx]], dtype=np.float32)
            p_scaled = model(tf.convert_to_tensor(test_input), training=False).numpy()
            
            dummy = np.zeros((1, len(col_names)))
            dummy[0, :config["pick_count"]] = p_scaled[0]
            p_raw = scaler.inverse_transform(dummy)[0, :config["pick_count"]]
            
            prediction = sorted([int(n) for n in np.round(np.clip(p_raw, 1, config["max_num"]))])
            hits = sorted(list(set(prediction) & set(actual_nums)))
            
            st.write(f"**実際の当選番号:** `{', '.join(map(str, actual_nums))}`")
            pred_display = [f"**{n}**" if n in hits else str(n) for n in prediction]
            st.write(f"**当時のAIベース予測 (DAE経由):** {', '.join(pred_display)}")
            
            if hits: st.success(f"✨ **{len(hits)}個的中！** (的中数字: {', '.join(map(str, hits))})")
            else: st.write("残念ながら的中数字はありませんでした。")
            st.caption("※遺伝的アルゴリズムや期待値フィルタ適用前の、AIの純粋な予測値による検証です。")

    st.markdown("---")
    st.subheader(f"✨ 第{last_draw_id + 1}回 ({next_date}) 向けのAI＆GA予測")
    
    st.markdown("##### 🧠 予測戦略の選択")
    use_expected_value_mode = st.checkbox(
        "💰 **期待値（配当）最大化モード** (人間心理のバイアス逆張り戦略)", 
        value=True,
        help="【導入済みロジック】32以上を含む / 連続数字を含む / 素数を含める / 人間が塗りやすい「等間隔数列」の排除"
    )
    
    if st.button(f"遺伝的アルゴリズム (GA) で第{last_draw_id + 1}回の最適解を生成", type="primary", use_container_width=True):
        with st.spinner('DAE+LSTMで予測推論し、遺伝的アルゴリズムで進化計算中...'):
            try:
                # 1. DAE+LSTM によるベース推論
                recent_input = np.array([scaled_data[-LOOKBACK_WINDOW:]], dtype=np.float32)
                scaled_pred = model(tf.convert_to_tensor(recent_input), training=False).numpy()
                
                dummy = np.zeros((1, len(col_names)))
                dummy[0, :config["pick_count"]] = scaled_pred[0]
                raw_pred = scaler.inverse_transform(dummy)[0, :config["pick_count"]]
                
                # 2. 遺伝的アルゴリズム (GA) による最適候補の探索・進化
                unique_results = genetic_algorithm_search(raw_pred, config, use_expected_value_mode)
                
                if unique_results:
                    if use_expected_value_mode:
                        st.success(f"🧬 **進化完了 (期待値重視)！** 高度心理バイアスフィルタをすべてクリアした最強の組み合わせ（上位{min(len(unique_results), 5)}件）です。")
                    else:
                        st.success(f"🧬 **進化完了！** 統計フィルタをクリアした最適解（上位{min(len(unique_results), 5)}件）です。")
                        
                    cols = st.columns(min(len(unique_results), 5))
                    for i, match in enumerate(unique_results[:5]):
                        with cols[i]:
                            match_str = ", ".join(map(str, [int(n) for n in match]))
                            st.markdown(f"""
                            <div style="background-color:{config['color']}; padding:15px; border-radius:12px; text-align:center; color:white;">
                                <div style="font-size:11px; opacity:0.8;">世代進化 エリート {i+1}</div>
                                <div style="font-size:22px; font-weight:bold; margin:8px 0;">{match_str}</div>
                                <div style="font-size:11px;">合計: {sum(match)}</div>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.warning("設定された世代数内で条件を満たす「完全体」に進化できませんでした。もう一度実行するか、モードをOFFにしてください。")
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
                    st.write(f"**AIの当時のベース予測:** {', '.join(pred_display)}")
                    if hits: st.success(f"的中した数字: {', '.join(map(str, hits))}")
                    else: st.info("的中した数字はありませんでした。")

    st.markdown("---")
    st.caption("※この検証は、モデルが学習時に見ていなかった将来のデータを予測する形式（バックテスト）で行っています。")