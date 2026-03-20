import numpy as np

# --- 定数・設定 ---
LOTO_CONFIG = {
    "miniloto": {"name": "ミニロト", "max_num": 31, "pick_count": 5, "sum_range": [60, 100], "max_parity_diff": 3, "color": "#ec4899"},
    "loto6": {"name": "ロト6", "max_num": 43, "pick_count": 6, "sum_range": [115, 150], "max_parity_diff": 2, "color": "#3b82f6"},
    "loto7": {"name": "ロト7", "max_num": 37, "pick_count": 7, "sum_range": [110, 155], "max_parity_diff": 3, "color": "#f59e0b"}
}

LOOKBACK_WINDOW = 10
ARTIFACT_SCHEMA_VERSION = 3
MANIFEST_SCHEMA_VERSION = 3
EVAL_REPORT_SCHEMA_VERSION = 4
RUN_TRACKING_SCHEMA_VERSION = 1
PREPROCESSING_VERSION = "feature_engineering_v2_multihot"
PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43}

# --- サンプリング関数 ---
def get_top_k_prediction(probs, k):
    """最も確率の高い上位k個を固定で選択する（Top-k）"""
    idx = np.argsort(probs)[-k:]
    return sorted(idx + 1)

def weighted_sample_without_replacement(probs, k):
    """確率ベクトルに基づく重複なし重み付き抽出"""
    p = np.clip(probs, 1e-9, 1.0)
    p = p / np.sum(p)
    idx = np.random.choice(np.arange(len(probs)), size=k, replace=False, p=p)
    return sorted(idx + 1)

# --- フィルタリング条件 ---
def has_arithmetic_progression(nums):
    """等差数列（例: 5, 10, 15）の有無を判定"""
    for i in range(len(nums)-2):
        for j in range(i+1, len(nums)-1):
            diff = nums[j] - nums[i]
            if nums[j] + diff in nums:
                return True
    return False

def check_statistical_filters(nums, config):
    """基本統計フィルタ（合計値、奇数偶数バランス）"""
    s_val = sum(nums)
    if not (config["sum_range"][0] <= s_val <= config["sum_range"][1]):
        return False
    odd_count = sum(1 for n in nums if n % 2 != 0)
    even_count = config["pick_count"] - odd_count
    if abs(odd_count - even_count) > config["max_parity_diff"]:
        return False
    return True

def check_psychological_filters(nums, config):
    """期待値最大化のための心理逆張りフィルタ"""
    # 1. 連続数字の強制
    if not any(nums[i+1] - nums[i] == 1 for i in range(len(nums)-1)):
        return False
    # 2. カレンダー外数字(32以上)の強制 (ロト6, ロト7)
    if config["name"] in ["ロト6", "ロト7"]:
        if not any(n >= 32 for n in nums):
            return False
    # 3. 素数の強制
    if sum(1 for n in nums if n in PRIMES) == 0:
        return False
    # 4. 等差数列の排除
    if has_arithmetic_progression(nums):
        return False
    return True

def generate_valid_sample(probs, config, use_psychological=True, use_statistical=True, max_attempts=1000, sampling_mode="weighted"):
    """フィルタ条件を満たすまでサンプリングを繰り返す"""
    for _ in range(max_attempts):
        if sampling_mode == "top-k":
            # top-kは乱数要素がないため、1回で確定させる
            cand = get_top_k_prediction(probs, config["pick_count"])
            # top-kの場合はフィルタを通らなくてもそのまま返す（最尤推定として）
            return cand
        else:
            cand = weighted_sample_without_replacement(probs, config["pick_count"])
            
        is_valid = True
        if use_statistical and not check_statistical_filters(cand, config):
            is_valid = False
        if use_psychological and not check_psychological_filters(cand, config):
            is_valid = False
            
        if is_valid:
            return cand
            
    # 上限に達した場合は、フィルタを無視して1回抽出して返す
    return weighted_sample_without_replacement(probs, config["pick_count"])
