import numpy as np

from config import LOOKBACK_WINDOW, LOTO_CONFIG


LEGACY_MODEL_VARIANT = "legacy"
MULTIHOT_MODEL_VARIANT = "multihot"
DEFAULT_MODEL_VARIANT = LEGACY_MODEL_VARIANT
MODEL_VARIANT_CHOICES = (LEGACY_MODEL_VARIANT, MULTIHOT_MODEL_VARIANT)
MODEL_VARIANT_LABELS = {
    LEGACY_MODEL_VARIANT: "Legacy Tabular LSTM",
    MULTIHOT_MODEL_VARIANT: "Multi-Hot Temporal LSTM",
}
MULTIHOT_FEATURE_CHANNELS = ("hit", "frequency", "gap")


def resolve_model_variant(value):
    normalized = (value or DEFAULT_MODEL_VARIANT).strip().lower()
    if normalized not in MODEL_VARIANT_CHOICES:
        raise ValueError(f"unknown model_variant: {value}")
    return normalized


def get_model_variant_label(model_variant):
    return MODEL_VARIANT_LABELS.get(resolve_model_variant(model_variant), model_variant)


def get_target_columns(pick_count):
    return [f"num{i + 1}" for i in range(pick_count)]


def create_multi_hot(target_numbers, max_num):
    target_array = np.asarray(target_numbers, dtype=np.int32)
    vectors = np.zeros((len(target_array), max_num), dtype=np.float32)
    if target_array.size == 0:
        return vectors

    for row_index, numbers in enumerate(target_array):
        for number in numbers.tolist():
            if 1 <= int(number) <= max_num:
                vectors[row_index, int(number) - 1] = 1.0
    return vectors


def build_multihot_feature_names(max_num):
    feature_names = []
    for number in range(1, max_num + 1):
        for channel_name in MULTIHOT_FEATURE_CHANNELS:
            feature_names.append(f"n{number:02d}_{channel_name}")
    return feature_names


def build_number_level_feature_planes(target_numbers, max_num):
    hits = create_multi_hot(target_numbers, max_num)
    frequency = np.zeros_like(hits, dtype=np.float32)
    gap = np.zeros_like(hits, dtype=np.float32)
    counts = np.zeros(max_num, dtype=np.float32)
    gap_state = np.zeros(max_num, dtype=np.float32)

    for row_index in range(len(target_numbers)):
        current_hits = hits[row_index]
        counts += current_hits
        gap_state += 1.0
        gap_state[current_hits > 0.5] = 0.0
        observed_draws = float(row_index + 1)
        frequency[row_index] = counts / observed_draws
        gap[row_index] = gap_state / observed_draws

    return {
        "hit": hits,
        "frequency": frequency,
        "gap": gap,
    }


def build_legacy_row_features(df, pick_count):
    target_cols = get_target_columns(pick_count)
    features_df = df.drop(["draw_id", "date"], axis=1)
    other_cols = [column for column in features_df.columns if column not in target_cols]
    feature_cols = target_cols + other_cols
    feature_frame = features_df[feature_cols]
    return feature_frame.to_numpy(dtype=np.float32), feature_cols, {
        "model_variant": LEGACY_MODEL_VARIANT,
        "feature_strategy": "tabular",
        "feature_channels": [],
        "raw_feature_count": int(len(feature_cols)),
    }


def build_multihot_row_features(df, pick_count, max_num):
    target_numbers = df[get_target_columns(pick_count)].to_numpy(dtype=np.int32)
    planes = build_number_level_feature_planes(target_numbers, max_num)
    blocks = []

    for number_index in range(max_num):
        for channel_name in MULTIHOT_FEATURE_CHANNELS:
            channel = planes[channel_name][:, number_index : number_index + 1]
            blocks.append(channel)

    feature_cols = build_multihot_feature_names(max_num)
    raw_features = np.concatenate(blocks, axis=1).astype(np.float32)
    return raw_features, feature_cols, {
        "model_variant": MULTIHOT_MODEL_VARIANT,
        "feature_strategy": "derived_multihot",
        "feature_channels": list(MULTIHOT_FEATURE_CHANNELS),
        "raw_feature_count": int(raw_features.shape[1]),
    }


def build_row_features(df, loto_type, model_variant):
    config = LOTO_CONFIG[loto_type]
    resolved_variant = resolve_model_variant(model_variant)
    if resolved_variant == LEGACY_MODEL_VARIANT:
        return build_legacy_row_features(df, config["pick_count"])
    return build_multihot_row_features(df, config["pick_count"], config["max_num"])


def prepare_model_dataset(df, loto_type, model_variant):
    config = LOTO_CONFIG[loto_type]
    raw_features, feature_cols, dataset_metadata = build_row_features(df, loto_type, model_variant)
    target_numbers = df[get_target_columns(config["pick_count"])].to_numpy(dtype=np.int32)
    targets = create_multi_hot(target_numbers, config["max_num"])[LOOKBACK_WINDOW:]
    dataset_metadata["lookback_window"] = int(LOOKBACK_WINDOW)
    return {
        "feature_cols": feature_cols,
        "raw_features": raw_features,
        "targets": targets,
        "dataset_metadata": dataset_metadata,
    }


def build_recent_model_input(df, loto_type, model_variant, scaler):
    dataset = prepare_model_dataset(df, loto_type, model_variant)
    raw_features = dataset["raw_features"]
    scaled_features = scaler.transform(raw_features)
    recent_input = np.asarray([scaled_features[-LOOKBACK_WINDOW:]], dtype=np.float32)
    return recent_input, dataset


def resolve_model_variant_from_manifest(manifest):
    training_context = {}
    if isinstance(manifest, dict):
        training_context = manifest.get("training_context") or {}
    try:
        return resolve_model_variant(training_context.get("model_variant", DEFAULT_MODEL_VARIANT))
    except Exception:
        return DEFAULT_MODEL_VARIANT


def resolve_feature_strategy_from_manifest(manifest):
    training_context = {}
    if isinstance(manifest, dict):
        training_context = manifest.get("training_context") or {}
    return training_context.get("feature_strategy", "tabular")
