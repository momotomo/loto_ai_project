import numpy as np

from config import LOOKBACK_WINDOW, LOTO_CONFIG


LEGACY_MODEL_VARIANT = "legacy"
MULTIHOT_MODEL_VARIANT = "multihot"
DEEPSETS_MODEL_VARIANT = "deepsets"
DEFAULT_MODEL_VARIANT = LEGACY_MODEL_VARIANT
MODEL_VARIANT_CHOICES = (LEGACY_MODEL_VARIANT, MULTIHOT_MODEL_VARIANT, DEEPSETS_MODEL_VARIANT)
MODEL_VARIANT_LABELS = {
    LEGACY_MODEL_VARIANT: "Legacy Tabular LSTM",
    MULTIHOT_MODEL_VARIANT: "Multi-Hot Temporal LSTM",
    DEEPSETS_MODEL_VARIANT: "Deep Sets Sequence Encoder",
}
MULTIHOT_FEATURE_CHANNELS = ("hit", "frequency", "gap")
DEEPSETS_ELEMENT_FEATURE_CHANNELS = ("number_norm", "frequency", "gap")


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


def build_deepsets_feature_names(pick_count):
    feature_names = []
    for slot_index in range(pick_count):
        for channel_name in DEEPSETS_ELEMENT_FEATURE_CHANNELS:
            feature_names.append(f"slot{slot_index + 1:02d}_{channel_name}")
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


def build_deepsets_row_features(df, pick_count, max_num):
    target_numbers = df[get_target_columns(pick_count)].to_numpy(dtype=np.int32)
    sorted_numbers = np.sort(target_numbers, axis=1)
    planes = build_number_level_feature_planes(target_numbers, max_num)
    row_indices = np.arange(len(df))[:, None]
    selected_indices = sorted_numbers - 1
    number_norm = (sorted_numbers.astype(np.float32) - 1.0) / max(float(max_num - 1), 1.0)
    frequency = planes["frequency"][row_indices, selected_indices]
    gap = planes["gap"][row_indices, selected_indices]
    element_tensor = np.stack([number_norm, frequency, gap], axis=-1).astype(np.float32)
    raw_features = element_tensor.reshape(len(df), -1)
    feature_cols = build_deepsets_feature_names(pick_count)
    input_summary = {
        "prepared_input_rank": 4,
        "lookback_window": int(LOOKBACK_WINDOW),
        "set_cardinality": int(pick_count),
        "element_feature_count": int(element_tensor.shape[-1]),
        "flattened_feature_count": int(raw_features.shape[1]),
        "scaler_feature_count": int(element_tensor.shape[-1]),
        "pooling": "mean",
        "temporal_head": "lstm",
        "element_feature_channels": list(DEEPSETS_ELEMENT_FEATURE_CHANNELS),
    }
    return raw_features, feature_cols, {
        "model_variant": DEEPSETS_MODEL_VARIANT,
        "feature_strategy": "set_sequence_deepsets",
        "feature_channels": list(DEEPSETS_ELEMENT_FEATURE_CHANNELS),
        "raw_feature_count": int(raw_features.shape[1]),
        "row_shape": [int(pick_count), int(element_tensor.shape[-1])],
        "set_cardinality": int(pick_count),
        "element_feature_count": int(element_tensor.shape[-1]),
        "scaler_feature_count": int(element_tensor.shape[-1]),
        "prepared_input_rank": 4,
        "pooling": "mean",
        "lookback_integration": "sequence_lstm",
        "input_summary": input_summary,
    }


def build_row_features(df, loto_type, model_variant):
    config = LOTO_CONFIG[loto_type]
    resolved_variant = resolve_model_variant(model_variant)
    if resolved_variant == LEGACY_MODEL_VARIANT:
        return build_legacy_row_features(df, config["pick_count"])
    if resolved_variant == MULTIHOT_MODEL_VARIANT:
        return build_multihot_row_features(df, config["pick_count"], config["max_num"])
    return build_deepsets_row_features(df, config["pick_count"], config["max_num"])


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


def get_scaler_feature_count(dataset_metadata, feature_cols):
    if isinstance(dataset_metadata, dict) and dataset_metadata.get("scaler_feature_count") is not None:
        return int(dataset_metadata["scaler_feature_count"])
    return int(len(feature_cols))


def reshape_flat_rows_for_model(flat_rows, dataset_metadata):
    metadata = dataset_metadata or {}
    row_shape = metadata.get("row_shape")
    if row_shape:
        array = np.asarray(flat_rows, dtype=np.float32)
        return array.reshape(array.shape[0], int(row_shape[0]), int(row_shape[1]))
    return np.asarray(flat_rows, dtype=np.float32)


def flatten_model_rows(model_rows):
    array = np.asarray(model_rows, dtype=np.float32)
    if array.ndim <= 2:
        return array
    return array.reshape(array.shape[0], -1)


def fit_scaler_for_variant(raw_features, dataset_metadata):
    scaler_input = reshape_flat_rows_for_model(raw_features, dataset_metadata)
    if scaler_input.ndim == 3:
        scaler_input = scaler_input.reshape(-1, scaler_input.shape[-1])
    return scaler_input.astype(np.float32)


def transform_features_for_variant(raw_features, scaler, dataset_metadata):
    scaler_input = reshape_flat_rows_for_model(raw_features, dataset_metadata)
    if scaler_input.ndim == 3:
        scaled = scaler.transform(scaler_input.reshape(-1, scaler_input.shape[-1]))
        return scaled.reshape(scaler_input.shape[0], -1).astype(np.float32)
    return scaler.transform(np.asarray(raw_features, dtype=np.float32)).astype(np.float32)


def build_model_samples_from_scaled_rows(scaled_features, sample_start, sample_end, dataset_metadata):
    X = [scaled_features[index : index + LOOKBACK_WINDOW] for index in range(sample_start, sample_end)]
    samples = np.asarray(X, dtype=np.float32)
    metadata = dataset_metadata or {}
    row_shape = metadata.get("row_shape")
    if row_shape and samples.size:
        return samples.reshape(samples.shape[0], LOOKBACK_WINDOW, int(row_shape[0]), int(row_shape[1]))
    if row_shape:
        return np.zeros((0, LOOKBACK_WINDOW, int(row_shape[0]), int(row_shape[1])), dtype=np.float32)
    return samples


def build_recent_model_input(df, loto_type, model_variant, scaler):
    dataset = prepare_model_dataset(df, loto_type, model_variant)
    raw_features = dataset["raw_features"]
    scaled_features = transform_features_for_variant(raw_features, scaler, dataset["dataset_metadata"])
    recent_input = build_model_samples_from_scaled_rows(
        scaled_features,
        len(scaled_features) - LOOKBACK_WINDOW,
        len(scaled_features) - LOOKBACK_WINDOW + 1,
        dataset["dataset_metadata"],
    )
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
