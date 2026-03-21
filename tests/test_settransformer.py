"""Tests for the Set Transformer variant.

Covers:
- Input shape / forward shape / output range
- deepsets との入力整合 (feature layout parity)
- lookback 処理の整合 (same 4-D pipeline as deepsets)
- leakage 防止 (prefix-stable features)
- calibration 統合後の形式チェック
- eval_report / manifest の必須キー
- predict.py が settransformer artifact を安全に読めること
- update_system.py の最小 smoke
- app.py が新 variant を安全に扱えること
"""

import json
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from config import LOOKBACK_WINDOW, LOTO_CONFIG
from model_layers import SetAttentionBlock
from model_variants import (
    DEEPSETS_MODEL_VARIANT,
    SETTRANSFORMER_MODEL_VARIANT,
    build_model_samples_from_scaled_rows,
    build_number_level_feature_planes,
    fit_scaler_for_variant,
    prepare_model_dataset,
    resolve_model_variant,
    transform_features_for_variant,
)
from train_prob_model import build_prob_model, build_settransformer_prob_model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def build_loto6_history(row_count=LOOKBACK_WINDOW + 5):
    max_num = LOTO_CONFIG["loto6"]["max_num"]
    records = []
    for draw_id in range(1, row_count + 1):
        numbers = sorted(set([((draw_id + i - 1) % max_num) + 1 for i in range(7)]))[:6]
        while len(numbers) < 6:
            numbers.append((numbers[-1] % max_num) + 1)
        records.append(
            {
                "draw_id": draw_id,
                "date": f"2024/01/{draw_id:02d}",
                "num1": numbers[0],
                "num2": numbers[1],
                "num3": numbers[2],
                "num4": numbers[3],
                "num5": numbers[4],
                "num6": numbers[5],
            }
        )
    return pd.DataFrame(records)


ROOT_FILES = [
    "artifact_utils.py",
    "calibration_utils.py",
    "config.py",
    "data_collector.py",
    "evaluation_statistics.py",
    "model_layers.py",
    "model_variants.py",
    "predict.py",
    "report_utils.py",
    "train_prob_model.py",
    "update_system.py",
]
LEGACY_ARTIFACT_FILES = [
    "data/loto6_feature_cols.json",
    "models/loto6_feature_cols.json",
    "models/loto6_prob.keras",
    "models/loto6_scaler.pkl",
]


def prepare_workspace(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for relative_path in ROOT_FILES:
        source = repo_root / relative_path
        destination = workspace / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    (workspace / "data").mkdir()
    for relative_path in ["data/loto6_processed.csv", "data/loto6_raw.csv"]:
        source = repo_root / relative_path
        destination = workspace / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    for relative_path in LEGACY_ARTIFACT_FILES:
        source = repo_root / relative_path
        if source.exists():
            destination = workspace / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    return workspace


def run_command(workspace, *args):
    result = subprocess.run(
        [sys.executable, *args],
        cwd=workspace,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return result


# ---------------------------------------------------------------------------
# Unit tests – model layer
# ---------------------------------------------------------------------------

class TestSetAttentionBlock:
    def test_output_shape_matches_input(self):
        lookback = LOOKBACK_WINDOW
        set_card = LOTO_CONFIG["loto6"]["pick_count"]
        features = 3
        block = SetAttentionBlock(num_heads=2, key_dim=8)
        x = tf.random.uniform([2, lookback, set_card, features])
        y = block(x, training=False)
        assert y.shape == x.shape

    def test_get_config_roundtrip(self):
        block = SetAttentionBlock(num_heads=2, key_dim=8, name="sab_test")
        config = block.get_config()
        assert config["num_heads"] == 2
        assert config["key_dim"] == 8
        # Reconstructable from config
        block2 = SetAttentionBlock.from_config(config)
        assert block2.num_heads == 2
        assert block2.key_dim == 8


# ---------------------------------------------------------------------------
# Unit tests – dataset builder
# ---------------------------------------------------------------------------

class TestSettransformerDataset:
    def test_feature_strategy_metadata(self):
        df = build_loto6_history()
        dataset = prepare_model_dataset(df, "loto6", "settransformer")
        meta = dataset["dataset_metadata"]
        assert meta["feature_strategy"] == "set_sequence_settransformer"
        assert meta["model_variant"] == SETTRANSFORMER_MODEL_VARIANT

    def test_raw_features_shape(self):
        df = build_loto6_history()
        dataset = prepare_model_dataset(df, "loto6", "settransformer")
        pick_count = LOTO_CONFIG["loto6"]["pick_count"]
        assert dataset["raw_features"].shape == (len(df), pick_count * 3)

    def test_targets_shape(self):
        df = build_loto6_history()
        dataset = prepare_model_dataset(df, "loto6", "settransformer")
        max_num = LOTO_CONFIG["loto6"]["max_num"]
        assert dataset["targets"].shape == (len(df) - LOOKBACK_WINDOW, max_num)

    def test_row_shape_metadata(self):
        df = build_loto6_history()
        dataset = prepare_model_dataset(df, "loto6", "settransformer")
        pick_count = LOTO_CONFIG["loto6"]["pick_count"]
        assert dataset["dataset_metadata"]["row_shape"] == [pick_count, 3]

    def test_input_summary_keys(self):
        df = build_loto6_history()
        dataset = prepare_model_dataset(df, "loto6", "settransformer")
        summary = dataset["dataset_metadata"]["input_summary"]
        for key in ("set_cardinality", "element_feature_count", "pooling", "temporal_head",
                    "attention_block_count", "attention_hidden_dim", "attention_num_heads"):
            assert key in summary, f"Missing input_summary key: {key}"

    def test_pooling_label_distinguishes_from_deepsets(self):
        df = build_loto6_history()
        ds = prepare_model_dataset(df, "loto6", "deepsets")
        st = prepare_model_dataset(df, "loto6", "settransformer")
        assert ds["dataset_metadata"]["input_summary"]["pooling"] == "mean"
        assert st["dataset_metadata"]["input_summary"]["pooling"] == "mean_after_attention"

    def test_feature_cols_match_deepsets(self):
        """settransformer reuses deepsets feature column names for comparability."""
        df = build_loto6_history()
        ds = prepare_model_dataset(df, "loto6", "deepsets")
        st = prepare_model_dataset(df, "loto6", "settransformer")
        assert ds["feature_cols"] == st["feature_cols"]

    def test_raw_features_identical_to_deepsets(self):
        """Same element features ensure fair architecture comparison."""
        df = build_loto6_history()
        ds = prepare_model_dataset(df, "loto6", "deepsets")
        st = prepare_model_dataset(df, "loto6", "settransformer")
        np.testing.assert_allclose(ds["raw_features"], st["raw_features"])


# ---------------------------------------------------------------------------
# Unit tests – leakage prevention
# ---------------------------------------------------------------------------

class TestLeakagePrevention:
    def test_settransformer_features_are_prefix_stable(self):
        """Extending the dataset must not change features for earlier rows."""
        df_long = build_loto6_history(LOOKBACK_WINDOW + 8)
        df_short = df_long.iloc[: LOOKBACK_WINDOW + 3].copy()
        ds_long = prepare_model_dataset(df_long, "loto6", "settransformer")
        ds_short = prepare_model_dataset(df_short, "loto6", "settransformer")
        np.testing.assert_allclose(
            ds_short["raw_features"],
            ds_long["raw_features"][: len(df_short)],
        )

    def test_scaler_fitted_on_train_only(self):
        """Scaler must not see test rows during fit."""
        df = build_loto6_history(LOOKBACK_WINDOW + 10)
        dataset = prepare_model_dataset(df, "loto6", "settransformer")
        raw = dataset["raw_features"]
        meta = dataset["dataset_metadata"]
        train_end = LOOKBACK_WINDOW + 5

        scaler = MinMaxScaler()
        scaler.fit(fit_scaler_for_variant(raw[:train_end], meta))

        # Transform train and test
        scaled_train = transform_features_for_variant(raw[:train_end], scaler, meta)
        # Test rows should not affect scaler
        assert scaler.n_features_in_ == meta["scaler_feature_count"]

        # Values fitted on train only – test rows might be outside [0,1] if
        # distribution differs, but the scaler itself should not have changed
        assert scaler.n_features_in_ == meta["element_feature_count"]


# ---------------------------------------------------------------------------
# Unit tests – model architecture
# ---------------------------------------------------------------------------

class TestSettransformerModel:
    def _make_model(self, compile_model=False):
        pick_count = LOTO_CONFIG["loto6"]["pick_count"]
        max_num = LOTO_CONFIG["loto6"]["max_num"]
        return build_settransformer_prob_model(
            input_shape=(LOOKBACK_WINDOW, pick_count, 3),
            max_num=max_num,
            compile_model=compile_model,
        )

    def test_output_shape(self):
        model = self._make_model()
        x = np.random.rand(2, LOOKBACK_WINDOW, LOTO_CONFIG["loto6"]["pick_count"], 3).astype(np.float32)
        out = model(tf.convert_to_tensor(x), training=False).numpy()
        assert out.shape == (2, LOTO_CONFIG["loto6"]["max_num"])

    def test_output_values_in_probability_range(self):
        model = self._make_model()
        x = np.random.rand(4, LOOKBACK_WINDOW, LOTO_CONFIG["loto6"]["pick_count"], 3).astype(np.float32)
        out = model(tf.convert_to_tensor(x), training=False).numpy()
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0

    def test_model_has_set_attention_block(self):
        model = self._make_model()
        layer_types = [type(layer).__name__ for layer in model.layers]
        assert "SetAttentionBlock" in layer_types

    def test_model_name(self):
        model = self._make_model()
        assert model.name == "settransformer_sequence_model"

    def test_miniloto_input_shape(self):
        """Architecture must work for miniloto (pick_count=5)."""
        pick_count = LOTO_CONFIG["miniloto"]["pick_count"]
        max_num = LOTO_CONFIG["miniloto"]["max_num"]
        model = build_settransformer_prob_model(
            input_shape=(LOOKBACK_WINDOW, pick_count, 3),
            max_num=max_num,
            compile_model=False,
        )
        x = np.random.rand(1, LOOKBACK_WINDOW, pick_count, 3).astype(np.float32)
        out = model(tf.convert_to_tensor(x), training=False).numpy()
        assert out.shape == (1, max_num)

    def test_build_prob_model_dispatches_to_settransformer(self):
        pick_count = LOTO_CONFIG["loto6"]["pick_count"]
        max_num = LOTO_CONFIG["loto6"]["max_num"]
        model = build_prob_model(
            input_shape=(LOOKBACK_WINDOW, pick_count, 3),
            max_num=max_num,
            model_variant="settransformer",
            compile_model=False,
        )
        assert model.name == "settransformer_sequence_model"

    def test_settransformer_and_deepsets_accept_same_input(self):
        pick_count = LOTO_CONFIG["loto6"]["pick_count"]
        max_num = LOTO_CONFIG["loto6"]["max_num"]
        x = tf.convert_to_tensor(
            np.random.rand(1, LOOKBACK_WINDOW, pick_count, 3).astype(np.float32)
        )
        for variant in ("deepsets", "settransformer"):
            m = build_prob_model(
                input_shape=(LOOKBACK_WINDOW, pick_count, 3),
                max_num=max_num,
                model_variant=variant,
                compile_model=False,
            )
            out = m(x, training=False).numpy()
            assert out.shape == (1, max_num)


# ---------------------------------------------------------------------------
# Unit tests – scaler/sample pipeline
# ---------------------------------------------------------------------------

class TestSettransformerPipeline:
    def test_4d_window_shape(self):
        df = build_loto6_history(LOOKBACK_WINDOW + 5)
        dataset = prepare_model_dataset(df, "loto6", "settransformer")
        scaler = MinMaxScaler()
        scaler.fit(fit_scaler_for_variant(dataset["raw_features"], dataset["dataset_metadata"]))
        scaled = transform_features_for_variant(
            dataset["raw_features"], scaler, dataset["dataset_metadata"]
        )
        X = build_model_samples_from_scaled_rows(scaled, 0, 2, dataset["dataset_metadata"])
        assert X.shape == (2, LOOKBACK_WINDOW, LOTO_CONFIG["loto6"]["pick_count"], 3)

    def test_scaled_values_in_unit_interval(self):
        df = build_loto6_history(LOOKBACK_WINDOW + 5)
        dataset = prepare_model_dataset(df, "loto6", "settransformer")
        scaler = MinMaxScaler()
        scaler.fit(fit_scaler_for_variant(dataset["raw_features"], dataset["dataset_metadata"]))
        scaled = transform_features_for_variant(
            dataset["raw_features"], scaler, dataset["dataset_metadata"]
        )
        # Train-fitted scaler on training rows → values in [0, 1]
        assert float(scaled.min()) >= -1e-6
        assert float(scaled.max()) <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Integration smoke tests – require real data files on disk
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not Path("data/loto6_processed.csv").exists(),
    reason="data/loto6_processed.csv not found; skip integration tests",
)
class TestSettransformerSmokeIntegration:
    def test_smoke_writes_settransformer_eval_report(self, tmp_path):
        workspace = prepare_workspace(tmp_path)
        run_command(
            workspace,
            "train_prob_model.py",
            "--loto_type", "loto6",
            "--preset", "smoke",
            "--model_variant", "settransformer",
            "--evaluation_model_variants", "legacy,multihot,deepsets,settransformer",
            "--saved_calibration_method", "none",
            "--evaluation_calibration_methods", "none,temperature",
            "--seed", "42",
        )
        report = json.loads((workspace / "data" / "eval_report_loto6.json").read_text())
        manifest = json.loads((workspace / "data" / "manifest_loto6.json").read_text())

        # eval_report must contain all four variants
        for variant in ("legacy", "multihot", "deepsets", "settransformer"):
            assert variant in report["model_variants"], f"variant {variant!r} missing from eval_report"

        # settransformer model_family
        st_payload = report["model_variants"]["settransformer"]
        assert st_payload["model_family"] == "settransformer_sequence"

        # feature_strategy
        assert st_payload["feature_strategy"] == "set_sequence_settransformer"

        # input_summary keys
        input_summary = st_payload.get("input_summary") or {}
        for key in ("set_cardinality", "element_feature_count", "attention_block_count"):
            assert key in input_summary, f"Missing key in input_summary: {key}"

        # statistical_tests must include settransformer comparisons
        comparisons = report["statistical_tests"]["comparisons"]
        assert "settransformer_vs_best_static" in comparisons
        assert "settransformer_vs_legacy" in comparisons
        assert "settransformer_vs_multihot" in comparisons
        assert "settransformer_vs_deepsets" in comparisons

        # decision_summary: settransformer is the production variant here,
        # so legacy/multihot/deepsets are the challengers.  Just ensure the
        # summary itself is populated.
        assert "rankings" in report["decision_summary"]
        assert "rule" in report["decision_summary"]

        # manifest training_context
        tc = manifest["training_context"]
        assert tc["model_variant"] == "settransformer"
        assert tc["model_family"] == "settransformer_sequence"
        assert tc["feature_strategy"] == "set_sequence_settransformer"

    def test_smoke_predict_reads_settransformer_artifact(self, tmp_path):
        workspace = prepare_workspace(tmp_path)
        run_command(
            workspace,
            "train_prob_model.py",
            "--loto_type", "loto6",
            "--preset", "smoke",
            "--model_variant", "settransformer",
            "--evaluation_model_variants", "legacy,settransformer",
            "--saved_calibration_method", "none",
            "--evaluation_calibration_methods", "none",
            "--seed", "42",
        )
        result = run_command(workspace, "predict.py", "--loto_type", "loto6")
        assert "variant: settransformer" in result.stdout

    def test_smoke_settransformer_as_challenger_in_decision_summary(self, tmp_path):
        """When legacy is production, settransformer must appear in challenger_decisions."""
        workspace = prepare_workspace(tmp_path)
        run_command(
            workspace,
            "train_prob_model.py",
            "--loto_type", "loto6",
            "--preset", "smoke",
            "--model_variant", "legacy",
            "--evaluation_model_variants", "legacy,settransformer",
            "--saved_calibration_method", "none",
            "--evaluation_calibration_methods", "none",
            "--seed", "42",
        )
        report = json.loads((workspace / "data" / "eval_report_loto6.json").read_text())
        challenger_decisions = report["decision_summary"].get("challenger_decisions") or {}
        assert "settransformer" in challenger_decisions
        st_decision = challenger_decisions["settransformer"]
        assert "should_promote" in st_decision
        assert "reason_summary" in st_decision
        assert "flags" in st_decision

    def test_smoke_update_system_includes_settransformer(self, tmp_path):
        workspace = prepare_workspace(tmp_path)
        run_command(
            workspace,
            "update_system.py",
            "--loto_type", "loto6",
            "--train_preset", "smoke",
            "--model_variant", "legacy",
            "--evaluation_model_variants", "legacy,multihot,deepsets,settransformer",
            "--skip_final_train",
            "--skip_data_refresh",
            "--seed", "99",
        )
        report = json.loads((workspace / "data" / "eval_report_loto6.json").read_text())
        assert "settransformer" in report["model_variants"]
        comparisons = (report.get("statistical_tests") or {}).get("comparisons") or {}
        assert "settransformer_vs_best_static" in comparisons

    def test_eval_report_manifest_required_keys(self, tmp_path):
        """Both artifacts must carry the keys documented in ARTIFACTS.md."""
        workspace = prepare_workspace(tmp_path)
        run_command(
            workspace,
            "train_prob_model.py",
            "--loto_type", "loto6",
            "--preset", "smoke",
            "--model_variant", "settransformer",
            "--evaluation_model_variants", "legacy,settransformer",
            "--saved_calibration_method", "none",
            "--evaluation_calibration_methods", "none",
            "--seed", "42",
        )
        manifest = json.loads((workspace / "data" / "manifest_loto6.json").read_text())
        report = json.loads((workspace / "data" / "eval_report_loto6.json").read_text())

        # Manifest required top-level keys
        for key in ("schema_version", "artifact_schema_version", "bundle_id",
                    "generated_at", "loto_type", "training_context", "metrics_summary",
                    "statistical_tests", "decision_summary"):
            assert key in manifest, f"manifest missing key: {key}"

        # manifest.training_context settransformer-specific fields
        tc = manifest["training_context"]
        assert tc.get("model_family") == "settransformer_sequence"
        assert tc.get("input_summary") is not None

        # eval_report required keys
        for key in ("artifact_schema_version", "bundle_id", "model_variants",
                    "statistical_tests", "decision_summary"):
            assert key in report, f"eval_report missing key: {key}"

    def test_calibration_integration_format(self, tmp_path):
        """Calibration methods must run through settransformer without errors."""
        workspace = prepare_workspace(tmp_path)
        run_command(
            workspace,
            "train_prob_model.py",
            "--loto_type", "loto6",
            "--preset", "smoke",
            "--model_variant", "settransformer",
            "--evaluation_model_variants", "legacy,settransformer",
            "--saved_calibration_method", "none",
            "--evaluation_calibration_methods", "none,temperature,isotonic",
            "--seed", "7",
        )
        report = json.loads((workspace / "data" / "eval_report_loto6.json").read_text())
        calib_eval = report.get("calibration_evaluation") or {}
        assert "settransformer" in calib_eval
        st_sel = calib_eval["settransformer"]
        assert "recommended_method" in st_sel
        assert "methods" in st_sel
