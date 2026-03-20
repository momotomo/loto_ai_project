from report_utils import (
    build_calibration_summary_rows,
    build_statistical_test_rows,
    build_variant_summary_rows,
    get_saved_calibration_method,
    get_saved_model_variant,
)


def test_report_utils_are_safe_for_legacy_artifacts():
    assert get_saved_model_variant(manifest={}) == "legacy"
    assert get_saved_calibration_method(manifest={}) == "none"
    assert build_variant_summary_rows({"Model (LSTM)": {"logloss": 1.0}}) == []
    assert build_calibration_summary_rows({"Model (LSTM)": {"logloss": 1.0}}) == []
    assert build_statistical_test_rows({"Baselines": {}}) == []


def test_report_utils_extract_rows_from_new_artifacts():
    report = {
        "model_variants": {
            "legacy": {
                "dataset_variant": "legacy",
                "model_family": "temporal_lstm",
                "feature_strategy": "tabular",
                "calibration_selection": {
                    "recommended_method": "temperature",
                    "recommended_metrics": {"logloss": 0.79, "brier": 0.19, "ece": 0.02, "sample_count": 24},
                    "raw_metrics": {"logloss": 0.8, "brier": 0.2, "ece": 0.04, "sample_count": 24},
                    "methods": [
                        {
                            "method": "none",
                            "metrics": {"logloss": 0.8, "brier": 0.2, "ece": 0.04, "sample_count": 24},
                            "delta_vs_raw": {"logloss": 0.0, "brier": 0.0, "ece": 0.0},
                            "selected": False,
                            "eligible": True,
                            "status": "raw_model",
                        },
                        {
                            "method": "temperature",
                            "metrics": {"logloss": 0.79, "brier": 0.19, "ece": 0.02, "sample_count": 24},
                            "delta_vs_raw": {"logloss": -0.01, "brier": -0.01, "ece": -0.02},
                            "selected": True,
                            "eligible": True,
                            "status": "fitted",
                        },
                    ],
                },
                "walk_forward": {
                    "aggregate": {
                        "model": {
                            "metric_summary": {
                                "logloss": {"mean": 0.8},
                                "brier": {"mean": 0.2},
                                "ece": {"mean": 0.04},
                                "mean_overlap_top_k": {"mean": 1.1},
                            }
                        }
                    }
                },
            },
            "deepsets": {
                "dataset_variant": "deepsets",
                "model_family": "deepsets_sequence",
                "feature_strategy": "set_sequence_deepsets",
                "feature_channels": ["number_norm", "frequency", "gap"],
                "input_summary": {"set_cardinality": 6, "element_feature_count": 3},
                "set_pooling": "mean",
                "lookback_integration": "sequence_lstm",
                "calibration_selection": {
                    "recommended_method": "none",
                    "recommended_metrics": {"logloss": 0.78, "brier": 0.18, "ece": 0.03, "sample_count": 24},
                    "raw_metrics": {"logloss": 0.78, "brier": 0.18, "ece": 0.03, "sample_count": 24},
                    "methods": [],
                },
                "walk_forward": {
                    "aggregate": {
                        "model": {
                            "metric_summary": {
                                "logloss": {"mean": 0.78},
                                "brier": {"mean": 0.18},
                                "ece": {"mean": 0.03},
                                "mean_overlap_top_k": {"mean": 1.2},
                            }
                        }
                    }
                },
            }
        },
        "statistical_tests": {
            "comparisons": {
                "multihot_vs_legacy": {
                    "candidate_name": "multihot",
                    "reference_name": "legacy",
                    "candidate_calibration_method": "isotonic",
                    "reference_calibration_method": "temperature",
                    "mean_delta": -0.01,
                    "bootstrap_ci": {"lower": -0.02, "upper": -0.001},
                    "permutation_test": {"p_value": 0.03},
                    "sample_count": 24,
                }
            }
        },
    }

    variant_rows = build_variant_summary_rows(report)
    calibration_rows = build_calibration_summary_rows(report)
    statistical_rows = build_statistical_test_rows(report)

    variant_rows_by_name = {row["variant"]: row for row in variant_rows}
    assert variant_rows_by_name["legacy"]["feature_strategy"] == "tabular"
    assert variant_rows_by_name["legacy"]["selected_calibration_method"] == "temperature"
    assert variant_rows_by_name["legacy"]["ece"] == 0.02
    assert variant_rows_by_name["deepsets"]["model_family"] == "deepsets_sequence"
    assert variant_rows_by_name["deepsets"]["set_cardinality"] == 6
    assert calibration_rows[1]["method"] == "temperature"
    assert calibration_rows[1]["selected"] is True
    assert statistical_rows[0]["comparison"] == "multihot_vs_legacy"
    assert statistical_rows[0]["p_value"] == 0.03
    assert statistical_rows[0]["candidate_calibration_method"] == "isotonic"
