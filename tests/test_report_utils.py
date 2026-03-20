from report_utils import build_statistical_test_rows, build_variant_summary_rows, get_saved_model_variant


def test_report_utils_are_safe_for_legacy_artifacts():
    assert get_saved_model_variant(manifest={}) == "legacy"
    assert build_variant_summary_rows({"Model (LSTM)": {"logloss": 1.0}}) == []
    assert build_statistical_test_rows({"Baselines": {}}) == []


def test_report_utils_extract_rows_from_new_artifacts():
    report = {
        "model_variants": {
            "legacy": {
                "dataset_variant": "legacy",
                "feature_strategy": "tabular",
                "walk_forward": {
                    "aggregate": {
                        "model": {
                            "metric_summary": {
                                "logloss": {"mean": 0.8},
                                "brier": {"mean": 0.2},
                                "mean_overlap_top_k": {"mean": 1.1},
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
                    "mean_delta": -0.01,
                    "bootstrap_ci": {"lower": -0.02, "upper": -0.001},
                    "permutation_test": {"p_value": 0.03},
                    "sample_count": 24,
                }
            }
        },
    }

    variant_rows = build_variant_summary_rows(report)
    statistical_rows = build_statistical_test_rows(report)

    assert variant_rows[0]["variant"] == "legacy"
    assert variant_rows[0]["feature_strategy"] == "tabular"
    assert statistical_rows[0]["comparison"] == "multihot_vs_legacy"
    assert statistical_rows[0]["p_value"] == 0.03
