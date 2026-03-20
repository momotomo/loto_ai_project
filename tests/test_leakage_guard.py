import numpy as np

from config import LOOKBACK_WINDOW
from train_prob_model import fit_scaler_for_split


def test_scaler_fit_stays_within_train_visible_rows():
    observed_rows = LOOKBACK_WINDOW + 2
    raw_features = np.arange((LOOKBACK_WINDOW + 6) * 2, dtype=np.float32).reshape(LOOKBACK_WINDOW + 6, 2)
    raw_features[observed_rows:] = 9999.0

    scaler = fit_scaler_for_split(raw_features, train_sample_count=2)

    np.testing.assert_allclose(scaler.data_max_, raw_features[:observed_rows].max(axis=0))
    np.testing.assert_allclose(scaler.data_min_, raw_features[:observed_rows].min(axis=0))
    assert float(scaler.data_max_[0]) < 9999.0
