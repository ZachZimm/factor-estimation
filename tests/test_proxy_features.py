from __future__ import annotations

import numpy as np
import pandas as pd

from ff5_predictor.proxy_features import build_proxy_features


def test_proxy_spreads_are_computed_and_missing_tickers_are_skipped() -> None:
    dates = pd.date_range("2024-01-01", periods=6)
    market_features = pd.DataFrame(
        {
            "SPY_ret_1d": [0.00, 0.01, 0.02, -0.01, 0.03, 0.01],
            "IWM_ret_1d": [0.00, 0.03, 0.01, -0.02, 0.04, 0.02],
            "IVE_ret_1d": [0.00, 0.02, 0.01, 0.01, 0.01, 0.02],
            "IVW_ret_1d": [0.00, 0.01, 0.03, 0.00, 0.02, 0.01],
        },
        index=dates,
    )

    features = build_proxy_features(
        market_features,
        {"proxy_features": {"enabled": True, "rolling_windows": [5]}},
    )

    assert np.isclose(features.loc[dates[1], "proxy_size_iwm_spy"], 0.02)
    assert np.isclose(features.loc[dates[2], "proxy_value_ive_ivw"], -0.02)
    assert "proxy_size_iwm_vti" not in features.columns
    assert not any(features[col].isna().all() for col in features.columns)
