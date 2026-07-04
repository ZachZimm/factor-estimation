from __future__ import annotations

import numpy as np
import pandas as pd

from ff5_predictor.momentum_features import build_momentum_features


def test_momentum_features_build_prior_12_2_signal_and_cross_sectional_proxy() -> None:
    dates = pd.date_range("2023-01-02", periods=260)
    tickers = ["SPY", "PDP", "IWM", "QQQ"]
    market = pd.DataFrame(index=dates)
    for idx, ticker in enumerate(tickers):
        close = pd.Series(np.linspace(100 + idx, 150 + idx * 3, len(dates)), index=dates)
        market[f"{ticker}_close"] = close

    config = {
        "momentum_features": {
            "enabled": True,
            "signals": [{"lookback_rows": 252, "skip_rows": 21}],
            "include_individual_signals": True,
            "include_cross_sectional_proxy": True,
            "top_quantile": 0.25,
            "bottom_quantile": 0.25,
            "min_assets": 2,
            "rolling_windows": [5, 21],
        }
    }

    features = build_momentum_features(market, config)

    target_date = dates[252]
    expected = market.loc[dates[231], "SPY_close"] / market.loc[dates[0], "SPY_close"] - 1.0
    assert features.loc[target_date, "mom_signal_SPY_252_21d"] == expected
    assert "proxy_momentum_xsec_252_21d" in features.columns
    assert "proxy_momentum_pdp_spy" in features.columns
    assert not features["proxy_momentum_xsec_252_21d"].dropna().empty


def test_momentum_features_disabled_returns_empty_frame() -> None:
    dates = pd.date_range("2024-01-02", periods=5)
    market = pd.DataFrame({"SPY_close": [1, 2, 3, 4, 5]}, index=dates)

    features = build_momentum_features(market, {"momentum_features": {"enabled": False}})

    assert features.empty
    assert list(features.index) == list(dates)
