from __future__ import annotations

import pandas as pd

from ff5_predictor.config import load_config
from ff5_predictor.nowcast_dataset import build_nowcast_dataset


TARGETS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]


def test_production_config_is_market_only_all_candidates() -> None:
    cfg = load_config("config/nowcast/production.yaml")

    for ticker in ["IWN", "IWO", "IWD", "IWF", "IJR", "IJH", "RSP", "SPHQ", "VNQ", "EFA", "EEM"]:
        assert ticker in cfg["data"]["tickers"]
    assert cfg["target_features"]["include_lagged_targets"] is False
    assert cfg["target_features"]["include_rf_lags"] is False
    assert cfg["availability"]["recursive_factor_lags"] is False
    assert cfg["nowcast"]["save_feature_attributions"] is True


def test_production_dataset_has_no_ff5_input_features() -> None:
    cfg = load_config("config/nowcast/production.yaml")
    cfg["data"]["tickers"] = ["SPY", "IWN", "IWO", "IJR", "IJH"]
    cfg["features"]["lookback_windows"] = [1]
    cfg["features"]["include_log_returns"] = False
    cfg["features"]["include_rolling_volatility"] = False
    cfg["features"]["include_ohlc_features"] = False
    cfg["features"]["include_drawdown"] = False
    cfg["proxy_features"]["rolling_windows"] = [5]
    dates = pd.date_range("2024-01-02", periods=8)
    ff5 = pd.DataFrame({target: [0.001 * i for i in range(8)] for target in TARGETS}, index=dates)
    ff5["RF"] = 0.0
    market_data = {}
    for ticker, offset in {"SPY": 0, "IWN": 1, "IWO": 2, "IJR": 3, "IJH": 4}.items():
        close = [100 + offset + i for i in range(9)]
        market_data.update(
            {
                f"{ticker}_open": close,
                f"{ticker}_high": close,
                f"{ticker}_low": close,
                f"{ticker}_close": close,
                f"{ticker}_volume": [1] * 9,
            }
        )
    market = pd.DataFrame(market_data, index=pd.date_range("2024-01-02", periods=9))

    dataset = build_nowcast_dataset(ff5, market, cfg)
    forbidden = [col for col in dataset.feature_columns if col.startswith(("Mkt-RF_", "SMB_", "HML_", "RMW_", "CMA_", "RF_"))]

    assert not forbidden
    assert "proxy_size_ijr_spy" in dataset.feature_columns
    assert "proxy_value_iwn_iwo" in dataset.feature_columns
