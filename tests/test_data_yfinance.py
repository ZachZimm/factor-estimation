from __future__ import annotations

import pandas as pd

import ff5_predictor.data_yfinance as data_yfinance
from ff5_predictor.data_yfinance import load_market_data
from ff5_predictor.io import write_parquet_with_metadata


def _config(tmp_path, date_filter: dict | None = None) -> dict:
    config = {
        "data": {
            "tickers": ["SPY"],
            "start_date": "2024-01-01",
            "end_date": None,
            "cache_dir": str(tmp_path),
            "force_refresh": False,
        }
    }
    if date_filter:
        config["date_filter"] = date_filter
    return config


def _market_frame(end_date: str) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", end_date)
    close = [100 + i for i in range(len(dates))]
    return pd.DataFrame(
        {
            "SPY_open": close,
            "SPY_high": close,
            "SPY_low": close,
            "SPY_close": close,
            "SPY_volume": [1] * len(dates),
        },
        index=dates,
    )


def _cache_path(tmp_path) -> object:
    config = _config(tmp_path)
    tickers = [ticker.upper() for ticker in config["data"]["tickers"]]
    ticker_key = data_yfinance.cache_key_for_tickers(tickers)
    return tmp_path / f"yfinance_{ticker_key}_{config['data']['start_date']}_latest.parquet"


def test_load_market_data_refreshes_cache_when_requested_date_is_missing(tmp_path, monkeypatch) -> None:
    cache_path = _cache_path(tmp_path)
    write_parquet_with_metadata(_market_frame("2024-01-03"), cache_path, {"source": "test"})
    calls = []

    def fake_fetch(tickers, start, end, auto_adjust=True):
        calls.append((tickers, start, end, auto_adjust))
        return _market_frame("2024-01-05")

    monkeypatch.setattr(data_yfinance, "fetch_yfinance_data", fake_fetch)

    df = load_market_data(_config(tmp_path, {"start_date": "2024-01-05", "end_date": "2024-01-05"}))

    assert calls
    assert pd.Timestamp("2024-01-05") in df.index


def test_load_market_data_uses_cache_when_requested_date_is_present(tmp_path, monkeypatch) -> None:
    cache_path = _cache_path(tmp_path)
    write_parquet_with_metadata(_market_frame("2024-01-05"), cache_path, {"source": "test"})
    calls = []

    def fake_fetch(tickers, start, end, auto_adjust=True):
        calls.append((tickers, start, end, auto_adjust))
        return _market_frame("2024-01-06")

    monkeypatch.setattr(data_yfinance, "fetch_yfinance_data", fake_fetch)

    df = load_market_data(_config(tmp_path, {"start_date": "2024-01-05", "end_date": "2024-01-05"}))

    assert calls == []
    assert df.index.max() == pd.Timestamp("2024-01-05")
