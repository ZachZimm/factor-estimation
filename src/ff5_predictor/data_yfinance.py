from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import yfinance as yf

from ff5_predictor.io import (
    cache_key_for_tickers,
    ensure_dir,
    normalize_datetime_index,
    package_version,
    read_parquet_with_metadata,
    utc_now_iso,
    write_parquet_with_metadata,
)

LOGGER = logging.getLogger(__name__)


def load_market_data(config: dict[str, Any]) -> pd.DataFrame:
    tickers = [ticker.upper() for ticker in config["data"]["tickers"]]
    start = config["data"].get("start_date")
    end = config["data"].get("end_date")
    end_key = end or "latest"
    ticker_key = cache_key_for_tickers(tickers)
    cache_dir = ensure_dir(config["data"]["cache_dir"])
    cache_path = cache_dir / f"yfinance_{ticker_key}_{start}_{end_key}.parquet"

    if cache_path.exists() and not bool(config["data"].get("force_refresh", False)):
        df, _ = read_parquet_with_metadata(cache_path)
        cached = normalize_datetime_index(df)
        if _cache_satisfies_requested_dates(cached, config):
            return cached
        LOGGER.info(
            "Cached yfinance data ends at %s but requested prediction dates extend through %s; refreshing market data",
            cached.index.max().date() if not cached.empty else None,
            _requested_market_end(config).date() if _requested_market_end(config) is not None else None,
        )
        try:
            return _fetch_and_cache_market_data(tickers, start, end, cache_path)
        except Exception as exc:
            LOGGER.warning("Market data refresh failed; using cached yfinance data: %s", exc)
            return cached

    return _fetch_and_cache_market_data(tickers, start, end, cache_path)


def _fetch_and_cache_market_data(
    tickers: list[str],
    start: str,
    end: str | None,
    cache_path,
) -> pd.DataFrame:
    df = fetch_yfinance_data(tickers, start=start, end=end, auto_adjust=True)
    metadata = {
        "source": "yfinance",
        "dataset": "adjusted_ohlcv",
        "download_timestamp_utc": utc_now_iso(),
        "start_date": start,
        "end_date": end,
        "tickers": tickers,
        "library_versions": {
            "pandas": package_version("pandas"),
            "yfinance": package_version("yfinance"),
        },
        "units": "adjusted_prices",
    }
    write_parquet_with_metadata(df, cache_path, metadata)
    return df


def _cache_satisfies_requested_dates(df: pd.DataFrame, config: dict[str, Any]) -> bool:
    requested_end = _requested_market_end(config)
    if requested_end is None:
        return True
    if df.empty:
        return False
    return pd.Timestamp(df.index.max()) >= requested_end


def _requested_market_end(config: dict[str, Any]) -> pd.Timestamp | None:
    date_filter = config.get("date_filter", {})
    end = date_filter.get("end_date") or date_filter.get("start_date")
    return pd.Timestamp(end) if end else None


def fetch_yfinance_data(
    tickers: list[str],
    start: str,
    end: str | None,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    if raw.empty:
        raise ValueError("yfinance returned no market data")
    return flatten_yfinance_columns(raw, tickers)


def flatten_yfinance_columns(df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    columns: dict[str, pd.Series] = {}
    field_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    upper_tickers = [ticker.upper() for ticker in tickers]

    if isinstance(df.columns, pd.MultiIndex):
        first_level = {str(value).upper() for value in df.columns.get_level_values(0)}
        ticker_first = bool(first_level.intersection(upper_tickers))
        for ticker in upper_tickers:
            for yf_field, out_field in field_map.items():
                key = (ticker, yf_field) if ticker_first else (yf_field, ticker)
                if key in df.columns:
                    columns[f"{ticker}_{out_field}"] = df[key]
    else:
        if len(upper_tickers) != 1:
            raise ValueError("Flat yfinance columns are only valid for one ticker")
        ticker = upper_tickers[0]
        for yf_field, out_field in field_map.items():
            if yf_field in df.columns:
                columns[f"{ticker}_{out_field}"] = df[yf_field]

    result = pd.DataFrame(columns, index=df.index)
    expected = [f"{ticker}_{field}" for ticker in upper_tickers for field in field_map.values()]
    missing = [col for col in expected if col not in result.columns]
    if missing:
        raise ValueError(f"Missing normalized yfinance columns: {missing}")

    result = result[expected].apply(pd.to_numeric, errors="coerce")
    result = normalize_datetime_index(result)
    return result.dropna(how="all")
