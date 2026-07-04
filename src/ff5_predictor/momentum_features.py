from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ff5_predictor.features import infer_tickers


def build_momentum_features(market_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    cfg = config.get("momentum_features", {})
    if not bool(cfg.get("enabled", False)):
        return pd.DataFrame(index=market_df.index)

    tickers = [ticker for ticker in infer_tickers(market_df) if f"{ticker}_close" in market_df.columns]
    if not tickers:
        return pd.DataFrame(index=market_df.index)

    close = pd.DataFrame({ticker: market_df[f"{ticker}_close"] for ticker in tickers}, index=market_df.index)
    same_day_returns = close.pct_change()
    signals = _signal_specs(cfg)
    series: dict[str, pd.Series] = {}

    for lookback_rows, skip_rows in signals:
        signal = close.shift(skip_rows) / close.shift(lookback_rows) - 1.0
        if bool(cfg.get("include_individual_signals", True)):
            for ticker in tickers:
                series[f"mom_signal_{ticker}_{lookback_rows}_{skip_rows}d"] = signal[ticker]
        if bool(cfg.get("include_cross_sectional_proxy", True)):
            name = f"proxy_momentum_xsec_{lookback_rows}_{skip_rows}d"
            series[name] = _winner_minus_loser_proxy(
                signal,
                same_day_returns,
                top_quantile=float(cfg.get("top_quantile", 0.25)),
                bottom_quantile=float(cfg.get("bottom_quantile", 0.25)),
                min_assets=int(cfg.get("min_assets", 8)),
            )

    if "PDP" in close.columns and "SPY" in close.columns:
        pdp_ret = same_day_returns["PDP"]
        spy_ret = same_day_returns["SPY"]
        series["proxy_momentum_pdp"] = pdp_ret
        series["proxy_momentum_pdp_spy"] = pdp_ret - spy_ret

    if not series:
        return pd.DataFrame(index=market_df.index)

    features = pd.DataFrame(series, index=market_df.index).replace([np.inf, -np.inf], np.nan)
    rolling: dict[str, pd.Series] = {}
    windows = [int(window) for window in cfg.get("rolling_windows", [5, 21, 63])]
    for name, values in features.items():
        if name.startswith("mom_signal_"):
            continue
        for window in windows:
            rolling[f"{name}_sum_{window}d"] = values.rolling(window).sum()
        if 21 in windows:
            rolling[f"{name}_vol_21d"] = values.rolling(21).std(ddof=0)
    if rolling:
        features = pd.concat([features, pd.DataFrame(rolling, index=market_df.index)], axis=1)
    return features.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")


def _signal_specs(cfg: dict[str, Any]) -> list[tuple[int, int]]:
    specs = cfg.get("signals")
    if specs:
        result = []
        for item in specs:
            result.append((int(item["lookback_rows"]), int(item.get("skip_rows", 21))))
        return result
    return [(252, 21), (189, 21), (126, 21)]


def _winner_minus_loser_proxy(
    signal: pd.DataFrame,
    same_day_returns: pd.DataFrame,
    *,
    top_quantile: float,
    bottom_quantile: float,
    min_assets: int,
) -> pd.Series:
    values = []
    index = signal.index
    top_q = min(max(top_quantile, 0.0), 1.0)
    bottom_q = min(max(bottom_quantile, 0.0), 1.0)
    for date in index:
        signal_row = signal.loc[date]
        return_row = same_day_returns.loc[date]
        valid = signal_row.notna() & return_row.notna()
        n_valid = int(valid.sum())
        if n_valid < min_assets:
            values.append(np.nan)
            continue
        ranked = signal_row[valid].sort_values()
        n_bottom = max(1, int(np.floor(n_valid * bottom_q)))
        n_top = max(1, int(np.floor(n_valid * top_q)))
        bottom_tickers = ranked.index[:n_bottom]
        top_tickers = ranked.index[-n_top:]
        value = float(return_row[top_tickers].mean() - return_row[bottom_tickers].mean())
        values.append(value)
    return pd.Series(values, index=index)
