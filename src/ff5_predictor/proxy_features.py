from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def build_proxy_features(
    market_features: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    proxy_config = config.get("proxy_features", {})
    if not proxy_config.get("enabled", True):
        return pd.DataFrame(index=market_features.index)

    series: dict[str, pd.Series] = {}

    def col(name: str) -> pd.Series | None:
        return market_features[name] if name in market_features.columns else None

    def add(name: str, value: pd.Series | None) -> None:
        if value is not None:
            series[name] = value

    def spread(name: str, left: str, right: str) -> None:
        left_s = col(left)
        right_s = col(right)
        if left_s is not None and right_s is not None:
            add(name, left_s - right_s)

    add("proxy_market_spy", col("SPY_ret_1d"))
    add("proxy_market_vti", col("VTI_ret_1d"))

    if proxy_config.get("include_relative_returns", True):
        spread("proxy_size_iwm_spy", "IWM_ret_1d", "SPY_ret_1d")
        spread("proxy_size_iwm_vti", "IWM_ret_1d", "VTI_ret_1d")
        spread("proxy_size_ijs_spy", "IJS_ret_1d", "SPY_ret_1d")
        spread("proxy_sector_xlk_spy", "XLK_ret_1d", "SPY_ret_1d")
        spread("proxy_sector_xlf_spy", "XLF_ret_1d", "SPY_ret_1d")
        spread("proxy_sector_xle_spy", "XLE_ret_1d", "SPY_ret_1d")
        spread("proxy_sector_xlv_spy", "XLV_ret_1d", "SPY_ret_1d")
        spread("proxy_sector_xli_spy", "XLI_ret_1d", "SPY_ret_1d")
        spread("proxy_sector_xly_spy", "XLY_ret_1d", "SPY_ret_1d")
        spread("proxy_sector_xlp_spy", "XLP_ret_1d", "SPY_ret_1d")
        spread("proxy_sector_xlu_spy", "XLU_ret_1d", "SPY_ret_1d")

    if proxy_config.get("include_factor_mimic_spreads", True):
        spread("proxy_value_ive_ivw", "IVE_ret_1d", "IVW_ret_1d")
        spread("proxy_value_vbr_vbk", "VBR_ret_1d", "VBK_ret_1d")
        spread("proxy_value_ijs_ijt", "IJS_ret_1d", "IJT_ret_1d")

    if proxy_config.get("include_risk_proxies", True):
        add("proxy_vix_ret", col("^VIX_ret_1d"))
        add("proxy_vix_log_ret", col("^VIX_log_ret_1d"))
        add("proxy_tlt_ret", col("TLT_ret_1d"))
        spread("proxy_credit_hyg_lqd", "HYG_ret_1d", "LQD_ret_1d")
        spread("proxy_credit_hyg_shy", "HYG_ret_1d", "SHY_ret_1d")

    if not series:
        return pd.DataFrame(index=market_features.index)

    proxies = pd.DataFrame(series, index=market_features.index).replace([np.inf, -np.inf], np.nan)
    rolling: dict[str, pd.Series] = {}
    windows = [int(window) for window in proxy_config.get("rolling_windows", [5, 21, 63])]
    for name, values in proxies.items():
        for window in windows:
            rolling[f"{name}_sum_{window}d"] = values.rolling(window).sum()
        if 21 in windows:
            rolling[f"{name}_vol_21d"] = values.rolling(21).std(ddof=0)
        mean_252 = values.rolling(252).mean()
        std_252 = values.rolling(252).std(ddof=0)
        rolling[f"{name}_z_252d"] = (values - mean_252) / std_252.replace(0, np.nan)

    if rolling:
        proxies = pd.concat([proxies, pd.DataFrame(rolling, index=market_features.index)], axis=1)
    return proxies.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
