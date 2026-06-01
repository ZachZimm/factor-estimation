from __future__ import annotations

import numpy as np
import pandas as pd

from ff5_predictor.feature_extraction import fit_feature_extractor, transform_feature_frame


TARGETS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]


def _train_df(n: int = 12) -> pd.DataFrame:
    x = np.arange(n, dtype=float)
    df = pd.DataFrame(
        {
            "SPY_ret_1d": x,
            "QQQ_ret_1d": x * 1.1,
            "SPY_vol_21d": x[::-1],
            "proxy_size_ijr_spy": x * 0.5,
            "proxy_value_iwn_iwo": x * -0.25,
        },
        index=pd.date_range("2024-01-02", periods=n),
    )
    for i, target in enumerate(TARGETS):
        df[target] = 0.001 * (x + i)
    return df


def _config(method: str) -> dict:
    return {
        "feature_extraction": {
            "enabled": method != "none",
            "method": method,
            "apply_to_models": ["ridge"],
            "group_pca": {
                "scale_before_pca": True,
                "groups": {
                    "market_returns": {"patterns": ["*_ret_1d"], "n_components": 2},
                    "rolling_volatility": {"patterns": ["*_vol_*d"], "n_components": 1},
                    "proxy_size": {"patterns": ["proxy_size_*"], "n_components": 1},
                    "proxy_value": {"patterns": ["proxy_value_*"], "n_components": 1},
                    "other": {"patterns": ["*"], "n_components": 1},
                },
            },
            "pls": {"n_components": 20, "scale_features": True, "scale_targets": False},
            "clustered": {
                "correlation_threshold": 0.9,
                "max_features_for_clustering": 20,
                "min_cluster_size": 2,
                "singleton_policy": "keep",
                "scale_before_clustering": True,
            },
        }
    }


def test_group_pca_pls_and_clustered_transform_shapes() -> None:
    df = _train_df()
    feature_columns = ["SPY_ret_1d", "QQQ_ret_1d", "SPY_vol_21d", "proxy_size_ijr_spy", "proxy_value_iwn_iwo"]
    for method in ["group_pca", "pls", "clustered"]:
        extractor = fit_feature_extractor(df, feature_columns, TARGETS, _config(method), model_type="ridge")
        assert extractor is not None
        transformed, columns = transform_feature_frame(extractor, df[feature_columns].tail(3))
        assert list(transformed.index) == list(df.tail(3).index)
        assert columns == list(transformed.columns)
        assert len(columns) > 0
        assert not transformed.isna().any(axis=None)


def test_disabled_feature_extraction_returns_none() -> None:
    df = _train_df()
    extractor = fit_feature_extractor(df, ["SPY_ret_1d"], TARGETS, _config("none"), model_type="ridge")
    assert extractor is None


def test_clustered_groups_correlated_features() -> None:
    df = _train_df()
    extractor = fit_feature_extractor(df, ["SPY_ret_1d", "QQQ_ret_1d"], TARGETS, _config("clustered"), model_type="ridge")
    assert extractor is not None
    assert extractor.metadata["n_clusters"] == 1


def test_keep_original_features_adds_raw_columns_to_extracted_columns() -> None:
    df = _train_df()
    config = _config("group_pca")
    config["feature_extraction"]["keep_original_features"] = True
    feature_columns = ["SPY_ret_1d", "QQQ_ret_1d"]
    extractor = fit_feature_extractor(df, feature_columns, TARGETS, config, model_type="ridge")
    assert extractor is not None
    transformed, columns = transform_feature_frame(extractor, df[feature_columns].tail(2))

    assert "SPY_ret_1d" in columns
    assert "QQQ_ret_1d" in columns
    assert any(column.startswith("fx_group_pca__") for column in columns)
    assert transformed.shape[1] > len(feature_columns)
