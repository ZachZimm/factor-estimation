from __future__ import annotations

import pandas as pd

import ff5_predictor.rolling_train as rolling_train
from ff5_predictor.rolling_train import rolling_predict


TARGETS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]


def _modeling_df(rows: int = 8) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    data = {"feature": range(rows)}
    for target in TARGETS:
        data[target] = [value / 100 for value in range(rows)]
    return pd.DataFrame(data, index=dates)


def _config() -> dict:
    return {
        "training": {
            "train_window_days": 3,
            "min_train_rows": 3,
            "step_size": 1,
            "model_type": "ridge",
            "scale_features": True,
            "save_models": False,
        },
        "output": {"models_dir": "data/models"},
    }


def test_training_window_excludes_target_date() -> None:
    predictions = rolling_predict(_modeling_df(), ["feature"], TARGETS, _config())

    for _, row in predictions.iterrows():
        assert pd.Timestamp(row["train_end_date"]) < pd.Timestamp(row["date"])


def test_scaler_fit_only_training_window(monkeypatch) -> None:
    fit_indices = []

    class SpyScaler:
        def fit_transform(self, X):
            fit_indices.append(list(X.index))
            return X.to_numpy()

        def transform(self, X):
            return X.to_numpy()

    monkeypatch.setattr(rolling_train, "make_scaler", lambda enabled: SpyScaler())

    predictions = rolling_predict(_modeling_df(), ["feature"], TARGETS, _config())

    first_prediction_date = pd.Timestamp(predictions.iloc[0]["date"])
    assert fit_indices[0] == list(pd.date_range("2024-01-01", periods=3, freq="D"))
    assert first_prediction_date not in fit_indices[0]
    assert all(date < first_prediction_date for date in fit_indices[0])
