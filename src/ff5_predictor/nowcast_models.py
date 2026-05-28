from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


@dataclass
class FittedNowcastModel:
    model_type: str
    model: Any
    scaler: StandardScaler | None
    feature_columns: list[str]
    target_columns: list[str]
    metadata: dict[str, Any]

    def predict_frame(self, features: pd.DataFrame) -> np.ndarray:
        X = features[self.feature_columns].astype(float).to_numpy()
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return np.asarray(self.model.predict(X))


@dataclass
class FittedTorchNowcastModel:
    model_type: str
    model: Any
    feature_scaler: Any
    target_scaler: Any
    feature_columns: list[str]
    target_columns: list[str]
    lookback_rows: int
    device: Any
    metadata: dict[str, Any]

    def predict_from_history(self, feature_history: pd.DataFrame) -> np.ndarray:
        import torch

        if len(feature_history) < self.lookback_rows:
            raise ValueError("Not enough feature history for TFT nowcast prediction")
        X = feature_history[self.feature_columns].tail(self.lookback_rows).to_numpy(dtype=np.float32)[None, :, :]
        X_scaled = self.feature_scaler.transform(X)
        self.model.eval()
        with torch.no_grad():
            pred_scaled = self.model(torch.from_numpy(X_scaled).to(self.device)).detach().cpu().numpy()
        return self.target_scaler.inverse_transform(pred_scaled)[0]


def fit_ridge_nowcast(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    target_columns: list[str],
    config: dict[str, Any],
) -> FittedNowcastModel:
    model_cfg = config.get("models", {}).get("ridge", {})
    nowcast_cfg = config.get("nowcast", {})
    train_window = int(nowcast_cfg.get("train_window_rows", 2520))
    if train_window > 0:
        train_df = train_df.tail(train_window)
    X = train_df[feature_columns].astype(float).to_numpy()
    y = train_df[target_columns].astype(float).to_numpy()
    scale = bool(model_cfg.get("scale_features", True))
    scaler = StandardScaler() if scale else None

    alpha = float(model_cfg.get("alpha", 10.0))
    selected_alpha = alpha
    validation_rows = int(model_cfg.get("validation_window_rows", 252))
    if bool(model_cfg.get("tune_alpha", True)) and len(train_df) > validation_rows + 10:
        fit_X = X[:-validation_rows]
        fit_y = y[:-validation_rows]
        val_X = X[-validation_rows:]
        val_y = y[-validation_rows:]
        best_score = float("inf")
        for candidate in [float(v) for v in model_cfg.get("alpha_grid", [alpha])]:
            candidate_scaler = StandardScaler() if scale else None
            candidate_fit_X = candidate_scaler.fit_transform(fit_X) if candidate_scaler else fit_X
            candidate_val_X = candidate_scaler.transform(val_X) if candidate_scaler else val_X
            candidate_model = Ridge(alpha=candidate)
            candidate_model.fit(candidate_fit_X, fit_y)
            pred = candidate_model.predict(candidate_val_X)
            score = float(np.sqrt(mean_squared_error(val_y, pred)))
            if score < best_score:
                best_score = score
                selected_alpha = candidate

    X_fit = scaler.fit_transform(X) if scaler else X
    model = Ridge(alpha=selected_alpha)
    model.fit(X_fit, y)
    return FittedNowcastModel(
        model_type="ridge",
        model=model,
        scaler=scaler,
        feature_columns=feature_columns,
        target_columns=target_columns,
        metadata={
            "alpha": selected_alpha,
            "n_train_rows": int(len(train_df)),
            "train_start_date": str(pd.Timestamp(train_df.index.min()).date()),
            "train_end_date": str(pd.Timestamp(train_df.index.max()).date()),
            "scale_features": scale,
        },
    )


def save_fitted_model(fitted: FittedNowcastModel, path) -> None:
    joblib.dump(fitted, path)


def fit_tft_nowcast(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    target_columns: list[str],
    config: dict[str, Any],
) -> FittedTorchNowcastModel:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from ff5_predictor.torch_models import make_torch_model
    from ff5_predictor.torch_models.common import (
        EarlyStopping,
        TargetStandardizer,
        WindowStandardizer,
        select_device,
        set_random_seed,
    )

    torch_cfg = config.get("torch", {})
    seq_cfg = config.get("sequence", {})
    model_cfg = config.get("models", {}).get("tft", {})
    lookback = int(seq_cfg.get("lookback_rows", 63))
    train_window = int(config.get("nowcast", {}).get("train_window_rows", 2520))
    train_df = train_df.tail(train_window)
    if len(train_df) < lookback + 2:
        raise ValueError("Not enough rows to train TFT nowcast model")

    set_random_seed(int(config.get("experiments", {}).get("random_seed", 42)))
    X_values = train_df[feature_columns].to_numpy(dtype=np.float32)
    y_values = train_df[target_columns].to_numpy(dtype=np.float32)
    X: list[np.ndarray] = []
    y: list[np.ndarray] = []
    for end in range(lookback - 1, len(train_df)):
        start = end - lookback + 1
        X.append(X_values[start : end + 1])
        y.append(y_values[end])
    X_array = np.stack(X).astype(np.float32)
    y_array = np.stack(y).astype(np.float32)

    validation_rows = min(int(torch_cfg.get("validation_window_rows", model_cfg.get("validation_window_rows", 252))), max(0, len(X_array) // 5))
    if validation_rows:
        train_positions = np.arange(0, len(X_array) - validation_rows)
        val_positions = np.arange(len(X_array) - validation_rows, len(X_array))
    else:
        train_positions = np.arange(0, len(X_array))
        val_positions = np.asarray([], dtype=int)

    feature_scaler = WindowStandardizer().fit(X_array[train_positions])
    target_scaler = TargetStandardizer(enabled=bool(torch_cfg.get("standardize_targets", True))).fit(y_array[train_positions])
    X_train = feature_scaler.transform(X_array[train_positions])
    y_train = target_scaler.transform(y_array[train_positions])
    batch_size = int(seq_cfg.get("batch_size", torch_cfg.get("batch_size", 128)))
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(seq_cfg.get("num_workers", 0)),
    )
    val_loader = None
    if len(val_positions):
        X_val = feature_scaler.transform(X_array[val_positions])
        y_val = target_scaler.transform(y_array[val_positions])
        val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)), batch_size=batch_size)

    device = select_device(config)
    model = make_torch_model("tft", lookback, len(feature_columns), len(target_columns), config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(torch_cfg.get("learning_rate", 0.0005)),
        weight_decay=float(torch_cfg.get("weight_decay", 0.001)),
    )
    loss_fn = nn.MSELoss()
    early_stopping = EarlyStopping(patience=int(torch_cfg.get("patience", 8)))
    max_epochs = int(torch_cfg.get("max_epochs", 50))
    gradient_clip_norm = float(torch_cfg.get("gradient_clip_norm", 1.0))
    best_state = None
    best_rmse = None
    best_epoch = None
    epoch_count = 0
    for epoch in range(max_epochs):
        epoch_count = epoch + 1
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(X_batch), y_batch)
            loss.backward()
            if gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
        if val_loader is not None:
            rmse = _torch_validation_rmse(model, val_loader, device)
            if best_rmse is None or rmse < best_rmse:
                best_rmse = rmse
                best_epoch = epoch_count
                best_state = deepcopy(model.state_dict())
            if early_stopping.step(rmse):
                break
    if best_state is not None and bool(torch_cfg.get("restore_best_checkpoint", True)):
        model.load_state_dict(best_state)

    return FittedTorchNowcastModel(
        model_type="tft",
        model=model,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        feature_columns=feature_columns,
        target_columns=target_columns,
        lookback_rows=lookback,
        device=device,
        metadata={
            "n_train_rows": int(len(train_df)),
            "train_start_date": str(pd.Timestamp(train_df.index.min()).date()),
            "train_end_date": str(pd.Timestamp(train_df.index.max()).date()),
            "lookback_rows": lookback,
            "epoch_count": epoch_count,
            "best_epoch": best_epoch,
            "best_validation_rmse": best_rmse,
            "device": str(device),
        },
    )


def rolling_mean_predict(history: pd.DataFrame, target_columns: list[str], window_rows: int) -> pd.Series:
    return history[target_columns].tail(window_rows).mean()


def ewma_predict(history: pd.DataFrame, target_columns: list[str], span: int) -> pd.Series:
    return history[target_columns].ewm(span=span, adjust=False).mean().iloc[-1]


def _torch_validation_rmse(model, loader, device) -> float:
    import torch

    model.eval()
    errors: list[torch.Tensor] = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            err = model(X_batch.to(device)) - y_batch.to(device)
            errors.append(err.pow(2).detach().cpu())
    return float(torch.cat(errors).mean().sqrt().item())
