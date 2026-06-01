from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from typing import Any
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, MultiTaskElasticNet, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from ff5_predictor.feature_extraction import (
    FittedFeatureExtractor,
    FittedPerTargetExtractor,
    fit_feature_extractor,
    should_apply_feature_extraction,
    transform_feature_frame,
)


@dataclass
class FittedNowcastModel:
    model_type: str
    model: Any
    scaler: StandardScaler | None
    feature_columns: list[str]
    target_columns: list[str]
    metadata: dict[str, Any]
    extractor: FittedFeatureExtractor | None = None

    def model_feature_frame(self, features: pd.DataFrame) -> pd.DataFrame:
        if self.extractor is not None:
            transformed, _ = transform_feature_frame(self.extractor, features[self.extractor.feature_columns_in])
            return transformed
        return features[self.feature_columns].astype(float)

    def predict_frame(self, features: pd.DataFrame) -> np.ndarray:
        X_frame = self.model_feature_frame(features)
        X = X_frame.to_numpy()
        if self.scaler is not None:
            X = self.scaler.transform(X_frame)
        return np.asarray(self.model.predict(X))


@dataclass
class FittedPerTargetNowcastModel:
    model_type: str
    fitted_by_target: dict[str, FittedNowcastModel]
    extractor: FittedPerTargetExtractor
    feature_columns: list[str]
    target_columns: list[str]
    metadata: dict[str, Any]

    def predict_frame(self, features: pd.DataFrame) -> np.ndarray:
        values = []
        for target in self.target_columns:
            fitted = self.fitted_by_target[target]
            transformed = self.extractor.transform_for_target(features, target)
            values.append(float(fitted.predict_frame(transformed)[0]))
        return np.asarray([values], dtype=float)


@dataclass
class FittedPerFactorNowcastModel:
    model_type: str
    fitted_by_target: dict[str, FittedNowcastModel]
    feature_columns: list[str]
    target_columns: list[str]
    metadata: dict[str, Any]

    def predict_frame(self, features: pd.DataFrame) -> np.ndarray:
        values = []
        for target in self.target_columns:
            values.append(float(self.fitted_by_target[target].predict_frame(features)[0]))
        return np.asarray([values], dtype=float)


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
    extractor: FittedFeatureExtractor | None = None

    def model_feature_frame(self, features: pd.DataFrame) -> pd.DataFrame:
        if self.extractor is not None:
            transformed, _ = transform_feature_frame(self.extractor, features[self.extractor.feature_columns_in])
            return transformed
        return features[self.feature_columns].astype(float)

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
    extractor = fit_feature_extractor(train_df, feature_columns, target_columns, config, model_type="ridge")
    if extractor is not None and not isinstance(extractor, FittedFeatureExtractor):
        raise ValueError("Ridge requires a single fitted feature extractor")
    if extractor is not None:
        X_frame, model_feature_columns = transform_feature_frame(extractor, train_df[feature_columns])
    else:
        X_frame = train_df[feature_columns].astype(float)
        model_feature_columns = feature_columns
    X = X_frame.to_numpy()
    y = train_df[target_columns].astype(float).to_numpy()
    scale = bool(model_cfg.get("scale_features", True))
    scaler = StandardScaler() if scale else None

    alpha = float(model_cfg.get("alpha", 10.0))
    selected_alpha = alpha
    validation_rows = int(model_cfg.get("validation_window_rows", 252))
    if bool(model_cfg.get("tune_alpha", True)) and len(train_df) > validation_rows + 10:
        fit_X = X_frame.iloc[:-validation_rows]
        fit_y = y[:-validation_rows]
        val_X = X_frame.iloc[-validation_rows:]
        val_y = y[-validation_rows:]
        best_score = float("inf")
        for candidate in [float(v) for v in model_cfg.get("alpha_grid", [alpha])]:
            candidate_scaler = StandardScaler() if scale else None
            candidate_fit_X = candidate_scaler.fit_transform(fit_X) if candidate_scaler else fit_X.to_numpy()
            candidate_val_X = candidate_scaler.transform(val_X) if candidate_scaler else val_X.to_numpy()
            candidate_model = Ridge(alpha=candidate)
            candidate_model.fit(candidate_fit_X, fit_y)
            pred = candidate_model.predict(candidate_val_X)
            score = float(np.sqrt(mean_squared_error(val_y, pred)))
            if score < best_score:
                best_score = score
                selected_alpha = candidate

    X_fit = scaler.fit_transform(X_frame) if scaler else X
    model = Ridge(alpha=selected_alpha)
    model.fit(X_fit, y)
    return FittedNowcastModel(
        model_type="ridge",
        model=model,
        scaler=scaler,
        feature_columns=model_feature_columns,
        target_columns=target_columns,
        metadata={
            "alpha": selected_alpha,
            "n_train_rows": int(len(train_df)),
            "train_start_date": str(pd.Timestamp(train_df.index.min()).date()),
            "train_end_date": str(pd.Timestamp(train_df.index.max()).date()),
            "scale_features": scale,
            "feature_extraction": extractor.metadata if extractor is not None else {"enabled": False},
            "n_raw_features": len(feature_columns),
            "n_model_features": len(model_feature_columns),
        },
        extractor=extractor,
    )


def fit_elasticnet_nowcast(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    target_columns: list[str],
    config: dict[str, Any],
) -> FittedNowcastModel:
    model_cfg = config.get("models", {}).get("elasticnet", {})
    nowcast_cfg = config.get("nowcast", {})
    train_window = int(nowcast_cfg.get("train_window_rows", 2520))
    if train_window > 0:
        train_df = train_df.tail(train_window)
    extractor = fit_feature_extractor(train_df, feature_columns, target_columns, config, model_type="elasticnet")
    if extractor is not None and not isinstance(extractor, FittedFeatureExtractor):
        raise ValueError("ElasticNet requires a single fitted feature extractor")
    if extractor is not None:
        X_frame, model_feature_columns = transform_feature_frame(extractor, train_df[feature_columns])
    else:
        X_frame = train_df[feature_columns].astype(float)
        model_feature_columns = feature_columns
    y = train_df[target_columns].astype(float).to_numpy()
    scale = bool(model_cfg.get("scale_features", True))
    scaler = StandardScaler() if scale else None

    alpha = float(model_cfg.get("alpha", 0.001))
    selected_alpha = alpha
    l1_ratio = float(model_cfg.get("l1_ratio", 0.05))
    selected_l1_ratio = l1_ratio
    max_iter = int(model_cfg.get("max_iter", 50000))
    tol = float(model_cfg.get("tol", 0.0001))
    validation_rows = int(model_cfg.get("validation_window_rows", 252))
    if bool(model_cfg.get("tune_alpha", True)) and len(train_df) > validation_rows + 10:
        fit_X = X_frame.iloc[:-validation_rows]
        fit_y = y[:-validation_rows]
        val_X = X_frame.iloc[-validation_rows:]
        val_y = y[-validation_rows:]
        best_score = float("inf")
        l1_candidates = (
            [float(v) for v in model_cfg.get("l1_ratio_grid", [l1_ratio])]
            if bool(model_cfg.get("tune_l1_ratio", False))
            else [l1_ratio]
        )
        for candidate_l1_ratio in l1_candidates:
            for candidate in [float(v) for v in model_cfg.get("alpha_grid", [alpha])]:
                candidate_scaler = StandardScaler() if scale else None
                candidate_fit_X = candidate_scaler.fit_transform(fit_X) if candidate_scaler else fit_X.to_numpy()
                candidate_val_X = candidate_scaler.transform(val_X) if candidate_scaler else val_X.to_numpy()
                candidate_model, converged = _fit_elasticnet_model(
                    candidate_fit_X,
                    fit_y,
                    alpha=candidate,
                    l1_ratio=candidate_l1_ratio,
                    max_iter=max_iter,
                    tol=tol,
                    selection=str(model_cfg.get("selection", "cyclic")),
                )
                if not converged:
                    continue
                pred = candidate_model.predict(candidate_val_X)
                score = float(np.sqrt(mean_squared_error(val_y, pred)))
                if score < best_score:
                    best_score = score
                    selected_alpha = candidate
                    selected_l1_ratio = candidate_l1_ratio

    X_fit = scaler.fit_transform(X_frame) if scaler else X_frame.to_numpy()
    model, converged = _fit_elasticnet_model(
        X_fit,
        y,
        alpha=selected_alpha,
        l1_ratio=selected_l1_ratio,
        max_iter=max_iter,
        tol=tol,
        selection=str(model_cfg.get("selection", "cyclic")),
    )
    return FittedNowcastModel(
        model_type="elasticnet",
        model=model,
        scaler=scaler,
        feature_columns=model_feature_columns,
        target_columns=target_columns,
        metadata={
            "alpha": selected_alpha,
            "l1_ratio": selected_l1_ratio,
            "n_train_rows": int(len(train_df)),
            "train_start_date": str(pd.Timestamp(train_df.index.min()).date()),
            "train_end_date": str(pd.Timestamp(train_df.index.max()).date()),
            "scale_features": scale,
            "converged": converged,
            "feature_extraction": extractor.metadata if extractor is not None else {"enabled": False},
            "n_raw_features": len(feature_columns),
            "n_model_features": len(model_feature_columns),
        },
        extractor=extractor,
    )


def fit_per_factor_elasticnet_nowcast(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    target_columns: list[str],
    config: dict[str, Any],
) -> FittedPerFactorNowcastModel:
    model_cfg = config.get("models", {}).get("per_factor_elasticnet", config.get("models", {}).get("elasticnet", {}))
    nowcast_cfg = config.get("nowcast", {})
    train_window = int(nowcast_cfg.get("train_window_rows", 2520))
    if train_window > 0:
        train_df = train_df.tail(train_window)
    X_frame = train_df[feature_columns].astype(float)
    scale = bool(model_cfg.get("scale_features", True))
    validation_rows = int(model_cfg.get("validation_window_rows", 252))
    alpha_grid = [float(v) for v in model_cfg.get("alpha_grid", [model_cfg.get("alpha", 0.001)])]
    l1_candidates = (
        [float(v) for v in model_cfg.get("l1_ratio_grid", [model_cfg.get("l1_ratio", 0.05)])]
        if bool(model_cfg.get("tune_l1_ratio", False))
        else [float(model_cfg.get("l1_ratio", 0.05))]
    )
    max_iter = int(model_cfg.get("max_iter", 50000))
    tol = float(model_cfg.get("tol", 0.0001))
    fitted_by_target: dict[str, FittedNowcastModel] = {}
    selected: dict[str, dict[str, float | bool]] = {}
    for target in target_columns:
        y = train_df[target].astype(float).to_numpy()
        selected_alpha = float(model_cfg.get("alpha", 0.001))
        selected_l1_ratio = float(model_cfg.get("l1_ratio", 0.05))
        if bool(model_cfg.get("tune_alpha", True)) and len(train_df) > validation_rows + 10:
            fit_X = X_frame.iloc[:-validation_rows]
            fit_y = y[:-validation_rows]
            val_X = X_frame.iloc[-validation_rows:]
            val_y = y[-validation_rows:]
            best_score = float("inf")
            for candidate_l1_ratio in l1_candidates:
                for candidate_alpha in alpha_grid:
                    candidate_scaler = StandardScaler() if scale else None
                    candidate_fit_X = candidate_scaler.fit_transform(fit_X) if candidate_scaler else fit_X.to_numpy()
                    candidate_val_X = candidate_scaler.transform(val_X) if candidate_scaler else val_X.to_numpy()
                    candidate_model, converged = _fit_single_elasticnet_model(
                        candidate_fit_X,
                        fit_y,
                        alpha=candidate_alpha,
                        l1_ratio=candidate_l1_ratio,
                        max_iter=max_iter,
                        tol=tol,
                        selection=str(model_cfg.get("selection", "cyclic")),
                    )
                    if not converged:
                        continue
                    pred = candidate_model.predict(candidate_val_X)
                    score = float(np.sqrt(mean_squared_error(val_y, pred)))
                    if score < best_score:
                        best_score = score
                        selected_alpha = candidate_alpha
                        selected_l1_ratio = candidate_l1_ratio
        scaler = StandardScaler() if scale else None
        X_fit = scaler.fit_transform(X_frame) if scaler else X_frame.to_numpy()
        model, converged = _fit_single_elasticnet_model(
            X_fit,
            y,
            alpha=selected_alpha,
            l1_ratio=selected_l1_ratio,
            max_iter=max_iter,
            tol=tol,
            selection=str(model_cfg.get("selection", "cyclic")),
        )
        fitted_by_target[target] = FittedNowcastModel(
            model_type="per_factor_elasticnet",
            model=model,
            scaler=scaler,
            feature_columns=feature_columns,
            target_columns=[target],
            metadata={
                "alpha": selected_alpha,
                "l1_ratio": selected_l1_ratio,
                "n_train_rows": int(len(train_df)),
                "train_start_date": str(pd.Timestamp(train_df.index.min()).date()),
                "train_end_date": str(pd.Timestamp(train_df.index.max()).date()),
                "scale_features": scale,
                "converged": converged,
                "feature_extraction": {"enabled": False},
                "n_raw_features": len(feature_columns),
                "n_model_features": len(feature_columns),
            },
        )
        selected[target] = {"alpha": selected_alpha, "l1_ratio": selected_l1_ratio, "converged": converged}
    return FittedPerFactorNowcastModel(
        model_type="per_factor_elasticnet",
        fitted_by_target=fitted_by_target,
        feature_columns=feature_columns,
        target_columns=target_columns,
        metadata={
            "n_train_rows": int(len(train_df)),
            "train_start_date": str(pd.Timestamp(train_df.index.min()).date()),
            "train_end_date": str(pd.Timestamp(train_df.index.max()).date()),
            "selected_by_target": selected,
            "feature_extraction": {"enabled": False},
            "n_raw_features": len(feature_columns),
            "n_model_features": len(feature_columns),
        },
    )


def fit_per_target_pls_ridge_nowcast(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    target_columns: list[str],
    config: dict[str, Any],
) -> FittedPerTargetNowcastModel:
    nowcast_cfg = config.get("nowcast", {})
    train_window = int(nowcast_cfg.get("train_window_rows", 2520))
    if train_window > 0:
        train_df = train_df.tail(train_window)
    extractor = fit_feature_extractor(train_df, feature_columns, target_columns, config, model_type="per_target_pls_ridge")
    if not isinstance(extractor, FittedPerTargetExtractor):
        raise ValueError("per_target_pls_ridge requires feature_extraction.method='per_target_pls'")
    fitted_by_target: dict[str, FittedNowcastModel] = {}
    selected_alpha_by_target: dict[str, float] = {}
    for target in target_columns:
        X_frame = extractor.transform_for_target(train_df[feature_columns], target)
        target_train_df = pd.concat([X_frame, train_df[[target]]], axis=1)
        single_config = _config_without_feature_extraction(config)
        fitted = fit_ridge_nowcast(target_train_df, list(X_frame.columns), [target], single_config)
        fitted_by_target[target] = fitted
        selected_alpha_by_target[target] = float(fitted.metadata.get("alpha"))
    return FittedPerTargetNowcastModel(
        model_type="per_target_pls_ridge",
        fitted_by_target=fitted_by_target,
        extractor=extractor,
        feature_columns=feature_columns,
        target_columns=target_columns,
        metadata={
            "n_train_rows": int(len(train_df)),
            "train_start_date": str(pd.Timestamp(train_df.index.min()).date()),
            "train_end_date": str(pd.Timestamp(train_df.index.max()).date()),
            "feature_extraction": extractor.metadata,
            "selected_alpha_by_target": selected_alpha_by_target,
            "n_raw_features": len(feature_columns),
            "n_model_features": sum(len(v) for v in extractor.feature_columns_by_target.values()),
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
    extractor = fit_feature_extractor(train_df, feature_columns, target_columns, config, model_type="tft")
    if extractor is not None and not isinstance(extractor, FittedFeatureExtractor):
        raise ValueError("TFT requires a single fitted feature extractor")
    if extractor is not None:
        X_frame, model_feature_columns = transform_feature_frame(extractor, train_df[feature_columns])
        train_df = pd.concat([X_frame, train_df[target_columns]], axis=1)
    else:
        model_feature_columns = feature_columns

    set_random_seed(int(config.get("experiments", {}).get("random_seed", 42)))
    X_values = train_df[model_feature_columns].to_numpy(dtype=np.float32)
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
    model = make_torch_model("tft", lookback, len(model_feature_columns), len(target_columns), config).to(device)
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
        feature_columns=model_feature_columns,
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
            "feature_extraction": extractor.metadata if extractor is not None else {"enabled": False},
            "n_raw_features": len(feature_columns),
            "n_model_features": len(model_feature_columns),
        },
        extractor=extractor,
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


def _fit_elasticnet_model(
    X,
    y,
    *,
    alpha: float,
    l1_ratio: float,
    max_iter: int,
    tol: float,
    selection: str,
) -> tuple[MultiTaskElasticNet, bool]:
    model = MultiTaskElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        max_iter=max_iter,
        tol=tol,
        selection=selection,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            model.fit(X, y)
            return model, True
        except ConvergenceWarning:
            return model, False


def _fit_single_elasticnet_model(
    X,
    y,
    *,
    alpha: float,
    l1_ratio: float,
    max_iter: int,
    tol: float,
    selection: str,
) -> tuple[ElasticNet, bool]:
    model = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        max_iter=max_iter,
        tol=tol,
        selection=selection,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            model.fit(X, y)
            return model, True
        except ConvergenceWarning:
            return model, False


def _config_without_feature_extraction(config: dict[str, Any]) -> dict[str, Any]:
    from copy import deepcopy

    copied = deepcopy(config)
    copied.setdefault("feature_extraction", {})["enabled"] = False
    return copied
