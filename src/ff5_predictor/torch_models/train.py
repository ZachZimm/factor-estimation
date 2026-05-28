from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
from typing import Any

import logging
import numpy as np
import pandas as pd

from ff5_predictor.io import ensure_dir
from ff5_predictor.rolling_train import prediction_output_columns
from ff5_predictor.sequence_dataset import build_sequence_arrays
from ff5_predictor.torch_models import make_torch_model, normalize_torch_model_type
from ff5_predictor.torch_models.common import (
    EarlyStopping,
    TargetStandardizer,
    WindowStandardizer,
    count_parameters,
    select_device,
    set_random_seed,
)
from ff5_predictor.walk_forward import build_walk_forward_checkpoints

LOGGER = logging.getLogger(__name__)


@dataclass
class TorchTrainingResult:
    predictions: pd.DataFrame
    checkpoint_metrics: list[dict[str, Any]]


def train_torch_walk_forward(
    modeling_df: pd.DataFrame,
    feature_columns: list[str],
    target_columns: list[str],
    model_type: str,
    config: dict[str, Any],
) -> TorchTrainingResult:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    seed = int(config.get("experiments", {}).get("random_seed", 42))
    set_random_seed(seed, deterministic=bool(config.get("torch", {}).get("deterministic", True)))
    device = select_device(config)
    torch_cfg = config.get("torch", {})
    if bool(torch_cfg.get("log_device", True)):
        if device.type == "cuda":
            LOGGER.info(
                "Using torch device %s: %s, torch=%s, cuda=%s",
                device,
                torch.cuda.get_device_name(device),
                torch.__version__,
                torch.version.cuda,
            )
        else:
            LOGGER.info("Using torch device %s, torch=%s", device, torch.__version__)
    seq_cfg = config.get("sequence", {})
    wf_cfg = config.get("walk_forward", {})
    lookback_rows = int(seq_cfg.get("lookback_rows", 63))
    arrays = build_sequence_arrays(
        modeling_df=modeling_df,
        feature_columns=feature_columns,
        target_columns=target_columns,
        lookback_rows=lookback_rows,
        min_sequence_rows=lookback_rows,
    )
    splits = build_walk_forward_checkpoints(
        dates=arrays.dates,
        train_window_rows=int(wf_cfg.get("train_window_rows", 1260)),
        min_train_rows=int(wf_cfg.get("min_train_rows", 1000)),
        validation_window_rows=int(wf_cfg.get("validation_window_rows", 252)),
        retrain_frequency=str(wf_cfg.get("retrain_frequency", wf_cfg.get("checkpoint_frequency", "monthly"))),
        require_validation=bool(wf_cfg.get("require_validation", False)),
    )

    max_epochs = int(torch_cfg.get("max_epochs", 50))
    batch_size = int(seq_cfg.get("batch_size", torch_cfg.get("batch_size", 128)))
    learning_rate = float(torch_cfg.get("learning_rate", 0.001))
    weight_decay = float(torch_cfg.get("weight_decay", 0.0001))
    patience = int(torch_cfg.get("patience", 8))
    gradient_clip_norm = float(torch_cfg.get("gradient_clip_norm", 1.0))
    num_workers = int(seq_cfg.get("num_workers", torch_cfg.get("num_workers", 0)))
    standardize_targets = bool(torch_cfg.get("standardize_targets", True))
    restore_best_checkpoint = bool(torch_cfg.get("restore_best_checkpoint", True))

    records: list[dict[str, Any]] = []
    checkpoint_metrics: list[dict[str, Any]] = []
    normalized_model_type = normalize_torch_model_type(model_type)

    for checkpoint_id, split in enumerate(splits):
        X_train = arrays.X[split.train_positions]
        y_train = arrays.y[split.train_positions]
        feature_scaler = WindowStandardizer().fit(X_train)
        target_scaler = TargetStandardizer(enabled=standardize_targets).fit(y_train)

        X_train_scaled = feature_scaler.transform(X_train)
        y_train_scaled = target_scaler.transform(y_train)
        train_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_train_scaled),
                torch.from_numpy(y_train_scaled),
            ),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        validation_loader = None
        if len(split.validation_positions):
            X_val = feature_scaler.transform(arrays.X[split.validation_positions])
            y_val = target_scaler.transform(arrays.y[split.validation_positions])
            validation_loader = DataLoader(
                TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )

        model = make_torch_model(
            model_type=normalized_model_type,
            lookback_rows=lookback_rows,
            n_features=len(feature_columns),
            n_targets=len(target_columns),
            config=config,
        ).to(device)
        if checkpoint_id == 0 and device.type == "cuda":
            LOGGER.info(
                "Initialized %s on CUDA with %.2f MiB allocated",
                normalized_model_type,
                torch.cuda.memory_allocated(device) / (1024 * 1024),
            )
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()
        early_stopping = EarlyStopping(patience=patience)
        best_validation_rmse: float | None = None
        best_epoch: int | None = None
        best_state: dict[str, Any] | None = None
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

            if validation_loader is not None:
                validation_rmse = _evaluate_rmse(model, validation_loader, device)
                if best_validation_rmse is None or validation_rmse < best_validation_rmse:
                    best_validation_rmse = validation_rmse
                    best_epoch = epoch_count
                    best_state = deepcopy(model.state_dict())
                if early_stopping.step(validation_rmse):
                    break

        if restore_best_checkpoint and best_state is not None:
            model.load_state_dict(best_state)

        X_pred = feature_scaler.transform(arrays.X[split.prediction_positions])
        pred_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_pred)),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        pred_scaled = _predict(model, pred_loader, device)
        pred = target_scaler.inverse_transform(pred_scaled)

        for row_idx, sequence_pos in enumerate(split.prediction_positions):
            target_date = arrays.dates[sequence_pos]
            actual = arrays.y[sequence_pos]
            record: dict[str, Any] = {
                "date": target_date,
                "model_type": normalized_model_type,
                "train_start_date": split.train_start_date,
                "train_end_date": split.train_end_date,
                "n_train_rows": len(split.train_positions),
                "checkpoint_id": checkpoint_id,
                "epoch_count": epoch_count,
                "best_validation_rmse": best_validation_rmse,
                "best_epoch": best_epoch,
                "device": str(device),
                "lookback_rows": lookback_rows,
                "n_fit_rows": len(split.train_positions),
                "n_validation_rows": len(split.validation_positions),
            }
            for column, value in zip(target_columns, pred[row_idx]):
                record[f"pred_{column}"] = float(value)
            for column, value in zip(target_columns, actual):
                record[f"actual_{column}"] = float(value)
            records.append(record)

        checkpoint_metrics.append(
            {
                "checkpoint_id": checkpoint_id,
                "model_type": normalized_model_type,
                "checkpoint_date": split.checkpoint_date.isoformat(),
                "train_start_date": split.train_start_date.isoformat(),
                "train_end_date": split.train_end_date.isoformat(),
                "predict_start_date": split.predict_start_date.isoformat(),
                "predict_end_date": split.predict_end_date.isoformat(),
                "n_train_rows": int(len(split.train_positions)),
                "n_fit_rows": int(len(split.train_positions)),
                "n_validation_rows": int(len(split.validation_positions)),
                "n_prediction_rows": int(len(split.prediction_positions)),
                "epoch_count": epoch_count,
                "best_epoch": best_epoch,
                "best_validation_rmse": best_validation_rmse,
                "device": str(device),
                "n_parameters": count_parameters(model),
            }
        )

        if bool(torch_cfg.get("save_checkpoints", False)):
            output_dir = Path(config.get("experiments", {}).get("output_dir", "data/experiments"))
            run_name = config.get("experiments", {}).get("run_name") or config.get("experiment", {}).get("name") or "experiment"
            checkpoint_dir = ensure_dir(output_dir / run_name / "checkpoints" / normalized_model_type)
            torch.save(model.state_dict(), checkpoint_dir / f"checkpoint_{checkpoint_id:04d}.pt")

    columns = prediction_output_columns(target_columns) + [
        "checkpoint_id",
        "epoch_count",
        "best_validation_rmse",
        "best_epoch",
        "device",
        "lookback_rows",
        "n_fit_rows",
        "n_validation_rows",
    ]
    return TorchTrainingResult(
        predictions=pd.DataFrame.from_records(records, columns=columns),
        checkpoint_metrics=checkpoint_metrics,
    )


def _evaluate_rmse(model, loader, device) -> float:
    import torch

    model.eval()
    squared_errors: list[torch.Tensor] = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            err = model(X_batch) - y_batch
            squared_errors.append(err.pow(2).detach().cpu())
    return float(torch.cat(squared_errors).mean().sqrt().item())


def _predict(model, loader, device) -> np.ndarray:
    import torch

    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for (X_batch,) in loader:
            out = model(X_batch.to(device)).detach().cpu().numpy()
            predictions.append(out)
    return np.vstack(predictions).astype(np.float32) if predictions else np.empty((0, 0), dtype=np.float32)
