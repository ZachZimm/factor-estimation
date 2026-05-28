from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ff5_predictor.io import ensure_dir


def nowcast_name(config: dict[str, Any]) -> str:
    return str(config.get("nowcast", {}).get("run_name", "daily_ff5_nowcast_v1"))


def nowcast_base_dir(config: dict[str, Any]) -> Path:
    return Path(config.get("nowcast", {}).get("output_dir", "data/nowcasts")) / nowcast_name(config)


def create_nowcast_run_dir(config: dict[str, Any]) -> Path:
    base = ensure_dir(nowcast_base_dir(config))
    if bool(config.get("output", {}).get("force_overwrite", False)):
        run_dir = base / "latest"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        return ensure_dir(run_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = base / timestamp
    counter = 1
    while run_dir.exists():
        run_dir = base / f"{timestamp}_{counter}"
        counter += 1
    return ensure_dir(run_dir)


def latest_nowcast_dir(config: dict[str, Any]) -> Path:
    return nowcast_base_dir(config) / "latest"


def sync_latest_copy(config: dict[str, Any], run_dir: Path) -> None:
    latest = latest_nowcast_dir(config)
    if latest.resolve() == run_dir.resolve():
        return
    if latest.exists():
        shutil.rmtree(latest)
    shutil.copytree(run_dir, latest)


def write_json(path: Path, data: dict[str, Any]) -> Path:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True, default=str)
    return path


def write_yaml(path: Path, data: dict[str, Any]) -> Path:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=True)
    return path


def write_nowcast_predictions(run_dir: Path, name: str, predictions: pd.DataFrame) -> Path:
    path = ensure_dir(run_dir / "predictions") / name
    predictions.to_csv(path, index=False)
    return path


def write_nowcast_dataset(run_dir: Path, name: str, df: pd.DataFrame) -> Path:
    path = ensure_dir(run_dir / "datasets") / name
    df.to_parquet(path)
    return path
