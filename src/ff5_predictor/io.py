from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def package_version(package_name: str) -> str | None:
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return None


def cache_key_for_tickers(tickers: list[str]) -> str:
    normalized = ",".join(sorted(t.upper() for t in tickers))
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]
    return f"{normalized.replace(',', '_')}_{digest}"


def metadata_path_for(data_path: str | Path) -> Path:
    path = Path(data_path)
    return path.with_suffix(path.suffix + ".metadata.json")


def write_parquet_with_metadata(
    df: pd.DataFrame,
    data_path: str | Path,
    metadata: dict[str, Any],
    metadata_path: str | Path | None = None,
) -> None:
    path = Path(data_path)
    ensure_dir(path.parent)
    df.to_parquet(path)
    final_metadata = {
        **metadata,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "written_timestamp_utc": utc_now_iso(),
    }
    sidecar = Path(metadata_path) if metadata_path else metadata_path_for(path)
    with sidecar.open("w", encoding="utf-8") as fh:
        json.dump(final_metadata, fh, indent=2, sort_keys=True, default=str)


def read_parquet_with_metadata(
    data_path: str | Path,
    metadata_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = Path(data_path)
    df = pd.read_parquet(path)
    sidecar = Path(metadata_path) if metadata_path else metadata_path_for(path)
    metadata: dict[str, Any] = {}
    if sidecar.exists():
        with sidecar.open("r", encoding="utf-8") as fh:
            metadata = json.load(fh)
    return df, metadata


def normalize_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result.index = pd.to_datetime(result.index)
    if getattr(result.index, "tz", None) is not None:
        result.index = result.index.tz_convert(None)
    result.index = result.index.normalize()
    result = result[~result.index.duplicated(keep="last")].sort_index()
    return result
