from __future__ import annotations

import hashlib
import io
import logging
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from ff5_predictor.io import (
    ensure_dir,
    normalize_datetime_index,
    package_version,
    read_parquet_with_metadata,
    utc_now_iso,
    write_parquet_with_metadata,
)

LOGGER = logging.getLogger(__name__)

KENNETH_FRENCH_FF5_DAILY_ZIP_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)
KENNETH_FRENCH_MOM_DAILY_ZIP_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Momentum_Factor_daily_CSV.zip"
)
FF5_COLUMNS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
MOM_COLUMN = "Mom"


def _load_ff5(config: dict[str, Any]) -> pd.DataFrame:
    data_config = config["data"]
    cache_dir = ensure_dir(data_config["cache_dir"])
    cache_path = cache_dir / "ff5_daily_decimal.parquet"
    force_refresh = bool(data_config.get("force_refresh", False))
    policy = str(data_config.get("official_cache_policy", "max_age"))
    max_age = float(data_config.get("official_cache_max_age_hours", 24))
    if policy not in {"max_age", "refresh", "cache_only"}:
        raise ValueError("data.official_cache_policy must be max_age, refresh, or cache_only")
    if not np.isfinite(max_age) or max_age < 0:
        raise ValueError("data.official_cache_max_age_hours must be finite and nonnegative")
    if force_refresh and policy == "cache_only":
        raise ValueError("force_refresh cannot be combined with official_cache_policy=cache_only")

    now = utc_now_iso()
    cached = None
    cache_metadata: dict[str, Any] = {}
    if cache_path.exists():
        try:
            candidate, cache_metadata = read_parquet_with_metadata(cache_path)
            candidate = _validate_official_frame(candidate, config)
            checksum = cache_metadata.get("content_sha256")
            if checksum and checksum != _frame_checksum(candidate):
                raise ValueError("Official factor cache checksum mismatch")
            cached = candidate
            from ff5_predictor.vintages import store_for
            store_for(config).ingest(candidate, "official", source="legacy_or_current_cache_observed_now")
        except (ValueError, OSError, KeyError) as exc:
            LOGGER.warning("Official factor cache is unusable: %s", exc)

    if policy == "cache_only":
        if cached is None:
            raise RuntimeError("cache_only requires a valid official factor cache")
        LOGGER.warning("Using official factor cache without checking the remote publication")
        return _with_provenance(cached, config, cache_metadata, "cache_only", now)

    age = _cache_age_hours(cache_metadata, now)
    if (
        cached is not None and not force_refresh and policy == "max_age"
        and cache_metadata.get("cache_scope") == "full_download"
        and age is not None and age < max_age
    ):
        return _with_provenance(cached, config, cache_metadata, "fresh_cache", now)

    try:
        df = _validate_official_frame(fetch_ff5_daily(config), config)
        if cached is not None and df.index.max() < cached.index.max():
            raise ValueError("Refreshed official factors end before the existing cache")
    except Exception as exc:
        if cached is not None and bool(data_config.get("allow_stale_official_cache", False)):
            LOGGER.warning("Official factor refresh failed; explicitly allowed stale cache used: %s", exc)
            return _with_provenance(cached, config, cache_metadata, "stale_fallback", now, str(exc))
        raise RuntimeError("Official factor refresh failed; existing cache was not replaced") from exc

    metadata = {
        "source": "getFamaFrenchFactors_or_kenneth_french_remote_zip",
        "dataset": "F-F_Research_Data_5_Factors_2x3_daily_plus_momentum_daily",
        "download_timestamp_utc": utc_now_iso(),
        "cache_scope": "full_download",
        "content_sha256": _frame_checksum(df),
        "start_date": str(df.index.min().date()),
        "end_date": str(df.index.max().date()),
        "tickers": [],
        "library_versions": {
            "pandas": package_version("pandas"),
            "requests": package_version("requests"),
            "getFamaFrenchFactors": package_version("getFamaFrenchFactors"),
        },
        "units": "decimal",
    }
    write_parquet_with_metadata(df, cache_path, metadata)
    return _with_provenance(df, config, metadata, "refreshed", now)


def _validate_official_frame(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    frame = normalize_datetime_index(df)
    required = list(dict.fromkeys(FF5_COLUMNS + list(config.get("prediction", {}).get("target_columns", []))))
    missing = [column for column in required if column not in frame]
    if frame.empty or frame.index.hasnans or missing:
        raise ValueError(f"Official factors are empty, have invalid dates, or lack columns: {missing}")
    if not np.isfinite(frame[required].to_numpy(dtype=float)).all():
        raise ValueError("Official factors contain nonfinite values")
    return frame


def _frame_checksum(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256(
        pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
    )
    digest.update(",".join(frame.columns).encode("utf-8"))
    return digest.hexdigest()


def _cache_age_hours(metadata: dict[str, Any], now: str) -> float | None:
    try:
        timestamp = pd.to_datetime(metadata["download_timestamp_utc"], utc=True)
        age = (pd.Timestamp(now) - timestamp).total_seconds() / 3600
        return float(age) if np.isfinite(age) and age >= 0 else None
    except (KeyError, ValueError, TypeError):
        return None


def _with_provenance(
    frame: pd.DataFrame, config: dict[str, Any], metadata: dict[str, Any],
    status: str, checked_at: str, refresh_error: str | None = None,
) -> pd.DataFrame:
    from ff5_predictor.vintages import store_for
    store = store_for(config)
    snapshot_id = store.ingest(frame, "official", raw=[store.get_bytes(k) for k in frame.attrs.get("raw_hashes", [])], source=metadata.get("source"))
    result = _filter_dates(frame, config).copy()
    result.attrs["snapshot_id"] = snapshot_id
    result.attrs["official_data"] = {
        "cache_status": status,
        "cache_checked_at_utc": checked_at,
        "download_timestamp_utc": metadata.get("download_timestamp_utc"),
        "source": metadata.get("source"),
        "content_sha256": _frame_checksum(frame),
        "latest_cached_factor_date": str(frame.index.max().date()),
        "refresh_error": refresh_error,
    }
    return result


def fetch_ff5_daily(config: dict[str, Any]) -> pd.DataFrame:
    wrapper_df = _try_get_fama_french_factors()
    if wrapper_df is not None:
        LOGGER.info("Loaded FF5 daily data using getFamaFrenchFactors")
        try:
            return _join_momentum(wrapper_df, _download_momentum_daily(config))
        except Exception as exc:
            if MOM_COLUMN in config.get("prediction", {}).get("target_columns", []):
                raise
            LOGGER.debug("Could not add daily momentum to wrapper FF5 output: %s", exc)
            return wrapper_df

    LOGGER.info("Downloading FF5 daily data from Kenneth French remote zip")
    response = requests.get(KENNETH_FRENCH_FF5_DAILY_ZIP_URL, timeout=60)
    response.raise_for_status()

    raw_dir = ensure_dir(config["data"].get("raw_dir", "data/raw"))
    raw_zip_path = raw_dir / "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
    raw_zip_path.write_bytes(response.content)
    from ff5_predictor.vintages import store_for
    store_for(config).put_bytes(response.content)
    ff5 = parse_kenneth_french_ff5_zip(response.content)
    ff5.attrs["raw_hashes"] = [store_for(config).put_bytes(response.content)]
    mom = _download_momentum_daily(config)
    return _join_momentum(ff5, mom)


def parse_kenneth_french_ff5_zip(zip_bytes: bytes) -> pd.DataFrame:
    """Parse a freshly supplied Kenneth French FF5 zip payload.

    The function accepts bytes only to avoid accidentally reading repository-local
    CSV or zip files when refresh is requested.
    """
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        csv_names = [name for name in zf.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError("Kenneth French zip did not contain a CSV file")
        text = zf.read(csv_names[0]).decode("latin1")

    lines = text.splitlines()
    header_idx = next(
        (
            i
            for i, line in enumerate(lines)
            if "Mkt-RF" in line and "SMB" in line and "RF" in line
        ),
        None,
    )
    if header_idx is None:
        raise ValueError("Could not find FF5 header row in Kenneth French CSV")

    data_lines = []
    for line in lines[header_idx:]:
        stripped = line.strip()
        if not stripped:
            break
        first_cell = stripped.split(",", 1)[0].strip()
        if line == lines[header_idx] or first_cell.isdigit():
            data_lines.append(line)

    df = pd.read_csv(io.StringIO("\n".join(data_lines)))
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    df = df.set_index("date")
    df = df[FF5_COLUMNS].apply(pd.to_numeric, errors="coerce")
    df = df / 100.0
    df = normalize_datetime_index(df)
    return df.dropna(subset=FF5_COLUMNS)


def parse_kenneth_french_momentum_zip(zip_bytes: bytes) -> pd.DataFrame:
    """Parse a freshly supplied Kenneth French daily momentum zip payload."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        csv_names = [name for name in zf.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError("Kenneth French momentum zip did not contain a CSV file")
        text = zf.read(csv_names[0]).decode("latin1")

    lines = text.splitlines()
    header_idx = next(
        (i for i, line in enumerate(lines) if any(cell.strip() == MOM_COLUMN for cell in line.split(","))),
        None,
    )
    if header_idx is None:
        raise ValueError("Could not find Mom header row in Kenneth French momentum CSV")

    data_lines = []
    for line in lines[header_idx:]:
        stripped = line.strip()
        if not stripped:
            break
        first_cell = stripped.split(",", 1)[0].strip()
        if line == lines[header_idx] or first_cell.isdigit():
            data_lines.append(line)

    df = pd.read_csv(io.StringIO("\n".join(data_lines)))
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    df = df.set_index("date")
    df = df[[MOM_COLUMN]].apply(pd.to_numeric, errors="coerce") / 100.0
    df = normalize_datetime_index(df)
    return df.dropna(subset=[MOM_COLUMN])


def _download_momentum_daily(config: dict[str, Any]) -> pd.DataFrame:
    LOGGER.info("Downloading daily momentum factor from Kenneth French remote zip")
    response = requests.get(KENNETH_FRENCH_MOM_DAILY_ZIP_URL, timeout=60)
    response.raise_for_status()
    raw_dir = ensure_dir(config["data"].get("raw_dir", "data/raw"))
    raw_zip_path = raw_dir / "F-F_Momentum_Factor_daily_CSV.zip"
    raw_zip_path.write_bytes(response.content)
    from ff5_predictor.vintages import store_for
    store_for(config).put_bytes(response.content)
    frame = parse_kenneth_french_momentum_zip(response.content)
    frame.attrs["raw_hashes"] = [store_for(config).put_bytes(response.content)]
    return frame


def _join_momentum(ff5: pd.DataFrame, mom: pd.DataFrame) -> pd.DataFrame:
    joined = normalize_datetime_index(ff5).join(normalize_datetime_index(mom), how="inner")
    columns = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", MOM_COLUMN, "RF"]
    result = joined[columns].dropna(subset=columns)
    result.attrs["raw_hashes"] = ff5.attrs.get("raw_hashes", []) + mom.attrs.get("raw_hashes", [])
    return result


def _try_get_fama_french_factors() -> pd.DataFrame | None:
    try:
        import getFamaFrenchFactors as gff  # type: ignore
    except Exception:
        return None

    candidates = [
        ("get_fama_french_factors", {"frequency": "d"}),
        ("getFamaFrenchFactors", {"frequency": "d"}),
    ]
    for function_name, kwargs in candidates:
        func = getattr(gff, function_name, None)
        if func is None:
            continue
        try:
            raw = func("F-F_Research_Data_5_Factors_2x3", **kwargs)
        except Exception as exc:
            LOGGER.debug("getFamaFrenchFactors call failed: %s", exc)
            continue
        try:
            return _normalize_wrapper_output(raw)
        except Exception as exc:
            LOGGER.debug("Could not normalize getFamaFrenchFactors output: %s", exc)
            continue
    return None


def _normalize_wrapper_output(raw: Any) -> pd.DataFrame:
    if isinstance(raw, dict):
        frames = [value for value in raw.values() if isinstance(value, pd.DataFrame)]
        if not frames:
            raise ValueError("Wrapper returned dict without DataFrame values")
        df = frames[0].copy()
    elif isinstance(raw, pd.DataFrame):
        df = raw.copy()
    else:
        raise ValueError(f"Unsupported wrapper output type: {type(raw)!r}")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    missing = [col for col in FF5_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Wrapper output missing columns: {missing}")

    df = df[FF5_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if df["Mkt-RF"].abs().median(skipna=True) > 0.2:
        df = df / 100.0
    return normalize_datetime_index(df).dropna(subset=FF5_COLUMNS)


def _filter_dates(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    result = df
    start = config["data"].get("start_date")
    end = config["data"].get("end_date")
    if start:
        result = result.loc[pd.Timestamp(start) :]
    if end:
        result = result.loc[: pd.Timestamp(end)]
    return result


def _missing_configured_columns(df: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    required = list(config.get("prediction", {}).get("target_columns", []))
    if config.get("prediction", {}).get("predict_rf", False) or "RF" in df.columns:
        required.append("RF")
    return [column for column in required if column not in df.columns]


def load_ff5(config: dict[str, Any]) -> pd.DataFrame:
    from ff5_predictor.vintages import load_snapshot
    return load_snapshot(config, "official", _load_ff5)
