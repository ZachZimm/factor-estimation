from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path
from typing import Any

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
FF5_COLUMNS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]


def load_ff5(config: dict[str, Any]) -> pd.DataFrame:
    cache_dir = ensure_dir(config["data"]["cache_dir"])
    cache_path = cache_dir / "ff5_daily_decimal.parquet"
    force_refresh = bool(config["data"].get("force_refresh", False))

    if cache_path.exists() and not force_refresh:
        df, _ = read_parquet_with_metadata(cache_path)
        return _filter_dates(normalize_datetime_index(df), config)

    df = fetch_ff5_daily(config)
    df = _filter_dates(df, config)
    metadata = {
        "source": "getFamaFrenchFactors_or_kenneth_french_remote_zip",
        "dataset": "F-F_Research_Data_5_Factors_2x3_daily",
        "download_timestamp_utc": utc_now_iso(),
        "start_date": config["data"].get("start_date"),
        "end_date": config["data"].get("end_date"),
        "tickers": [],
        "library_versions": {
            "pandas": package_version("pandas"),
            "requests": package_version("requests"),
            "getFamaFrenchFactors": package_version("getFamaFrenchFactors"),
        },
        "units": "decimal",
    }
    write_parquet_with_metadata(df, cache_path, metadata)
    return df


def fetch_ff5_daily(config: dict[str, Any]) -> pd.DataFrame:
    wrapper_df = _try_get_fama_french_factors()
    if wrapper_df is not None:
        LOGGER.info("Loaded FF5 daily data using getFamaFrenchFactors")
        return wrapper_df

    LOGGER.info("Downloading FF5 daily data from Kenneth French remote zip")
    response = requests.get(KENNETH_FRENCH_FF5_DAILY_ZIP_URL, timeout=60)
    response.raise_for_status()

    raw_dir = ensure_dir(config["data"].get("raw_dir", "data/raw"))
    raw_zip_path = raw_dir / "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
    raw_zip_path.write_bytes(response.content)
    return parse_kenneth_french_ff5_zip(response.content)


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
