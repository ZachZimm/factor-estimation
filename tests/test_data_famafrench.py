from __future__ import annotations

import io
import zipfile

import pandas as pd
import pytest

from ff5_predictor import data_famafrench
from ff5_predictor.io import write_parquet_with_metadata

from ff5_predictor.data_famafrench import parse_kenneth_french_ff5_zip, parse_kenneth_french_momentum_zip


def _loader_config(tmp_path):
    return {
        "data": {"cache_dir": str(tmp_path), "start_date": "2024-01-01"},
        "prediction": {"target_columns": ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]},
    }


def _official_frame(periods=2):
    return pd.DataFrame(
        {column: [0.001] * periods for column in [*data_famafrench.FF5_COLUMNS, "Mom"]},
        index=pd.bdate_range("2024-01-02", periods=periods),
    )


def test_old_official_cache_refreshes_without_force_refresh(tmp_path, monkeypatch):
    config = _loader_config(tmp_path)
    write_parquet_with_metadata(
        _official_frame(), tmp_path / "ff5_daily_decimal.parquet",
        {"download_timestamp_utc": "2024-01-04T00:00:00+00:00"},
    )
    monkeypatch.setattr(data_famafrench, "utc_now_iso", lambda: "2024-02-01T00:00:00+00:00")
    monkeypatch.setattr(data_famafrench, "fetch_ff5_daily", lambda _: _official_frame(3))

    result = data_famafrench.load_ff5(config)

    assert result.index.max() == pd.Timestamp("2024-01-04")
    assert result.attrs["official_data"]["cache_status"] == "refreshed"


def test_recent_cache_reused_until_ttl_then_refreshed(tmp_path, monkeypatch):
    config = _loader_config(tmp_path)
    calls = []
    monkeypatch.setattr(data_famafrench, "utc_now_iso", lambda: "2024-02-01T00:00:00+00:00")
    def fetch(_):
        calls.append(True)
        return _official_frame(3)
    monkeypatch.setattr(data_famafrench, "fetch_ff5_daily", fetch)
    first = data_famafrench.load_ff5(config)
    monkeypatch.setattr(data_famafrench, "utc_now_iso", lambda: "2024-02-01T23:59:59+00:00")
    second = data_famafrench.load_ff5(config)
    assert len(calls) == 1
    assert second.attrs["official_data"]["cache_status"] == "fresh_cache"
    assert second.attrs["official_data"]["content_sha256"] == first.attrs["official_data"]["content_sha256"]
    monkeypatch.setattr(data_famafrench, "utc_now_iso", lambda: "2024-02-02T00:00:00+00:00")
    data_famafrench.load_ff5(config)
    assert len(calls) == 2


@pytest.mark.parametrize("timestamp", [None, "invalid", "2030-01-01T00:00:00Z"])
def test_unverifiable_cache_age_refreshes(tmp_path, monkeypatch, timestamp):
    config = _loader_config(tmp_path)
    write_parquet_with_metadata(_official_frame(), tmp_path / "ff5_daily_decimal.parquet", {
        "download_timestamp_utc": timestamp, "cache_scope": "full_download",
    })
    monkeypatch.setattr(data_famafrench, "utc_now_iso", lambda: "2024-02-01T00:00:00+00:00")
    monkeypatch.setattr(data_famafrench, "fetch_ff5_daily", lambda _: _official_frame(3))
    assert len(data_famafrench.load_ff5(config)) == 3


@pytest.mark.parametrize("failure", ["network", "empty", "nonfinite", "missing", "older"])
def test_failed_refresh_preserves_cache_and_requires_opt_in_fallback(tmp_path, monkeypatch, failure):
    config = _loader_config(tmp_path)
    monkeypatch.setattr(data_famafrench, "fetch_ff5_daily", lambda _: _official_frame(3))
    data_famafrench.load_ff5(config)
    cache = tmp_path / "ff5_daily_decimal.parquet"
    sidecar = tmp_path / "ff5_daily_decimal.parquet.metadata.json"
    before = (cache.read_bytes(), sidecar.read_bytes())
    config["data"]["official_cache_policy"] = "refresh"
    def fetch(_):
        if failure == "network":
            raise data_famafrench.requests.ConnectionError("offline")
        result = _official_frame(2 if failure == "older" else 3)
        if failure == "empty":
            return result.iloc[:0]
        if failure == "nonfinite":
            result.iloc[0, 0] = float("nan")
        if failure == "missing":
            return result.drop(columns="Mom")
        return result
    monkeypatch.setattr(data_famafrench, "fetch_ff5_daily", fetch)
    with pytest.raises(RuntimeError, match="existing cache was not replaced"):
        data_famafrench.load_ff5(config)
    assert (cache.read_bytes(), sidecar.read_bytes()) == before
    config["data"]["allow_stale_official_cache"] = True
    result = data_famafrench.load_ff5(config)
    assert result.attrs["official_data"]["cache_status"] == "stale_fallback"
    assert result.attrs["official_data"]["refresh_error"]
    assert (cache.read_bytes(), sidecar.read_bytes()) == before


def test_filtered_request_does_not_truncate_shared_cache(tmp_path, monkeypatch):
    config = _loader_config(tmp_path)
    config["data"]["end_date"] = "2024-01-02"
    monkeypatch.setattr(data_famafrench, "fetch_ff5_daily", lambda _: _official_frame(3))
    assert len(data_famafrench.load_ff5(config)) == 1
    config["data"].pop("end_date")
    def no_network(_):
        pytest.fail("Full fresh cache should cover wider request")
    monkeypatch.setattr(data_famafrench, "fetch_ff5_daily", no_network)
    assert len(data_famafrench.load_ff5(config)) == 3


def test_offline_policy_is_explicit_and_force_refresh_conflicts(tmp_path, monkeypatch):
    config = _loader_config(tmp_path)
    config["data"]["official_cache_policy"] = "cache_only"
    with pytest.raises(RuntimeError, match="valid official factor cache"):
        data_famafrench.load_ff5(config)
    write_parquet_with_metadata(_official_frame(), tmp_path / "ff5_daily_decimal.parquet", {})
    def no_network(_):
        pytest.fail("cache_only must never fetch")
    monkeypatch.setattr(data_famafrench, "fetch_ff5_daily", no_network)
    assert data_famafrench.load_ff5(config).attrs["official_data"]["cache_status"] == "cache_only"
    config["data"]["force_refresh"] = True
    with pytest.raises(ValueError, match="cannot be combined"):
        data_famafrench.load_ff5(config)


def test_force_refresh_replaces_even_recent_cache(tmp_path, monkeypatch):
    config = _loader_config(tmp_path)
    monkeypatch.setattr(data_famafrench, "fetch_ff5_daily", lambda _: _official_frame())
    data_famafrench.load_ff5(config)
    monkeypatch.setattr(data_famafrench, "fetch_ff5_daily", lambda _: _official_frame(3))
    config["data"]["force_refresh"] = True
    assert len(data_famafrench.load_ff5(config)) == 3


def test_parse_kenneth_french_zip_uses_supplied_bytes_and_decimal_units() -> None:
    csv_text = "\n".join(
        [
            "Header text",
            ",Mkt-RF,SMB,HML,RMW,CMA,RF",
            "20240102,   1.25,   0.50,  -0.25,   0.10,   0.20,   0.01",
            "20240103,  -2.00,   0.10,   0.20,  -0.30,   0.40,   0.01",
            "",
        ]
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("fresh_download.csv", csv_text)

    df = parse_kenneth_french_ff5_zip(buffer.getvalue())

    assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    assert df.index[0] == pd.Timestamp("2024-01-02")
    assert df.loc[pd.Timestamp("2024-01-02"), "Mkt-RF"] == 0.0125


def test_parse_kenneth_french_momentum_zip_uses_supplied_bytes_and_decimal_units() -> None:
    csv_text = "\n".join(
        [
            "Header text",
            ",Mom,",
            "20240102,   1.25,",
            "20240103,  -2.00,",
            "",
        ]
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("fresh_momentum_download.csv", csv_text)

    df = parse_kenneth_french_momentum_zip(buffer.getvalue())

    assert list(df.columns) == ["Mom"]
    assert df.index[0] == pd.Timestamp("2024-01-02")
    assert df.loc[pd.Timestamp("2024-01-02"), "Mom"] == 0.0125
