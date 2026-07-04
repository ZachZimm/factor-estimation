from __future__ import annotations

import io
import zipfile

import pandas as pd

from ff5_predictor.data_famafrench import parse_kenneth_french_ff5_zip, parse_kenneth_french_momentum_zip


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
