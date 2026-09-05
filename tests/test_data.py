"""Tests for the data layer's parsers (offline, synthetic inputs)."""

import numpy as np
import pandas as pd

from poorcast import data


def test_shiller_month_index_keeps_october():
    # Excel returns the fractional-year date column as floats: 1871.10 -> 1871.1
    col = pd.Series([np.nan, "Date", 1871.01, 1871.09, 1871.1, 1871.11, 1871.12,
                     2023.09, "notes"])
    rows, idx = data.shiller_month_index(col)
    assert rows.tolist() == [False, False, True, True, True, True, True, True, False]
    assert list(idx.astype(str)) == [
        "1871-01", "1871-09", "1871-10", "1871-11", "1871-12", "2023-09"
    ]


def test_custom_override_replaces_and_extends_a_series(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "CUSTOM_DIR", tmp_path)
    (tmp_path / "cash.csv").write_text("month,return\n1960-02,0.5\n1959-12,0.25\n")
    idx = pd.period_range("1960-01", periods=3, freq="M")
    built = pd.Series([0.01, 0.02, 0.03], index=idx, name="cash")
    out = data._apply_custom_override("cash", built)
    assert out.name == "cash"
    assert out[pd.Period("1960-02", "M")] == 0.5      # replaced
    assert out[pd.Period("1959-12", "M")] == 0.25     # extended earlier
    assert out[pd.Period("1960-01", "M")] == 0.01     # untouched
    # no file -> unchanged
    assert data._apply_custom_override("us_equities", built).equals(built)
