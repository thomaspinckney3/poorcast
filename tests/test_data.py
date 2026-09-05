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
