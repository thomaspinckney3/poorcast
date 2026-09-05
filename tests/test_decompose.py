"""Decomposition test - uses cached data, skipped if the cache is absent."""

import pytest

from poorcast.data import CACHE_DIR


@pytest.mark.skipif(
    not (CACHE_DIR / "shiller_ie_data.xls").exists()
    or not (CACHE_DIR / "fred_CP.csv").exists(),
    reason="needs cached Shiller/FRED data (run 'poorcast fetch')",
)
def test_decomposition_components_sum_to_total():
    from poorcast.decompose import equity_return_decomposition

    d = equity_return_decomposition()
    assert 0.06 < d["actual_total_return"] < 0.14
    assert abs(d["residual"]) < 0.015  # components explain the total closely
    assert d["margin_expansion"] > 0  # margins rose over the sample
    assert -0.01 < d["multiple_expansion"] < 0.02
    total = (d["dividend_yield"] + d["inflation"] + d["real_eps_growth"]
             + d["multiple_expansion"])
    assert abs(total - d["sum_of_components"]) < 1e-12


def test_cape_from_pde_is_ten_year_real_pe():
    import numpy as np
    import pandas as pd
    from poorcast.decompose import cape_from_pde

    idx = pd.period_range("2000-01", periods=132, freq="M")
    sh = pd.DataFrame({"P": np.full(132, 200.0), "E": np.full(132, 10.0),
                       "CPI": np.full(132, 100.0)}, index=idx)
    s = cape_from_pde(sh)
    assert len(s) == 13  # needs 120 months of earnings
    assert np.allclose(s, 20.0)
