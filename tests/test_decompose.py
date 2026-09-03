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
