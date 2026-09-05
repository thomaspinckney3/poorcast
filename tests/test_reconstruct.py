"""Unit tests for the reconstruction anchoring math (no network needed)."""

import numpy as np
import pandas as pd

from poorcast.reconstruct import _anchor_to_eafe, _anchored_local_returns


def test_anchored_returns_compound_to_jst_annual():
    idx = pd.period_range("1979-12", "1981-12", freq="M")
    rng = np.random.default_rng(0)
    price = pd.Series(100 * np.cumprod(1 + rng.normal(0.01, 0.04, len(idx))), index=idx)
    jst = pd.DataFrame(
        {"year": [1980, 1981], "eq_tr": [0.25, -0.10], "eq_dp": [0.04, 0.04]}
    )
    out = _anchored_local_returns(price, jst)
    for year, target in [(1980, 0.25), (1981, -0.10)]:
        months = out[out.index.year == year]
        assert len(months) == 12
        assert np.isclose((1 + months).prod(), 1 + target)


def test_anchoring_preserves_within_year_shape():
    idx = pd.period_range("1979-12", "1980-12", freq="M")
    price = pd.Series(np.linspace(100, 150, len(idx)), index=idx)
    jst = pd.DataFrame({"year": [1980], "eq_tr": [0.30], "eq_dp": [0.0]})
    out = _anchored_local_returns(price, jst)
    raw = price.pct_change().dropna()
    # multiplicative scaling keeps the ordering of good and bad months
    assert (np.argsort(out.to_numpy()) == np.argsort(raw.to_numpy())).all()


def test_partial_year_prorated():
    idx = pd.period_range("1980-06", "1980-12", freq="M")  # 6 monthly returns
    price = pd.Series(100.0 + np.arange(len(idx)), index=idx)
    jst = pd.DataFrame({"year": [1980], "eq_tr": [0.20], "eq_dp": [0.0]})
    out = _anchored_local_returns(price, jst)
    assert len(out) == 6
    assert np.isclose((1 + out).prod(), 1.20 ** (6 / 12))


def test_eafe_anchor_pins_published_years():
    idx = pd.period_range("1970-01", "1970-12", freq="M")
    series = pd.Series(np.full(12, 0.02), index=idx)
    out = _anchor_to_eafe(series)
    assert np.isclose((1 + out).prod(), 1 - 0.105123)  # published EAFE 1970


def test_eafe_anchor_leaves_uncovered_years_alone():
    idx = pd.period_range("1965-01", "1965-12", freq="M")
    series = pd.Series(np.full(12, 0.02), index=idx)
    out = _anchor_to_eafe(series)
    assert np.allclose(out, series)


def test_parity_steps_are_devaluations_not_redenominations():
    # A currency redenomination (old franc -> new franc, 100:1) must be
    # carried in one unit; a genuine parity step is never more than ~30%.
    from poorcast.reconstruct import PARITIES

    for key, steps in PARITIES.items():
        rates = [r for _, r in steps]
        for a, b in zip(rates, rates[1:]):
            assert abs(b / a - 1) < 0.3, f"{key}: {a} -> {b}"
    assert PARITIES["france"][0][0] == "1955-01"


def test_reconstruction_starts_in_1955_by_default():
    import inspect
    from poorcast.reconstruct import reconstruct_intl

    assert inspect.signature(reconstruct_intl).parameters["first_year"].default == 1955
