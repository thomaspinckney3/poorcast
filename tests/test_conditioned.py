"""Tests for valuation-conditioned block sampling and drift re-centering."""

import numpy as np
import pandas as pd
import pytest

from poorcast.simulate import SimConfig, simulate

IDX = pd.period_range("1960-01", periods=480, freq="M")


def two_regime_panel():
    # High-valuation months (first half) return +2%; low-valuation -1%.
    rets = np.where(np.arange(480) < 240, 0.02, -0.01)
    return pd.DataFrame({"a": rets, "inflation": np.zeros(480)}, index=IDX)


def state(levels):
    return pd.Series(levels, index=IDX)


def cfg(**kw):
    base = dict(allocation={"a": 1.0}, initial=1000.0, years=2, n_sims=16,
                seed=0, block_months=1)
    base.update(kw)
    return SimConfig(**base)


def test_conditioning_draws_from_matching_regime():
    s = state(np.where(np.arange(480) < 240, 30.0, 15.0))
    r = simulate(two_regime_panel(), cfg(
        state_series=s, state_path=np.full(24, 30.0), state_bandwidth=0.05))
    # Every sampled month comes from the high-valuation (+2%) regime.
    assert np.allclose(r.balance[:, -1], 1000.0 * 1.02**24)
    lo = simulate(two_regime_panel(), cfg(
        state_series=s, state_path=np.full(24, 15.0), state_bandwidth=0.05))
    assert np.allclose(lo.balance[:, -1], 1000.0 * 0.99**24)


def test_drift_recentering_replaces_conditional_drift():
    # Constant historical state (conditional drift 0); an assumed path
    # halving over 2y adds exactly its own drift to the adjusted asset.
    panel = pd.DataFrame({"a": np.zeros(480), "inflation": np.zeros(480)},
                         index=IDX)
    path = 20.0 * np.exp(np.linspace(0.0, np.log(0.5), 25))[:-1]
    r = simulate(panel, cfg(
        state_series=state(np.full(480, 20.0)), state_path=path,
        state_bandwidth=0.1, state_adjust_assets=("a",)))
    d = np.log(0.5) / 2.0  # annual drift of the path
    # 23 drifting months (the last month's forward step is zero by design)
    expected = 1000.0 * (1 + d * (23 / 24) / 12) ** 0  # placeholder, computed below
    steps = np.diff(np.log(path)) * 12.0
    expected = 1000.0 * np.prod(1 + np.concatenate([steps, [0.0]]) / 12.0)
    assert np.allclose(r.balance[:, -1], expected)


def test_conditioning_validation():
    panel = two_regime_panel()
    s = state(np.full(480, 20.0))
    with pytest.raises(ValueError, match="bootstrap"):
        simulate(panel, cfg(state_series=s, state_path=np.full(24, 20.0),
                            mode="historical"))
    with pytest.raises(ValueError, match="state_path"):
        simulate(panel, cfg(state_series=s))
    with pytest.raises(ValueError, match="length"):
        simulate(panel, cfg(state_series=s, state_path=np.full(7, 20.0)))
    with pytest.raises(ValueError, match="near the assumed"):
        simulate(panel, cfg(state_series=s, state_path=np.full(24, 500.0),
                            state_bandwidth=0.05))
