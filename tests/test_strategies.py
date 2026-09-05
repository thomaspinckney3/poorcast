"""Tests for the shipped allocation rules."""

import numpy as np
import pandas as pd

from poorcast.simulate import SimConfig, Withdrawal, simulate
from poorcast.strategies import ASSETS, ratchet_rule


def make_panel(n_months=240):
    idx = pd.period_range("1960-01", periods=n_months, freq="M")
    rng = np.random.default_rng(3)
    cols = {a: rng.normal(0.008, 0.04, n_months) for a in ASSETS}
    cols["inflation"] = np.zeros(n_months)
    return pd.DataFrame(cols, index=idx)


def test_ratchet_rule_forgets_the_previous_run():
    panel = make_panel()
    rule = ratchet_rule(threshold=1.2)
    alloc = {a: w for a, w in zip(ASSETS, [0.5, 0.2, 0.25, 0.05])}
    base = dict(allocation=alloc, allocation_rule=rule, years=10, n_sims=50,
                seed=1, account="roth", withdrawal=Withdrawal("fixed_real", rate=0.03))
    first = simulate(panel, SimConfig(**base))
    again = simulate(panel, SimConfig(**base))  # same rule object, same n_paths
    fresh = simulate(panel, SimConfig(**{**base, "allocation_rule": ratchet_rule(1.2)}))
    assert np.allclose(first.balance[:, -1], fresh.balance[:, -1])
    assert np.allclose(first.balance[:, -1], again.balance[:, -1])
