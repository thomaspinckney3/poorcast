"""Tests for capital-gains and income tax modeling."""

import numpy as np
import pandas as pd
import pytest

from poorcast.simulate import SimConfig, Withdrawal, simulate


def panel_with_income(returns_a, returns_b=None, income_a=0.0, name_a="a"):
    n = len(returns_a)
    idx = pd.period_range("1960-01", periods=n, freq="M")
    data = {name_a: returns_a, "inflation": np.zeros(n), f"income_{name_a}": np.full(n, income_a)}
    if returns_b is not None:
        data["b"] = returns_b
        data["income_b"] = np.zeros(n)
    return pd.DataFrame(data, index=idx)


def test_zero_tax_is_noop():
    idx = pd.period_range("1960-01", periods=480, freq="M")
    rng = np.random.default_rng(5)
    panel = pd.DataFrame(
        {"a": rng.normal(0.008, 0.04, 480), "b": np.full(480, 0.003),
         "inflation": np.zeros(480)}, index=idx)
    base = dict(allocation={"a": 0.5, "b": 0.5}, initial=1000.0, years=10,
                n_sims=20, seed=0, withdrawal=Withdrawal("fixed_real", rate=0.04))
    r0 = simulate(panel, SimConfig(**base))
    r1 = simulate(panel, SimConfig(**base, tax_rate=0.0))
    assert np.array_equal(r0.balance, r1.balance)


def test_rebalance_gain_taxed_hand_computed():
    # 50/50, a +10% in month 1 only: Q1 rebalance sells 25 of a with basis
    # fraction 500/550 -> realized gain 25*(1-500/550) = 2.2727, tax 20% = 0.4545
    rets = np.zeros(240)
    rets[0] = 0.10
    panel = panel_with_income(rets, returns_b=np.zeros(240))
    cfg = SimConfig(allocation={"a": 0.5, "b": 0.5}, initial=1000.0, years=1,
                    n_sims=2, seed=0, mode="historical", tax_rate=0.20)
    r = simulate(panel, cfg)
    expected_tax = 0.20 * 25 * (1 - 500 / 550)
    assert abs(r.balance[0, -1] - (1050 - expected_tax)) < 0.01
    assert abs(r.total_tax_real[0] - expected_tax) < 0.01


def test_income_tax_exact_geometric_drag():
    # flat prices, 1%/mo dividend income, 20% tax: each quarter pays exactly
    # 0.6% of balance -> terminal = 1000 * 0.994^40 over 10 years
    panel = panel_with_income(np.zeros(480), income_a=0.01)
    cfg = SimConfig(allocation={"a": 1.0}, initial=1000.0, years=10, n_sims=2,
                    seed=0, tax_rate=0.20)
    r = simulate(panel, cfg)
    assert np.allclose(r.balance[:, -1], 1000 * (1 - 0.006) ** 40, rtol=1e-9)


def test_lower_starting_basis_means_more_tax():
    rets = np.full(480, 0.006)
    panel = panel_with_income(rets, returns_b=np.zeros(480))
    base = dict(allocation={"a": 0.5, "b": 0.5}, initial=1000.0, years=10,
                n_sims=2, seed=0, tax_rate=0.20,
                withdrawal=Withdrawal("fixed_real", rate=0.04))
    fresh = simulate(panel, SimConfig(**base, cost_basis_start=1.0))
    appreciated = simulate(panel, SimConfig(**base, cost_basis_start=0.5))
    assert (appreciated.total_tax_real > fresh.total_tax_real).all()
    assert (appreciated.balance[:, -1] < fresh.balance[:, -1]).all()


def test_interest_taxed_at_ordinary_rate():
    # asset named us_bonds_10yr is interest-classed: taxed even with tax_rate=0
    panel = panel_with_income(np.zeros(480), income_a=0.005, name_a="us_bonds_10yr")
    cfg = SimConfig(allocation={"us_bonds_10yr": 1.0}, initial=1000.0, years=10,
                    n_sims=2, seed=0, tax_rate=0.0, tax_ordinary=0.40)
    r = simulate(panel, cfg)
    assert (r.total_tax_real > 0).all()
    assert np.allclose(r.balance[:, -1], 1000 * (1 - 3 * 0.005 * 0.40) ** 40, rtol=1e-9)


def test_missing_income_columns_rejected_when_taxed():
    idx = pd.period_range("1960-01", periods=480, freq="M")
    panel = pd.DataFrame({"a": np.full(480, 0.005), "inflation": np.zeros(480)}, index=idx)
    with pytest.raises(ValueError, match="income columns"):
        simulate(panel, SimConfig(allocation={"a": 1.0}, initial=1000.0, years=10,
                                  n_sims=2, seed=0, tax_rate=0.15))


def test_withdrawal_covered_by_income_realizes_no_gains():
    # interest-classed asset, ordinary rate 0, gains rate 20%: if withdrawals
    # never exceed the income stream, no sales are needed -> zero tax despite
    # constant price appreciation
    rets = np.full(480, 0.01)
    panel = panel_with_income(rets, income_a=0.005, name_a="us_bonds_10yr")
    base = dict(allocation={"us_bonds_10yr": 1.0}, initial=1000.0, years=10,
                n_sims=2, seed=0, tax_rate=0.20, tax_ordinary=0.0)
    covered = simulate(panel, SimConfig(
        **base, withdrawal=Withdrawal("fixed_real", amount=48.0)))  # $4/mo < $5/mo income
    assert np.allclose(covered.total_tax_real, 0.0)

    # withdrawals above the income stream must sell and realize gains
    uncovered = simulate(panel, SimConfig(
        **base, withdrawal=Withdrawal("fixed_real", amount=120.0)))  # $10/mo > income
    assert (uncovered.total_tax_real > 0).all()


def test_income_first_spending_reduces_tax_vs_gap():
    # sanity: larger income gap -> more realized gains -> more tax
    rets = np.full(480, 0.008)
    panel = panel_with_income(rets, income_a=0.004, name_a="us_bonds_10yr")
    base = dict(allocation={"us_bonds_10yr": 1.0}, initial=1000.0, years=10,
                n_sims=2, seed=0, tax_rate=0.20, tax_ordinary=0.0)
    small_gap = simulate(panel, SimConfig(
        **base, withdrawal=Withdrawal("fixed_real", amount=60.0)))
    big_gap = simulate(panel, SimConfig(
        **base, withdrawal=Withdrawal("fixed_real", amount=100.0)))
    assert (big_gap.total_tax_real > small_gap.total_tax_real).all()


def test_spent_income_is_still_taxed():
    # flat prices, 0.5%/mo interest fully consumed by withdrawals: no capital
    # gains exist anywhere, yet income tax is owed on every dollar received
    panel = panel_with_income(np.zeros(480), income_a=0.005, name_a="us_bonds_10yr")
    cfg = SimConfig(allocation={"us_bonds_10yr": 1.0}, initial=1000.0, years=10,
                    n_sims=2, seed=0, tax_rate=0.0, tax_ordinary=0.40,
                    withdrawal=Withdrawal("fixed_real", amount=60.0))  # $5/mo = income
    r = simulate(panel, cfg)
    assert (r.total_tax_real > 0).all()
    # ~40% of ~$5/mo income over 120 months on a shrinking base (withdrawals
    # and the tax itself drain it); the constant-base ceiling would be $240
    assert 120 < r.total_tax_real[0] < 240
