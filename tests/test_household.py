"""Tests for multi-account (household) simulation: waterfall withdrawals,
joint tax settlement, RMD transfers, per-account allocations."""

import numpy as np
import pandas as pd
import pytest

from poorcast.simulate import Account, IncomeStream, SimConfig, Withdrawal, simulate


def make_panel(n_months=480, ret_a=0.0, ret_b=0.0, income_a=0.0):
    idx = pd.period_range("1960-01", periods=n_months, freq="M")
    return pd.DataFrame(
        {
            "a": np.full(n_months, ret_a),
            "b": np.full(n_months, ret_b),
            "inflation": np.zeros(n_months),
            "income_a": np.full(n_months, income_a),
            "income_b": np.zeros(n_months),
        },
        index=idx,
    )


def cfg(**kw):
    base = dict(allocation={"a": 1.0}, years=1, n_sims=4, seed=0)
    base.update(kw)
    return SimConfig(**base)


def test_single_account_list_matches_legacy_exactly():
    idx = pd.period_range("1960-01", periods=480, freq="M")
    rng = np.random.default_rng(11)
    panel = pd.DataFrame(
        {"a": rng.normal(0.008, 0.04, 480), "b": np.full(480, 0.002),
         "inflation": np.full(480, 0.002),
         "income_a": np.full(480, 0.002), "income_b": np.zeros(480)}, index=idx)
    base = dict(years=10, n_sims=50, seed=1, tax_brackets="single",
                state_tax=0.05, withdrawal=Withdrawal("fixed_real", rate=0.04))
    legacy = simulate(panel, cfg(
        allocation={"a": 0.6, "b": 0.4}, initial=1e6, cost_basis_start=0.7, **base))
    multi = simulate(panel, cfg(
        allocation={"a": 0.6, "b": 0.4},
        accounts=(Account("taxable", 1e6, cost_basis=0.7),), **base))
    assert np.array_equal(legacy.balance, multi.balance)
    assert np.array_equal(legacy.total_tax_real, multi.total_tax_real)


def test_waterfall_drains_taxable_before_roth():
    panel = make_panel()
    c = cfg(
        accounts=(Account("taxable", 100.0), Account("roth", 1000.0)),
        withdrawal=Withdrawal("fixed_real", amount=240.0),
    )
    r = simulate(panel, c)
    # 20/month: taxable covers months 0-4, roth pays from month 5.
    assert np.allclose(r.balance[:, 5], 1000.0)
    assert np.allclose(r.balance[:, -1], 1100.0 - 240.0)
    assert np.allclose(r.account_terminal[:, 0], 0.0)
    assert np.allclose(r.account_terminal[:, 1], 860.0)
    assert r.account_kinds == ("taxable", "roth")


def test_withdraw_order_override():
    panel = make_panel()
    c = cfg(
        accounts=(Account("taxable", 1000.0), Account("roth", 1000.0)),
        withdraw_order=("roth", "taxable"),
        withdrawal=Withdrawal("fixed_real", amount=240.0),
    )
    r = simulate(panel, c)
    assert np.allclose(r.account_terminal[:, 0], 1000.0)  # taxable untouched
    assert np.allclose(r.account_terminal[:, 1], 760.0)


def test_joint_bracket_settlement_stacks_income():
    # Traditional draws first (order override); the taxable account's
    # dividends stack on top in ONE annual settlement with ONE deduction:
    # ordinary 24,000 - 16,100 = 7,900 -> tax 790; 12,000 of dividends stack
    # into the 0% LTCG bracket -> 0. Paid from the taxable account.
    panel = make_panel(income_a=0.001)
    c = cfg(
        accounts=(
            Account("taxable", 1_000_000.0, allocation={"a": 1.0}),
            Account("traditional", 100_000.0, allocation={"b": 1.0}),
        ),
        withdraw_order=("traditional", "taxable"),
        age=60,
        tax_brackets="single",
        withdrawal=Withdrawal("fixed_real", amount=24_000.0),
    )
    r = simulate(panel, c)
    assert np.allclose(r.total_tax_real, 790.0)
    assert np.allclose(r.account_terminal[:, 0], 1_000_000.0 - 790.0)
    assert np.allclose(r.account_terminal[:, 1], 100_000.0 - 24_000.0)


def test_rmd_shortfall_transfers_to_taxable():
    # Age 75, no spending: RMD = 1000/24.6 must move from the IRA to the
    # taxable account (basis = value) and be taxed at the flat ordinary rate,
    # paid from the IRA.
    panel = make_panel()
    c = cfg(
        accounts=(Account("taxable", 1000.0), Account("traditional", 1000.0)),
        age=75,
        tax_ordinary=0.20,
    )
    r = simulate(panel, c)
    rmd = 1000.0 / 24.6
    assert np.allclose(r.account_terminal[:, 0], 1000.0 + rmd)
    assert np.allclose(r.account_terminal[:, 1], 1000.0 - rmd - 0.20 * rmd)
    assert np.allclose(r.total_tax_real, 0.20 * rmd)


def test_contributions_land_in_taxable():
    panel = make_panel()
    c = cfg(
        accounts=(Account("roth", 500.0), Account("taxable", 500.0)),
        contribution_monthly=10.0,
    )
    r = simulate(panel, c)
    assert np.allclose(r.account_terminal[:, 0], 500.0)
    assert np.allclose(r.account_terminal[:, 1], 500.0 + 120.0)


def test_per_account_allocations_use_own_assets():
    panel = make_panel(ret_a=0.01, ret_b=0.0)
    c = cfg(
        accounts=(
            Account("taxable", 1000.0, allocation={"a": 1.0}),
            Account("roth", 1000.0, allocation={"b": 1.0}),
        ),
    )
    r = simulate(panel, c)
    assert np.allclose(r.account_terminal[:, 0], 1000.0 * 1.01**12)
    assert np.allclose(r.account_terminal[:, 1], 1000.0)


def test_income_offsets_before_waterfall():
    panel = make_panel()
    c = cfg(
        accounts=(Account("taxable", 1000.0), Account("roth", 1000.0)),
        withdrawal=Withdrawal("fixed_real", amount=240.0),
        income=(IncomeStream(120.0),),
    )
    r = simulate(panel, c)
    assert np.allclose(r.account_terminal[:, 0], 1000.0 - 120.0)  # net draw only
    assert np.allclose(r.account_terminal[:, 1], 1000.0)


def test_household_depletion_needs_all_accounts_empty():
    panel = make_panel()
    c = cfg(
        accounts=(Account("taxable", 60.0), Account("roth", 60.0)),
        withdrawal=Withdrawal("fixed_real", amount=120.0),
    )
    r = simulate(panel, c)
    assert (r.depleted_month == 11).all()  # 10/mo, dies in month 12


def test_validation_rejects_bad_household():
    panel = make_panel()
    with pytest.raises(ValueError, match="at most one taxable"):
        simulate(panel, cfg(accounts=(Account("taxable", 1.0), Account("taxable", 1.0))))
    with pytest.raises(ValueError, match="withdraw_order"):
        simulate(panel, cfg(accounts=(Account("roth", 1.0),),
                            withdraw_order=("taxable",)))
    with pytest.raises(ValueError, match="single-account"):
        simulate(panel, cfg(accounts=(Account("roth", 1.0),),
                            allocation_end={"a": 1.0}))
