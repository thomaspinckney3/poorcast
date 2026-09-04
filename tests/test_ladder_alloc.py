"""Tests for TIPS-ladder-as-allocation (the reserved `tips_ladder` asset).

Zero-yield ladders keep the arithmetic hand-computable: an N-year ladder at
0% real costs N x annual and pays annual/yr with no coupons.
"""

import numpy as np
import pandas as pd
import pytest

from poorcast.simulate import Account, SimConfig, Withdrawal, simulate


def make_panel(n_months=480):
    idx = pd.period_range("1960-01", periods=n_months, freq="M")
    return pd.DataFrame(
        {
            "a": np.zeros(n_months),
            "inflation": np.zeros(n_months),
            "income_a": np.zeros(n_months),
        },
        index=idx,
    )


def cfg(**kw):
    base = dict(
        allocation={"a": 1.0}, initial=1000.0, years=2, n_sims=4, seed=0,
        ladder_yield=0.0,
    )
    base.update(kw)
    return SimConfig(**base)


def test_ladder_allocation_conserves_value():
    # 20% buys a 2y ladder (cost 200 -> 100/yr); with no spending the rung
    # income is reinvested, so total wealth stays exactly 1000 at year ends.
    r = simulate(make_panel(), cfg(allocation={"a": 0.8, "tips_ladder": 0.2}))
    assert np.allclose(r.balance[:, 0], 1000.0)
    assert np.allclose(r.balance[:, 12], 1000.0)  # liquid 900 + rung 100
    assert np.allclose(r.balance[:, -1], 1000.0)
    assert r.ladder_annual == pytest.approx(100.0)


def test_ladder_income_offsets_withdrawal():
    r = simulate(make_panel(), cfg(
        allocation={"a": 0.8, "tips_ladder": 0.2},
        withdrawal=Withdrawal("fixed_real", amount=100.0),
    ))
    # The 100/yr of rung income funds the withdrawal; liquid assets untouched.
    assert np.allclose(r.balance[:, -1], 800.0)
    assert np.allclose(r.total_withdrawn, 0.0)  # nothing drawn from holdings


def test_traditional_held_rungs_are_taxed_distributions():
    panel = make_panel()
    c = cfg(
        age=60,
        accounts=(
            Account("taxable", 1000.0, allocation={"a": 1.0}),
            Account("traditional", 200.0, allocation={"tips_ladder": 1.0}),
        ),
        tax_ordinary=0.25,
        ladder_years=2,
    )
    r = simulate(panel, c)
    # 100/yr of IRA payouts: reinvested in taxable, taxed 25/yr as ordinary
    # income (collected from taxable - the IRA holds no liquid assets).
    assert np.allclose(r.balance[:, 0], 1200.0)
    assert np.allclose(r.account_terminal[:, 0], 1000.0 + 200.0 - 50.0)
    assert np.allclose(r.account_terminal[:, 1], 0.0)
    assert np.allclose(r.total_tax_real, 50.0)


def test_roth_ladder_early_payouts_penalized_beyond_basis():
    panel = make_panel()
    c = cfg(
        years=1,
        age=50,
        accounts=(Account("roth", 240.0, cost_basis=0.0,
                          allocation={"tips_ladder": 1.0}),),
        ladder_years=1,
    )
    r = simulate(panel, c)
    # 240/yr of payouts, all earnings, before 59.5: 10% penalty = 24, paid
    # from the reinvested income at year end.
    assert np.allclose(r.total_tax_real, 24.0)
    assert np.allclose(r.balance[:, -1], 240.0 - 24.0)


def test_ladder_allocation_validation():
    panel = make_panel()
    with pytest.raises(ValueError, match="not both"):
        from poorcast.ladder import build_ladder

        simulate(panel, cfg(
            allocation={"a": 0.8, "tips_ladder": 0.2},
            ladder=build_ladder(10.0, 2, 0.01),
        ))
    with pytest.raises(ValueError, match="529"):
        simulate(panel, cfg(
            accounts=(Account("529", 100.0, allocation={"tips_ladder": 1.0}),),
        ))
    with pytest.raises(ValueError, match="non-ladder"):
        simulate(panel, cfg(allocation={"tips_ladder": 1.0}))


# --- audit fixes -------------------------------------------------------------


def test_single_account_all_ladder_conserves():
    # Rebalancing must not zero an all-ladder portfolio's reinvested income.
    r = simulate(make_panel(), cfg(
        allocation={"a": 0.0, "tips_ladder": 1.0}, ladder_years=2))
    assert np.allclose(r.balance[:, -1], 1000.0)
    assert (r.depleted_month == -1).all()


def test_renormalization_does_not_leak_value():
    # A weight sum inside the 1e-6 tolerance must not compound into a loss
    # when renormalized over a tiny liquid sleeve.
    r = simulate(make_panel(), cfg(
        allocation={"a": 0.0009995, "tips_ladder": 0.999},
        years=30, rebalance_months=1, ladder_years=30,
    ))
    assert np.allclose(r.balance[:, -1], 1000.0, rtol=1e-5)


def test_rmd_recognized_on_rung_value():
    # Age 80, all-ladder IRA: payouts 1010/yr fall short of the RMD computed
    # on the rungs' full value; the shortfall must still be recognized as
    # taxable income (in-kind distribution), though nothing can transfer.
    # Year 1: RMD 30300/20.2 = 1500 -> tax 375. Year 2: 29290/19.4 = 1509.79
    # -> tax 377.45.
    panel = make_panel()
    c = cfg(
        years=2,
        age=80,
        accounts=(
            Account("taxable", 10_000.0, allocation={"a": 1.0}),
            Account("traditional", 30_300.0, allocation={"tips_ladder": 1.0}),
        ),
        tax_ordinary=0.25,
        ladder_years=30,
    )
    r = simulate(panel, c)
    assert np.allclose(r.total_tax_real, 752.448, atol=0.01)


def test_zero_liquid_with_covering_income_is_not_depletion():
    # The expense drains the liquid 20 at month 0; rung income (490/yr at
    # 0% on a 980-cost 2y ladder) exactly covers the 490/yr target from then
    # on. Every dollar delivered, terminal $0 -> still a success.
    r = simulate(make_panel(), cfg(
        allocation={"a": 0.02, "tips_ladder": 0.98}, ladder_years=2,
        withdrawal=Withdrawal("fixed_real", amount=490.0),
        expenses=((0, 20.0),),
    ))
    assert (r.depleted_month == -1).all()
    assert np.allclose(r.total_unmet_real, 0.0)


def test_unmet_spending_is_tracked():
    # Spending target exceeds income + liquid: the shortfall is recorded.
    r = simulate(make_panel(), cfg(
        years=1,
        allocation={"a": 0.05, "tips_ladder": 0.95}, ladder_years=1,
        withdrawal=Withdrawal("fixed_real", amount=1200.0),
    ))
    # Deliverable: 50 liquid + 950 income = 1000; target 1200 -> 200 unmet.
    assert np.allclose(r.total_unmet_real, 200.0)
    assert (r.depleted_month >= 0).all()


def test_ladder_years_validated():
    with pytest.raises(ValueError, match="ladder_years"):
        simulate(make_panel(), cfg(
            allocation={"a": 0.5, "tips_ladder": 0.5}, ladder_years=0))
