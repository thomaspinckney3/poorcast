"""Unit tests for the simulation engine using a small synthetic panel."""

import numpy as np
import pandas as pd
import pytest

from poorcast.simulate import SimConfig, Withdrawal, simulate


def make_panel(n_months=480, ret_a=0.01, ret_b=0.005, inflation=0.002):
    """Deterministic panel: constant returns so outcomes are computable by hand."""
    idx = pd.period_range("1960-01", periods=n_months, freq="M")
    return pd.DataFrame(
        {
            "a": np.full(n_months, ret_a),
            "b": np.full(n_months, ret_b),
            "inflation": np.full(n_months, inflation),
        },
        index=idx,
    )


def cfg(**kw):
    base = dict(
        allocation={"a": 1.0},
        initial=1000.0,
        years=10,
        n_sims=8,
        seed=0,
        sample_start="1960-01",
    )
    base.update(kw)
    return SimConfig(**base)


def test_single_asset_compounds_exactly():
    panel = make_panel()
    r = simulate(panel, cfg())
    expected = 1000.0 * 1.01 ** 120
    assert np.allclose(r.balance[:, -1], expected)


def test_quarterly_rebalancing_restores_target_weights():
    panel = make_panel()
    c = cfg(allocation={"a": 0.5, "b": 0.5}, years=1)
    r = simulate(panel, c)
    # With constant returns, a 50/50 mix rebalanced quarterly compounds at the
    # blended rate within each quarter, then resets. Compute the same by hand.
    bal = 1000.0
    hold = np.array([500.0, 500.0])
    for m in range(12):
        hold = hold * np.array([1.01, 1.005])
        if (m + 1) % 3 == 0:
            hold = np.full(2, hold.sum() / 2)
    assert np.allclose(r.balance[:, -1], hold.sum())


def test_no_rebalancing_differs_from_rebalanced():
    idx = pd.period_range("1960-01", periods=480, freq="M")
    rng = np.random.default_rng(3)
    panel = pd.DataFrame(
        {
            "a": rng.normal(0.01, 0.05, 480),
            "b": rng.normal(0.002, 0.01, 480),
            "inflation": np.zeros(480),
        },
        index=idx,
    )
    base = dict(allocation={"a": 0.5, "b": 0.5}, years=10, n_sims=50, seed=1)
    quarterly = simulate(panel, cfg(**base))
    never = simulate(panel, cfg(**base, rebalance_months=10_000))
    assert not np.allclose(quarterly.balance[:, -1], never.balance[:, -1])


def test_fixed_real_withdrawal_reduces_balance_and_tracks_inflation():
    panel = make_panel(inflation=0.0)
    c = cfg(withdrawal=Withdrawal("fixed_real", rate=0.04))
    r = simulate(panel, c)
    no_wd = simulate(panel, cfg())
    assert (r.balance[:, -1] < no_wd.balance[:, -1]).all()
    # zero inflation: total withdrawn = rate * initial * years exactly
    assert np.allclose(r.total_withdrawn, 0.04 * 1000.0 * 10)

    inflated = simulate(make_panel(inflation=0.01), c)
    assert (inflated.total_withdrawn > r.total_withdrawn).all()


def test_depletion_detected_and_balance_stays_zero():
    panel = make_panel(ret_a=0.0, inflation=0.0)
    # withdraw 20%/yr of initial from a zero-return portfolio: dead in ~5 years
    c = cfg(withdrawal=Withdrawal("fixed_real", rate=0.20), years=10)
    r = simulate(panel, c)
    assert (r.depleted_month >= 0).all()
    assert r.success_rate == 0.0
    dead_at = r.depleted_month[0]
    assert abs(dead_at - 60) <= 1
    assert np.allclose(r.balance[:, dead_at + 1 :], 0.0)


def test_percent_of_balance_never_depletes():
    panel = make_panel(ret_a=0.0, inflation=0.0)
    c = cfg(withdrawal=Withdrawal("percent_of_balance", rate=0.5), years=20)
    r = simulate(panel, c)
    assert r.success_rate == 1.0
    assert (r.balance[:, -1] > 0).all()


def test_contribution_grows_balance():
    panel = make_panel(ret_a=0.0, inflation=0.0)
    c = cfg(contribution_monthly=100.0)
    r = simulate(panel, c)
    assert np.allclose(r.balance[:, -1], 1000.0 + 100.0 * 120)


def test_real_balance_deflates_by_inflation():
    panel = make_panel(ret_a=0.0, inflation=0.01)
    r = simulate(panel, cfg())
    assert np.allclose(r.balance[:, -1], 1000.0)
    assert np.allclose(r.real_balance[:, -1], 1000.0 / 1.01 ** 120)


def test_historical_mode_covers_every_window():
    panel = make_panel(n_months=132)
    c = cfg(mode="historical", years=10)
    r = simulate(panel, c)
    assert r.n_paths == 132 - 120 + 1
    assert (r.months[:, 0] == np.arange(13)).all()


def test_bootstrap_blocks_are_contiguous():
    panel = make_panel(n_months=200)
    c = cfg(block_months=24, years=2, n_sims=100)
    r = simulate(panel, c)
    diffs = np.diff(r.months, axis=1)
    contiguous = (diffs == 1) | (diffs == 1 - 200)  # circular wrap allowed
    # within each 24-month block all steps are contiguous
    assert contiguous[:, :23].all()


def test_bad_allocation_rejected():
    panel = make_panel()
    with pytest.raises(ValueError, match="sum"):
        simulate(panel, cfg(allocation={"a": 0.5, "b": 0.4}))
    with pytest.raises(ValueError, match="unknown asset"):
        simulate(panel, cfg(allocation={"nope": 1.0}))


def test_horizon_longer_than_history_rejected_in_historical_mode():
    panel = make_panel(n_months=130)
    with pytest.raises(ValueError, match="exceeds"):
        simulate(panel, cfg(mode="historical", years=20))


def test_seed_reproducible():
    panel = make_panel()
    idx = pd.period_range("1960-01", periods=480, freq="M")
    rng = np.random.default_rng(9)
    panel["a"] = rng.normal(0.008, 0.04, 480)
    r1 = simulate(panel, cfg(seed=123))
    r2 = simulate(panel, cfg(seed=123))
    assert np.array_equal(r1.balance, r2.balance)


def test_flex_matches_fixed_when_above_water():
    # positive returns, small withdrawal: balance never dips below initial
    panel = make_panel(ret_a=0.01, inflation=0.0)
    fixed = simulate(panel, cfg(withdrawal=Withdrawal("fixed_real", rate=0.03)))
    flex = simulate(
        panel, cfg(withdrawal=Withdrawal("fixed_real", rate=0.03, flex_floor=0.75))
    )
    assert np.allclose(fixed.balance, flex.balance)
    assert np.allclose(fixed.total_withdrawn, flex.total_withdrawn)


def test_flex_reduces_withdrawals_when_under_water():
    # negative drift puts the portfolio under water; flex should withdraw less
    panel = make_panel(ret_a=-0.005, inflation=0.0)
    fixed = simulate(panel, cfg(withdrawal=Withdrawal("fixed_real", rate=0.05)))
    flex = simulate(
        panel, cfg(withdrawal=Withdrawal("fixed_real", rate=0.05, flex_floor=0.75))
    )
    assert (flex.total_withdrawn < fixed.total_withdrawn).all()
    assert (flex.balance[:, -1] >= fixed.balance[:, -1]).all()
    # deeply under water, every month floors at exactly 75% of target
    target_monthly = 0.05 * 1000.0 / 12
    months_alive = 120
    assert (flex.total_withdrawn >= 0.75 * target_monthly * months_alive * 0.99).all()


def test_flex_floor_binds_exactly_when_deep_under_water():
    # zero returns, huge withdrawals: balance falls fast, floor binds
    panel = make_panel(ret_a=0.0, inflation=0.0)
    r = simulate(
        panel,
        cfg(withdrawal=Withdrawal("fixed_real", rate=0.12, flex_floor=0.75), years=5),
    )
    target_monthly = 0.12 * 1000.0 / 12  # 10/month
    # month 0: at initial, full withdrawal
    first_month_wd = 1000.0 - r.balance[0, 1]  # zero returns => pure withdrawal
    assert np.isclose(first_month_wd, target_monthly)
    # by year 5 the balance is below 75% of initial, so the floor binds exactly
    assert r.balance[0, -1] < 750
    late_wd = r.balance[0, -2] - r.balance[0, -1]
    assert np.isclose(late_wd, 0.75 * target_monthly)


def test_flex_tracks_real_balance_under_inflation():
    # nominal balance flat but inflation erodes real balance -> flex kicks in
    panel = make_panel(ret_a=0.0, inflation=0.01)
    flex = simulate(
        panel, cfg(withdrawal=Withdrawal("fixed_real", rate=0.04, flex_floor=0.75))
    )
    fixed = simulate(panel, cfg(withdrawal=Withdrawal("fixed_real", rate=0.04)))
    assert (flex.total_withdrawn < fixed.total_withdrawn).all()


def test_flex_rejected_for_percent_of_balance():
    panel = make_panel()
    with pytest.raises(ValueError, match="flex_floor only applies"):
        simulate(
            panel,
            cfg(withdrawal=Withdrawal("percent_of_balance", rate=0.03, flex_floor=0.75)),
        )
    with pytest.raises(ValueError, match="flex_floor must be"):
        simulate(
            panel, cfg(withdrawal=Withdrawal("fixed_real", rate=0.03, flex_floor=1.5))
        )


def test_failure_year_stats():
    from poorcast.report import failure_year_stats

    # 20%/yr from a zero-return portfolio: the 60th withdrawal (month index 59,
    # the last month of year 5) empties it
    panel = make_panel(ret_a=0.0, inflation=0.0)
    r = simulate(panel, cfg(withdrawal=Withdrawal("fixed_real", rate=0.20), years=10))
    f = failure_year_stats(r)
    assert f["share"] == 1.0
    assert f["earliest"] == f["median"] == 5

    # no failures -> None
    ok = simulate(panel, cfg(withdrawal=Withdrawal("fixed_real", rate=0.01), years=10))
    assert failure_year_stats(ok) is None


def test_glidepath_equals_static_when_end_matches_start():
    panel = make_panel()
    static = simulate(panel, cfg(allocation={"a": 0.5, "b": 0.5}))
    glide = simulate(
        panel,
        cfg(
            allocation={"a": 0.5, "b": 0.5},
            allocation_end={"a": 0.5, "b": 0.5},
            glide_years=5,
        ),
    )
    assert np.allclose(static.balance, glide.balance)


def test_glidepath_ends_at_target_mix():
    # constant unequal returns make the mix observable: after the glide completes,
    # quarterly rebalancing should track the end weights, so the last quarter's
    # growth matches a static end-mix portfolio's quarter
    panel = make_panel(ret_a=0.02, ret_b=0.0, inflation=0.0)
    glide = simulate(
        panel,
        cfg(
            allocation={"a": 0.2, "b": 0.8},
            allocation_end={"a": 0.8, "b": 0.2},
            glide_years=2,
            years=10,
        ),
    )
    static_end = simulate(panel, cfg(allocation={"a": 0.8, "b": 0.2}, years=10))
    g = glide.balance[0, -1] / glide.balance[0, -4]  # last quarter growth
    s = static_end.balance[0, -1] / static_end.balance[0, -4]
    assert np.isclose(g, s, rtol=1e-9)


def test_glidepath_between_start_and_end_outcomes():
    panel = make_panel(ret_a=0.01, ret_b=0.002, inflation=0.0)
    lo = simulate(panel, cfg(allocation={"a": 0.2, "b": 0.8}))
    hi = simulate(panel, cfg(allocation={"a": 0.8, "b": 0.2}))
    glide = simulate(
        panel,
        cfg(allocation={"a": 0.2, "b": 0.8}, allocation_end={"a": 0.8, "b": 0.2}),
    )
    assert (lo.balance[0, -1] < glide.balance[0, -1] < hi.balance[0, -1])


def test_glidepath_asset_mismatch_rejected():
    panel = make_panel()
    with pytest.raises(ValueError, match="same assets"):
        simulate(
            panel,
            cfg(allocation={"a": 1.0}, allocation_end={"b": 1.0}),
        )


def test_allocation_rule_overrides_targets():
    # a rule that always returns 100% asset b should track a static-b portfolio
    # from the first rebalance on
    panel = make_panel(ret_a=0.02, ret_b=0.005, inflation=0.0)

    def all_b(state):
        return np.tile([0.0, 1.0], (len(state.balance), 1))

    ruled = simulate(
        panel, cfg(allocation={"a": 0.5, "b": 0.5}, allocation_rule=all_b, years=5)
    )
    static_b = simulate(panel, cfg(allocation={"a": 0.0, "b": 1.0}, years=5))
    # identical growth after the first rebalance settles into 100% b
    g_ruled = ruled.balance[0, -1] / ruled.balance[0, 3]
    g_static = static_b.balance[0, -1] / static_b.balance[0, 3]
    assert np.isclose(g_ruled, g_static)


def test_rule_state_reports_real_balance():
    panel = make_panel(ret_a=0.0, inflation=0.01)
    seen = {}

    def spy(state):
        seen["real"] = state.real_balance.copy()
        seen["nominal"] = state.balance.copy()
        return np.tile([1.0], (len(state.balance), 1))

    simulate(panel, cfg(allocation={"a": 1.0}, allocation_rule=spy, years=1))
    assert (seen["real"] < seen["nominal"]).all()


def test_sequence_risk_stats():
    from poorcast.report import sequence_risk_stats

    idx = pd.period_range("1960-01", periods=480, freq="M")
    rng = np.random.default_rng(11)
    panel = pd.DataFrame(
        {"a": rng.normal(0.004, 0.05, 480), "inflation": np.zeros(480)}, index=idx)
    r = simulate(panel, cfg(allocation={"a": 1.0}, n_sims=2000, seed=3,
                            withdrawal=Withdrawal("fixed_real", rate=0.07), years=20))
    s = sequence_risk_stats(r)
    assert s is not None and 0 < s["terminal_r2"] < 1
    # a bad start must raise failure odds above the base rate
    assert s["fail_rate_bad_start"] > s["fail_rate_base"] > 0
    assert 0 < s["failures_with_bad_start"] <= 1

    # degenerate case: constant returns -> no variance -> None
    flat = simulate(make_panel(ret_a=0.005, inflation=0.0),
                    cfg(allocation={"a": 1.0}, n_sims=300))
    assert sequence_risk_stats(flat) is None


def test_return_adjustment_exact():
    panel = make_panel(ret_a=0.01, ret_b=0.005, inflation=0.0)
    plain = simulate(panel, cfg(allocation={"a": 1.0}))
    adj = simulate(panel, cfg(allocation={"a": 1.0},
                              return_adjustments={"a": -0.012}))
    assert np.allclose(adj.balance[:, -1], 1000.0 * (1 + 0.01 - 0.001) ** 120)
    assert (adj.balance[:, -1] < plain.balance[:, -1]).all()
    # adjustment for an asset not in the portfolio is ignored
    noop = simulate(panel, cfg(allocation={"a": 1.0},
                               return_adjustments={"b": -0.5}))
    assert np.allclose(noop.balance, plain.balance)


def test_fee_is_exact_monthly_drag():
    panel = make_panel(ret_a=0.0, inflation=0.0)
    r = simulate(panel, cfg(fee_annual=0.012))
    assert np.allclose(r.balance[:, -1], 1000.0 * (1 - 0.001) ** 120)


def test_fee_combines_with_return_adjustments():
    panel = make_panel(ret_a=0.0, inflation=0.0)
    r = simulate(
        panel, cfg(fee_annual=0.012, return_adjustments={"a": 0.012})
    )
    assert np.allclose(r.balance[:, -1], 1000.0)  # drag and boost cancel


def test_fee_rejects_implausible_values():
    panel = make_panel()
    with pytest.raises(ValueError, match="fee_annual"):
        simulate(panel, cfg(fee_annual=0.5))


def test_per_month_return_adjustment_path():
    panel = make_panel(ret_a=0.0, inflation=0.0)
    adj = np.concatenate([np.full(12, 0.12), np.zeros(108)])  # +1%/mo year 1
    r = simulate(panel, cfg(return_adjustments={"a": adj}))
    assert np.allclose(r.balance[:, 12], 1000.0 * 1.01**12)
    assert np.allclose(r.balance[:, -1], 1000.0 * 1.01**12)
    with pytest.raises(ValueError, match="length"):
        simulate(panel, cfg(return_adjustments={"a": np.zeros(7)}))
