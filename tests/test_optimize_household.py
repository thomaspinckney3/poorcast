"""Tests for household-mode optimization (equity share x ladder total)."""

import numpy as np
import pandas as pd
import pytest

from poorcast.optimize import household_candidate, optimize_household
from poorcast.simulate import Account, SimConfig, Withdrawal

BASE = (
    Account("taxable", 12_000_000.0, allocation={
        "us_equities": 0.525, "muni_bonds": 0.30625, "cash": 0.04375,
        "tips_ladder": 0.125}),
    Account("traditional", 500_000.0, allocation={"tips_ladder": 1.0}),
    Account("roth", 500_000.0, allocation={
        "us_equities": 0.6, "us_bonds_10yr": 0.35, "cash": 0.05}),
    Account("529", 1_000_000.0, allocation={
        "us_equities": 0.6, "us_bonds_10yr": 0.35, "cash": 0.05}),
)


def test_candidate_rescales_and_places_ladder():
    cand = household_candidate(BASE, None, equity=0.5, ladder_total=1_000_000.0)
    tax, trad, roth, plan529 = cand
    # Ladder: traditional filled first (500k), remainder 500k -> taxable
    # (wl = 500k/12M); liquid rescaled to 50/50 with muni:cash = 87.5:12.5.
    wl = 500_000 / 12_000_000
    liq = 1 - wl
    assert tax.allocation["tips_ladder"] == pytest.approx(wl)
    assert tax.allocation["us_equities"] == pytest.approx(liq * 0.5)
    assert tax.allocation["muni_bonds"] == pytest.approx(liq * 0.5 * 0.875)
    assert tax.allocation["cash"] == pytest.approx(liq * 0.5 * 0.125)
    assert trad.allocation["tips_ladder"] == 1.0
    assert sum(v for k, v in trad.allocation.items() if k != "tips_ladder") == 0.0
    assert roth.allocation["us_equities"] == pytest.approx(0.5)
    assert roth.allocation["us_bonds_10yr"] == pytest.approx(0.5 * 0.875)
    assert plan529.allocation == BASE[3].allocation  # untouched
    for a in cand:
        assert sum(a.allocation.values()) == pytest.approx(1.0)


def test_candidate_zero_ladder_uses_fallback_template():
    cand = household_candidate(BASE, None, equity=0.6, ladder_total=0.0)
    trad = cand[1]
    # The all-ladder IRA has no liquid template of its own: it falls back to
    # the household's (equities + muni-heavy defensive), rescaled to 60%.
    assert "tips_ladder" not in trad.allocation
    assert trad.allocation["us_equities"] == pytest.approx(0.6)
    assert sum(trad.allocation.values()) == pytest.approx(1.0)


def test_candidate_clips_ladder_to_capacity():
    cand = household_candidate(BASE, None, equity=0.6, ladder_total=50e6)
    assert cand[0].allocation["tips_ladder"] == 1.0  # taxable maxed
    assert cand[1].allocation["tips_ladder"] == 1.0
    assert "tips_ladder" not in cand[3].allocation  # 529 never takes rungs


def test_optimize_household_search_runs():
    idx = pd.period_range("1960-01", periods=480, freq="M")
    rng = np.random.default_rng(3)
    panel = pd.DataFrame({
        "us_equities": rng.normal(0.008, 0.04, 480),
        "muni_bonds": rng.normal(0.003, 0.01, 480),
        "us_bonds_10yr": rng.normal(0.003, 0.01, 480),
        "cash": np.full(480, 0.002),
        "inflation": np.full(480, 0.002),
    }, index=idx)
    base = SimConfig(
        accounts=(
            Account("taxable", 1_000_000.0, allocation={
                "us_equities": 0.6, "muni_bonds": 0.35, "cash": 0.05}),
            Account("roth", 200_000.0, allocation={
                "us_equities": 0.6, "us_bonds_10yr": 0.35, "cash": 0.05}),
        ),
        years=10, n_sims=200, seed=1, ladder_yield=0.02,
        withdrawal=Withdrawal("fixed_real", rate=0.05),
    )
    best, board = optimize_household(
        panel, base, equity_grid=[0.4, 0.8], ladder_grid=[0.0, 200_000.0],
        screen_sims=100, refine_seeds=(1, 2), top_k=2,
    )
    assert len(board) == 2
    assert board[0]["success"] >= board[1]["success"] - 1e-9
    assert isinstance(best, tuple) and len(best) == 2


# --- audit fixes -------------------------------------------------------------


def test_candidates_share_one_asset_universe():
    # Every candidate carries the full union of liquid assets (zero-weighted
    # where unused) so all candidates resolve the same sampling window.
    cand = household_candidate(BASE, None, equity=0.5, ladder_total=1_000_000.0)
    for a in cand[:3]:  # non-529 accounts
        assert {"us_equities", "muni_bonds", "us_bonds_10yr", "cash"} <= set(a.allocation)


def test_equity_search_requires_both_buckets():
    idx = pd.period_range("1960-01", periods=480, freq="M")
    panel = pd.DataFrame({"us_equities": np.zeros(480),
                          "inflation": np.zeros(480)}, index=idx)
    base = SimConfig(
        accounts=(Account("taxable", 1000.0, allocation={"us_equities": 1.0}),),
        years=2, n_sims=8, seed=1,
    )
    with pytest.raises(ValueError, match="defensive"):
        optimize_household(panel, base, equity_grid=[0.4, 0.6], screen_sims=8)


def test_zero_balance_account_does_not_crash_candidate_builder():
    accts = BASE[:3] + (Account("roth", 0.0, allocation={"us_equities": 1.0}),)
    cand = household_candidate(accts, None, equity=0.5, ladder_total=0.0)
    assert "tips_ladder" not in cand[3].allocation  # no division crash


def test_clipped_ladder_candidates_labeled_and_deduped():
    idx = pd.period_range("1960-01", periods=480, freq="M")
    rng = np.random.default_rng(5)
    panel = pd.DataFrame({
        "us_equities": rng.normal(0.005, 0.03, 480),
        "cash": np.full(480, 0.002), "inflation": np.zeros(480),
    }, index=idx)
    base = SimConfig(
        accounts=(Account("taxable", 1_000_000.0,
                          allocation={"us_equities": 0.7, "cash": 0.3}),),
        years=5, n_sims=50, seed=3, ladder_yield=0.02,
        withdrawal=Withdrawal("fixed_real", rate=0.04),
    )
    best, board = optimize_household(
        panel, base, equity_grid=[0.7], ladder_grid=[2e6, 5e6],
        screen_sims=50, refine_seeds=(3,), top_k=5,
    )
    assert len(board) == 1  # both grid points clip to the same $1M portfolio
    assert "$1M" in board[0]["label"] and "clipped" in board[0]["label"]


def test_screen_is_deterministic_without_a_seed():
    idx = pd.period_range("1960-01", periods=480, freq="M")
    rng = np.random.default_rng(9)
    panel = pd.DataFrame({
        "us_equities": rng.normal(0.005, 0.03, 480),
        "cash": np.full(480, 0.002), "inflation": np.zeros(480),
    }, index=idx)
    base = SimConfig(
        accounts=(Account("taxable", 1_000_000.0,
                          allocation={"us_equities": 0.7, "cash": 0.3}),),
        years=5, n_sims=50, seed=None,
        withdrawal=Withdrawal("fixed_real", rate=0.04),
    )
    kw = dict(equity_grid=[0.5, 0.7], screen_sims=50, refine_seeds=(3,), top_k=2)
    _, b1 = optimize_household(panel, base, **kw)
    _, b2 = optimize_household(panel, base, **kw)
    assert [r["success"] for r in b1] == [r["success"] for r in b2]


def test_rescale_equity_keeps_bucket_proportions():
    from poorcast.optimize import equity_share, rescale_equity

    alloc = {"us_equities": 0.45, "intl_equities": 0.15, "muni_bonds": 0.3,
             "cash": 0.1, "tips_ladder": 0.0}
    assert equity_share(alloc) == pytest.approx(0.6)
    out = rescale_equity(alloc, 0.8)
    assert out == pytest.approx({"us_equities": 0.6, "intl_equities": 0.2,
                                 "muni_bonds": 0.15, "cash": 0.05})
    # a missing bucket falls back to the template
    out = rescale_equity({"muni_bonds": 1.0}, 0.5, fallback_eq={"us_equities": 1.0})
    assert out == pytest.approx({"us_equities": 0.5, "muni_bonds": 0.5})
    with pytest.raises(ValueError, match="no equity"):
        rescale_equity({"muni_bonds": 1.0}, 0.5)


# --- tolerance-based selection -----------------------------------------------


def _rows():
    # success falls as the ladder shrinks, estate rises: the classic frontier
    return [
        {"label": "L8", "success": 0.999, "stress_success": 0.989, "median": 25.9, "p5": 5.0},
        {"label": "L6", "success": 0.990, "stress_success": 0.949, "median": 30.4, "p5": 4.0},
        {"label": "L5", "success": 0.983, "stress_success": 0.925, "median": 32.7, "p5": 3.5},
        {"label": "L4", "success": 0.975, "stress_success": 0.902, "median": 35.0, "p5": 3.0},
    ]


def test_pick_within_tolerance_slides_with_the_band():
    from poorcast.optimize import pick_within_tolerance, tolerance_picks

    rows = _rows()
    assert pick_within_tolerance(rows, 0.0)["label"] == "L8"      # strict: best success
    assert pick_within_tolerance(rows, 0.01)["label"] == "L6"     # within 1 pt: max estate
    assert pick_within_tolerance(rows, 0.02)["label"] == "L5"
    assert pick_within_tolerance(rows, 0.03)["label"] == "L4"
    # anchored to stress the band is tighter: only L8 is within 1 pt of 98.9%
    assert pick_within_tolerance(rows, 0.01, anchor="stress")["label"] == "L8"
    assert pick_within_tolerance(rows, 0.05, anchor="stress")["label"] == "L6"
    picks = tolerance_picks(rows, "base")
    assert [r["label"] for r in picks.values()] == ["L6", "L5", "L4"]
    with pytest.raises(ValueError, match="anchor"):
        pick_within_tolerance(rows, 0.01, anchor="worst")
    with pytest.raises(ValueError, match="stress scenario"):
        pick_within_tolerance([{"success": 1.0, "median": 1.0, "p5": 0.0}], 0.01, anchor="stress")


def _frontier_panel():
    idx = pd.period_range("1960-01", periods=480, freq="M")
    rng = np.random.default_rng(3)
    return pd.DataFrame({
        "us_equities": rng.normal(0.008, 0.045, 480),
        "muni_bonds": rng.normal(0.002, 0.01, 480),
        "cash": np.full(480, 0.002),
        "inflation": np.full(480, 0.002),
    }, index=idx)


def test_optimize_household_tolerance_prefers_estate_within_band():
    panel = _frontier_panel()
    base = SimConfig(
        accounts=(Account("taxable", 1_000_000.0,
                          allocation={"us_equities": 0.6, "muni_bonds": 0.35, "cash": 0.05}),),
        years=10, n_sims=300, seed=1, ladder_yield=0.02,
        withdrawal=Withdrawal("fixed_real", rate=0.05),
    )
    kw = dict(equity_grid=[0.4, 0.8], ladder_grid=[0.0, 500_000.0],
              screen_sims=300, refine_seeds=(1, 2), top_k=4)
    strict_best, strict_board = optimize_household(panel, base, **kw)
    loose_best, loose_board, screened = optimize_household(
        panel, base, success_tolerance=0.5, return_screen=True, **kw)
    assert len(screened) == 4
    strict_row = max(strict_board, key=lambda r: (r["success"], r["p5"], r["median"]))
    loose_row = next(r for r in loose_board if r["accounts"] == loose_best)
    # a 50-point band ties everything, so the loose pick is the max-estate
    # candidate, which never beats the strict pick on success
    assert loose_row["median"] == max(r["median"] for r in loose_board)
    assert loose_row["success"] <= strict_row["success"] + 1e-9
    with pytest.raises(ValueError, match="stress scenario"):
        optimize_household(panel, base, success_tolerance=0.01, anchor="stress", **kw)


def test_optimize_household_scores_a_stress_world():
    from dataclasses import replace

    panel = _frontier_panel()
    base = SimConfig(
        accounts=(Account("taxable", 1_000_000.0,
                          allocation={"us_equities": 0.6, "muni_bonds": 0.35, "cash": 0.05}),),
        years=10, n_sims=200, seed=1, ladder_yield=0.02,
        withdrawal=Withdrawal("fixed_real", rate=0.05),
    )
    stress = replace(base, return_adjustments={"us_equities": -0.06})
    best, board = optimize_household(
        panel, base, equity_grid=[0.5], ladder_grid=[0.0, 500_000.0],
        screen_sims=200, refine_seeds=(1,), top_k=2, stress=stress,
        success_tolerance=0.02, anchor="stress")
    assert all("stress_success" in r and r["stress_success"] <= r["success"] + 1e-9 for r in board)
