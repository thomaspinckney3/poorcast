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
    assert trad.allocation == {"tips_ladder": 1.0}
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
    assert cand[0].allocation == {"tips_ladder": 1.0}  # taxable maxed
    assert cand[1].allocation == {"tips_ladder": 1.0}
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
