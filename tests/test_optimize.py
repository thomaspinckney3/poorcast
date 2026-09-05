"""Optimizer tests on a synthetic panel where the best mix is known."""

import numpy as np
import pandas as pd

from poorcast.optimize import grid_allocation, optimize
from poorcast.simulate import SimConfig, Withdrawal


def synthetic_panel(n=480):
    idx = pd.period_range("1960-01", periods=n, freq="M")
    rng = np.random.default_rng(1)
    # equities: strong returns; bonds/cash/munis: weak. High equity should win.
    cols = {
        "us_equities": rng.normal(0.009, 0.04, n),
        "us_small_cap": rng.normal(0.009, 0.05, n),
        "intl_equities": rng.normal(0.007, 0.045, n),
        "muni_bonds": rng.normal(0.002, 0.01, n),
        "us_bonds_10yr": rng.normal(0.002, 0.015, n),
        "cash": np.full(n, 0.001),
        "inflation": np.full(n, 0.002),
    }
    return pd.DataFrame(cols, index=idx)


def test_grid_allocation_sums_to_one():
    a = grid_allocation(0.7, "us+small", "muni+cash")
    assert np.isclose(sum(a.values()), 1.0)
    assert all(v > 0 for v in a.values())


def test_optimizer_prefers_equity_when_it_dominates():
    panel = synthetic_panel()
    base = SimConfig(allocation={"us_equities": 1.0}, initial=1e6, years=15,
                     n_sims=100, seed=0,
                     withdrawal=Withdrawal("fixed_real", rate=0.05))
    best, board = optimize(panel, base, screen_sims=200, refine_sims=400,
                           refine_seeds=(1, 2), top_k=3,
                           equity_levels=[0.3, 0.6, 0.9])
    equity = sum(v for k, v in best.items()
                 if k in ("us_equities", "us_small_cap", "intl_equities"))
    assert equity >= 0.9
    assert board[0]["success"] >= board[-1]["success"]
    assert "success_sd" in board[0]


def test_screen_candidates_share_one_sampling_window(monkeypatch):
    # intl has no data before 1965: a mix without it would otherwise sample
    # from 1960 while a mix with it samples from 1965, breaking common
    # random numbers. Every candidate must resolve the same window.
    import poorcast.optimize as opt

    panel = synthetic_panel()
    panel.loc[panel.index < pd.Period("1965-01", "M"), "intl_equities"] = np.nan
    windows = set()
    real = opt.simulate

    def spy(p, cfg):
        r = real(p, cfg)
        windows.add((str(r.window.min()), str(r.window.max())))
        return r

    monkeypatch.setattr(opt, "simulate", spy)
    base = SimConfig(allocation={"us_equities": 1.0}, initial=1e6, years=5,
                     n_sims=50, seed=0, withdrawal=Withdrawal("fixed_real", rate=0.04))
    optimize(panel, base, screen_sims=50, refine_sims=50, refine_seeds=(1,),
             top_k=2, equity_levels=[0.5])
    assert windows == {("1965-01", "1999-12")}
