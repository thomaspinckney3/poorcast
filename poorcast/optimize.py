"""Allocation search: find the asset mix that maximizes success probability.

Two stages: a coarse screen over a structured grid (equity level x equity
split x defensive split) with common random numbers, then refinement of the
distinct leaders across several seeds. Objective: success rate, tie-broken by
5th-percentile then median real terminal wealth.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .simulate import SimConfig, simulate

EQUITY_LEVELS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
EQ_SPLITS = {  # us_equities, us_small_cap, intl_equities
    "us-heavy": (5 / 7, 0, 2 / 7),
    "us-only": (1, 0, 0),
    "tilt-small": (0.5, 0.2, 0.3),
    "balanced": (0.4, 0.3, 0.3),
    "us+small": (0.6, 0.3, 0.1),
}
DEF_SPLITS = {  # muni_bonds, us_bonds_10yr, cash
    "muni": (1, 0, 0),
    "muni+cash": (0.8, 0, 0.2),
    "muni+tsy": (0.5, 0.5, 0),
    "tsy": (0, 1, 0),
    "mixed": (0.6, 0.2, 0.2),
}


def grid_allocation(equity: float, eq_split: str, def_split: str) -> dict[str, float]:
    us, sm, il = EQ_SPLITS[eq_split]
    mu, ty, ca = DEF_SPLITS[def_split]
    alloc = {
        "us_equities": equity * us,
        "us_small_cap": equity * sm,
        "intl_equities": equity * il,
        "muni_bonds": (1 - equity) * mu,
        "us_bonds_10yr": (1 - equity) * ty,
        "cash": (1 - equity) * ca,
    }
    return {k: v for k, v in alloc.items() if v > 1e-9}


def optimize(
    panel: pd.DataFrame,
    base: SimConfig,
    screen_sims: int = 2000,
    refine_sims: int = 4000,
    refine_seeds: tuple[int, ...] = (42, 7, 123),
    top_k: int = 8,
    equity_levels=None,
    progress=None,
) -> tuple[dict[str, float], list[dict]]:
    """Search the grid; return (best allocation, leaderboard of refined rows).

    `base` supplies everything except allocation/n_sims/seed (horizon,
    withdrawal, taxes, rebalancing...).
    """
    levels = equity_levels or EQUITY_LEVELS

    def run(alloc, sims, seed):
        cfg = replace(base, allocation=alloc, n_sims=sims, seed=seed)
        return simulate(panel, cfg)

    screened = []
    combos = [(e, q, d) for e in levels for q in EQ_SPLITS for d in DEF_SPLITS]
    for i, (e, q, d) in enumerate(combos):
        r = run(grid_allocation(e, q, d), screen_sims, 42)
        p5 = float(np.percentile(r.real_balance[:, -1], 5))
        screened.append((r.success_rate, p5, e, q, d))
        if progress and (i + 1) % 25 == 0:
            progress(f"  screened {i + 1}/{len(combos)} allocations...")
    # tie-break ties in success (common when many mixes never deplete) so the
    # refinement stage sees the genuinely best candidates, not grid order
    screened.sort(key=lambda x: (-x[0], -x[1]))

    leaderboard = []
    for _, _, e, q, d in screened[:top_k]:
        alloc = grid_allocation(e, q, d)
        succ, p5s, meds, sds = [], [], [], []
        for seed in refine_seeds:
            r = run(alloc, refine_sims, seed)
            t = r.real_balance[:, -1]
            succ.append(r.success_rate)
            p5s.append(float(np.percentile(t, 5)))
            meds.append(float(np.median(t)))
            sds.append(float(t.std()))
        leaderboard.append(
            {
                "allocation": alloc,
                "label": f"{e:.0%} equity [{q}] / defensive [{d}]",
                "success": float(np.mean(succ)),
                "success_sd": float(np.std(succ, ddof=1)) if len(succ) > 1 else 0.0,
                "terminal_p5": float(np.mean(p5s)),
                "terminal_median": float(np.mean(meds)),
                "terminal_sd": float(np.mean(sds)),
            }
        )
    leaderboard.sort(
        key=lambda r: (-r["success"], -r["terminal_p5"], -r["terminal_median"])
    )
    return leaderboard[0]["allocation"], leaderboard
