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
GRID_ASSETS = [
    "us_equities", "us_small_cap", "intl_equities",
    "muni_bonds", "us_bonds_10yr", "cash",
]


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
    # Every candidate carries the whole grid universe (zero-weighted where
    # unused) so the engine resolves the SAME sampling window for each -
    # otherwise common random numbers break between mixes that include a
    # short-history asset and mixes that don't.
    universe = [a for a in GRID_ASSETS if a in panel.columns]

    def run(alloc, sims, seed):
        full = {a: 0.0 for a in universe}
        full.update(alloc)
        cfg = replace(base, allocation=full, n_sims=sims, seed=seed)
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


# --- household-mode optimization --------------------------------------------

EQUITY_ASSETS = frozenset({"us_equities", "us_small_cap", "intl_equities"})
LADDER = "tips_ladder"


def _buckets(alloc: dict) -> tuple[dict, dict]:
    eq = {k: v for k, v in alloc.items() if k in EQUITY_ASSETS}
    de = {k: v for k, v in alloc.items() if k not in EQUITY_ASSETS and k != LADDER}
    return eq, de


def _norm(d: dict) -> dict:
    s = sum(d.values())
    return {k: v / s for k, v in d.items()} if s > 0 else {}


def equity_share(alloc: dict) -> float:
    """Equity fraction of an allocation's liquid sleeve (ladder excluded)."""
    eq, de = _buckets(alloc)
    tot = sum(eq.values()) + sum(de.values())
    return sum(eq.values()) / tot if tot > 0 else 0.0


def rescale_equity(alloc: dict, equity: float, fallback_eq=None, fallback_de=None) -> dict:
    """Liquid-sleeve weights (summing to 1) at `equity` equities / (1-equity)
    defensive, preserving alloc's own intra-bucket proportions (falling back
    to the given bucket templates when alloc lacks a bucket)."""
    eqp = _norm(_buckets(alloc)[0]) or (fallback_eq or {})
    dep = _norm(_buckets(alloc)[1]) or (fallback_de or {})
    if equity > 0 and not eqp:
        raise ValueError("no equity assets to glide into (add an equity template)")
    if equity < 1 and not dep:
        raise ValueError("no defensive assets to glide into (add a defensive template)")
    new: dict = {}
    for k, v in eqp.items():
        new[k] = new.get(k, 0.0) + equity * v
    for k, v in dep.items():
        new[k] = new.get(k, 0.0) + (1.0 - equity) * v
    return new


def household_bucket_templates(accounts, base_alloc) -> tuple[dict, dict]:
    """The household's balance-weighted equity and defensive bucket
    proportions (non-529 accounts): the fallback templates for an account
    that lacks a bucket of its own."""
    agg_eq: dict = {}
    agg_de: dict = {}
    for a in accounts:
        if a.kind == "529":
            continue
        alloc = a.allocation or base_alloc or {}
        eq, de = _buckets(alloc)
        for k, v in eq.items():
            agg_eq[k] = agg_eq.get(k, 0.0) + v * a.balance
        for k, v in de.items():
            agg_de[k] = agg_de.get(k, 0.0) + v * a.balance
    return _norm(agg_eq), _norm(agg_de)


def household_candidate(accounts, base_alloc, equity: float, ladder_total: float):
    """Rebuild the accounts for a household equity share and total ladder cost.

    The account structure is fixed; each non-529 account's liquid sleeve is
    rescaled to `equity` equities / (1 - equity) defensive, preserving its
    own intra-bucket proportions (falling back to the household's when an
    account lacks a bucket). Ladder dollars fill traditional accounts first,
    then taxable (clipped to capacity); 529s are left untouched.
    """
    agg_eq, agg_de = household_bucket_templates(accounts, base_alloc)
    # Every candidate carries the full union of liquid assets (zero-weighted
    # where unused) so the engine resolves the SAME sampling window for every
    # candidate - otherwise common random numbers silently break at grid
    # edges that drop short-history assets.
    union = sorted(set(agg_eq) | set(agg_de))

    remaining = ladder_total
    assigned: dict[int, float] = {}
    for kind in ("traditional", "taxable"):
        for i, a in enumerate(accounts):
            if a.kind == kind:
                take = min(remaining, a.balance)
                assigned[i] = take
                remaining -= take

    out = []
    for i, a in enumerate(accounts):
        if a.kind == "529":
            out.append(a)
            continue
        alloc = a.allocation or base_alloc or {}
        wl = assigned.get(i, 0.0) / a.balance if a.balance > 0 else 0.0
        liquid = 1.0 - wl
        new: dict = {k: 0.0 for k in union}
        for k, v in rescale_equity(alloc, equity, agg_eq, agg_de).items():
            new[k] = new.get(k, 0.0) + liquid * v
        if wl > 1e-12:
            new[LADDER] = wl
        out.append(replace(a, allocation=new))
    return tuple(out)


def optimize_household(
    panel,
    base,
    equity_grid=None,
    ladder_grid=None,
    screen_sims: int = 4000,
    refine_seeds: tuple[int, ...] = (42, 7, 123),
    top_k: int = 5,
    progress=None,
):
    """Search household (equity share x total ladder) candidates.

    Screens the grid with common random numbers, refines the leaders across
    several seeds, and returns (best account tuple, leaderboard rows sorted
    by mean success, then p5, then median real terminal wealth).
    """
    accounts = base.accounts
    if equity_grid is None or ladder_grid is None:
        liq = eqd = 0.0
        cur_l = 0.0
        for a in accounts:
            alloc = a.allocation or base.allocation or {}
            wl = alloc.get(LADDER, 0.0)
            cur_l += wl * a.balance
            if a.kind == "529":
                continue
            liq += (1 - wl) * a.balance
            eqd += sum(v for k, v in alloc.items() if k in EQUITY_ASSETS) * a.balance
        if equity_grid is None:
            equity_grid = [eqd / liq if liq > 0 else 0.6]
        if ladder_grid is None:
            ladder_grid = [cur_l]

    if len(equity_grid) > 1:
        has_eq = has_de = False
        for a in accounts:
            if a.kind == "529":
                continue
            eq, de = _buckets(a.allocation or base.allocation or {})
            has_eq = has_eq or bool(eq)
            has_de = has_de or bool(de)
        if not (has_eq and has_de):
            raise ValueError(
                "an equity search needs both equity and defensive assets "
                "somewhere in the household's allocations"
            )
    capacity = sum(
        a.balance for a in accounts if a.kind in ("traditional", "taxable")
    )
    screen_seed = base.seed if base.seed is not None else refine_seeds[0]

    def stats(r):
        term = r.real_balance[:, -1]
        return {
            "success": r.success_rate,
            "p5": float(np.percentile(term, 5)),
            "median": float(np.median(term)),
            "floor": float(r.ladder_annual or 0.0),
        }

    rows = []
    seen = set()
    for L in ladder_grid:
        actual = min(L, capacity)
        clipped = "" if actual >= L else f" (clipped from ${L / 1e6:g}M)"
        for e in equity_grid:
            key = (round(actual, 6), round(e, 9))
            if key in seen:
                continue  # a clipped duplicate of an already-screened candidate
            seen.add(key)
            cand = household_candidate(accounts, base.allocation, e, actual)
            r = simulate(
                panel,
                replace(base, accounts=cand, n_sims=screen_sims, seed=screen_seed),
            )
            s = stats(r)
            row = {"label": f"ladder ${actual / 1e6:g}M · equity {e:.0%}{clipped}",
                   "accounts": cand, **s, "success_sd": 0.0}
            rows.append(row)
            if progress:
                progress(f"  {row['label']}: success {s['success']:.1%}, "
                         f"p5 ${s['p5'] / 1e6:.2f}M, median ${s['median'] / 1e6:.1f}M")
    rows.sort(key=lambda r: (-r["success"], -r["p5"], -r["median"]))

    refined = []
    for row in rows[:top_k]:
        succ, p5s, meds = [], [], []
        floor = row["floor"]
        for seed in refine_seeds:
            r = simulate(panel, replace(base, accounts=row["accounts"], seed=seed))
            s = stats(r)
            succ.append(s["success"])
            p5s.append(s["p5"])
            meds.append(s["median"])
        refined.append({
            "label": row["label"], "accounts": row["accounts"], "floor": floor,
            "success": float(np.mean(succ)),
            "success_sd": float(np.std(succ, ddof=1)) if len(succ) > 1 else 0.0,
            "p5": float(np.mean(p5s)), "median": float(np.mean(meds)),
        })
    refined.sort(key=lambda r: (-r["success"], -r["p5"], -r["median"]))
    return refined[0]["accounts"], refined
