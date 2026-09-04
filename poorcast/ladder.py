"""TIPS ladder construction: cash-flow matching a real spending floor.

A ladder is priced off the real yield at purchase and held to maturity, so it
needs no return series and carries no mark-to-market risk in the simulation:
each rung matures in the year it funds. Rungs are assumed to be par TIPS
(coupon = yield). Later rungs pay coupons in earlier years, so face values are
solved backwards (the system is lower-triangular).

Everything is in real (inflation-adjusted) terms; TIPS principal accrual makes
that exact by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LadderSpec:
    annual: float  # real dollars per year the ladder delivers
    years: int
    real_yield: float  # representative (cost-weighted) real yield at purchase
    cost: float  # up-front cost (sum of rung face values, par bonds)
    faces: tuple = ()  # real face value per rung, years 1..years
    coupons: tuple = ()  # real coupon rate per rung
    # Held in a taxable account: coupons AND the annual inflation accrual on
    # remaining principal ("phantom income") are federal ordinary income
    # (state-exempt, Treasury). False = tax-deferred account.
    taxable: bool = False

    def coupon_income_real(self) -> np.ndarray:
        """Real coupon income received during year t (0-indexed)."""
        f, c = np.array(self.faces), np.array(self.coupons)
        return np.array([(c[t:] * f[t:]).sum() for t in range(self.years)])

    def remaining_principal_real(self) -> np.ndarray:
        """Real principal outstanding during year t (0-indexed)."""
        f = np.array(self.faces)
        return np.array([f[t:].sum() for t in range(self.years)])


def rung_faces(annual: float, years: int, real_yield) -> np.ndarray:
    """Face value of the rung maturing in each year 1..years (par TIPS).
    real_yield: scalar, or an array of per-maturity yields (years 1..years)."""
    c = np.broadcast_to(np.asarray(real_yield, dtype=float), (years,))
    face = np.zeros(years)
    for y in range(years - 1, -1, -1):
        coupons_from_later = (c[y + 1 :] * face[y + 1 :]).sum()
        face[y] = (annual - coupons_from_later) / (1 + c[y])
    return face


def _build(annual: float, years: int, yields: np.ndarray, taxable: bool) -> LadderSpec:
    if annual <= 0:
        raise ValueError("ladder annual amount must be positive")
    if not ((0 <= yields) & (yields < 0.2)).all():
        raise ValueError(
            f"unsupported real yield(s) {yields}: the par-TIPS construction "
            "(coupon = yield) cannot represent negative real yields"
        )
    faces = rung_faces(annual, years, yields)
    cost = float(faces.sum())
    rep = float((yields * faces).sum() / cost)
    return LadderSpec(annual=annual, years=years, real_yield=rep, cost=cost,
                      faces=tuple(faces), coupons=tuple(yields), taxable=taxable)


def build_ladder(
    annual: float, years: int, real_yield: float, taxable: bool = False
) -> LadderSpec:
    return _build(annual, years, np.full(years, float(real_yield)), taxable)


def build_ladder_curve(
    annual: float,
    years: int,
    curve: dict[float, float],
    taxable: bool = False,
    tail_yield: float | None = None,
) -> LadderSpec:
    """curve: {maturity_years: real_yield} points (e.g. from FRED DFII series),
    linearly interpolated across rung maturities, flat beyond the endpoints.

    tail_yield prices rungs BEYOND the curve's longest maturity (typically
    30y - such rungs cannot be bought today and must be rolled into later at
    future long real yields). Default: flat at the longest observed yield,
    which approximates locking the tail with bridge bonds; a conservative
    bracket is the historical DFII30 median (~1%)."""
    mats = sorted(curve)
    ys = np.interp(np.arange(1, years + 1), mats, [curve[m] for m in mats])
    if tail_yield is not None:
        ys = np.where(np.arange(1, years + 1) > mats[-1], tail_yield, ys)
    return _build(annual, years, ys, taxable)


def current_real_curve(refresh: bool = True) -> dict[float, float]:
    """Latest TIPS real yields from FRED (5/7/10/20/30y), as decimals."""
    import io

    import pandas as pd

    from .data import _download

    out = {}
    for mat, series in [(5, "DFII5"), (7, "DFII7"), (10, "DFII10"),
                        (20, "DFII20"), (30, "DFII30")]:
        blob = _download(
            f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}",
            f"fred_{series}.csv", refresh,
        )
        s = pd.read_csv(io.BytesIO(blob), na_values=".").dropna()
        out[mat] = float(s.iloc[-1, 1]) / 100.0
    return out
