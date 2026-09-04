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


def ladder_rows(spec: "LadderSpec") -> list[dict]:
    """The rung-by-rung buy list: one row per maturity year with the real
    face value to purchase, its coupon rate, and the year's total real
    income (which equals spec.annual for every year by construction)."""
    faces = np.asarray(spec.faces, dtype=float)
    coupons = np.asarray(spec.coupons, dtype=float)
    rows = []
    for t in range(spec.years):
        income = float(faces[t] + (coupons[t:] * faces[t:]).sum())
        rows.append({
            "year": t + 1,
            "face": float(faces[t]),
            "coupon": float(coupons[t]),
            "income": income,
        })
    return rows


def outstanding_tips(refresh: bool = False, asof=None) -> list[dict]:
    """Currently outstanding TIPS (not yet matured) from TreasuryDirect, one
    row per CUSIP: {cusip, maturity (date), coupon (decimal), term}."""
    import datetime
    import json

    from .data import _download

    blob = _download(
        "https://www.treasurydirect.gov/TA_WS/securities/search?type=TIPS&format=json",
        "treasurydirect_tips.json", refresh,
    )
    asof = asof or datetime.date.today()
    by_cusip: dict[str, dict] = {}
    for d in json.loads(blob):
        c, m = d.get("cusip"), (d.get("maturityDate") or "")[:10]
        if not c or not m:
            continue
        mat = datetime.date.fromisoformat(m)
        if mat <= asof or c in by_cusip:  # skip matured; dedupe reopenings
            continue
        by_cusip[c] = {
            "cusip": c, "maturity": mat, "term": d.get("securityTerm", ""),
            "coupon": float(d.get("interestRate") or 0.0) / 100.0,
        }
    return sorted(by_cusip.values(), key=lambda r: r["maturity"])


def match_cusips(n_rungs: int, tips: list[dict], base_year: int) -> list[dict | None]:
    """For each rung year (base_year+1 .. base_year+n_rungs), the outstanding
    TIPS maturing in that calendar year (earliest, so funds arrive in time),
    or None when no TIPS matures then."""
    from collections import defaultdict

    by_year: dict[int, list] = defaultdict(list)
    for t in tips:
        by_year[t["maturity"].year].append(t)
    picks = []
    for k in range(1, n_rungs + 1):
        cands = sorted(by_year.get(base_year + k, []), key=lambda t: t["maturity"])
        picks.append(cands[0] if cands else None)
    return picks


def build_available_ladder(
    annual: float, avail: list[int], coupons: dict[int, float], horizon: int
) -> list[dict]:
    """Gap-adjusted rung faces when TIPS mature only in `avail` years (1-based
    offsets), covering spending years 1..horizon.

    Each spending year is funded by the nearest available maturity at or
    before it; a year with no maturing bond is covered by holding the prior
    rung's principal in short TIPS (~0% real) until it is needed. Faces are
    solved backward so coupons from longer rungs offset earlier needs, exactly
    as the full ladder does (and reducing to it when every year is available).

    Returns one row per available maturity: {offset, coupon, covers (list of
    covered spending-year offsets), face}. Caller handles years beyond the last
    available maturity (the bridge tail).
    """
    avail = sorted(a for a in avail if 1 <= a <= horizon)
    if not avail:
        return []
    # Assign each spending year to the latest available maturity <= it
    # (years before the first maturity attach to it — money arrives slightly
    # late, an unavoidable near-term edge).
    covers: dict[int, list[int]] = {a: [] for a in avail}
    for y in range(1, horizon + 1):
        earlier = [a for a in avail if a <= y]
        covers[max(earlier)].append(y) if earlier else covers[avail[0]].append(y)
    faces: dict[int, float] = {}
    coup_later = 0.0  # sum of c_k * F_k for rungs longer than the current one
    for a in reversed(avail):
        L = len(covers[a])
        c = coupons.get(a, 0.0)
        f = L * (annual - coup_later) / (1 + c)
        faces[a] = f
        coup_later += c * f
    return [
        {"offset": a, "coupon": coupons.get(a, 0.0), "covers": covers[a], "face": faces[a]}
        for a in avail
    ]


def format_ladder_gap_adjusted(
    spec: "LadderSpec", tips: list[dict], base_year: int, label: str = ""
) -> str:
    """Buy list adjusted for real TIPS availability: one row per purchasable
    CUSIP (face consolidated to also cover the gap years it funds), plus a
    tail note for spending years past the last issuable maturity."""
    avail_by_year = {}
    for t in tips:
        y = t["maturity"].year
        if y not in avail_by_year or t["maturity"] < avail_by_year[y]["maturity"]:
            avail_by_year[y] = t
    last_avail = max((t["maturity"].year for t in tips), default=base_year)
    buildable = min(spec.years, last_avail - base_year)
    avail_off = [y - base_year for y in avail_by_year if base_year < y <= base_year + buildable]
    coupons = {y - base_year: avail_by_year[y]["coupon"] for y in avail_by_year}
    rungs = build_available_ladder(spec.annual, avail_off, coupons, buildable)

    out = []
    head = f"TIPS ladder{': ' + label if label else ''} (gap-adjusted to real CUSIPs)"
    out.append(
        f"{head} — ${spec.annual:,.0f}/yr real for {spec.years}y; "
        f"buy {len(rungs)} securities:"
    )
    out.append(f"  {'CUSIP':>11}  {'matures':>10}  {'real face $':>13}  "
               f"{'coupon':>7}  covers")
    cost = 0.0
    for r in rungs:
        t = avail_by_year[base_year + r["offset"]]
        cy = [base_year + y for y in r["covers"]]
        span = f"{cy[0]}" if len(cy) == 1 else f"{cy[0]}-{cy[-1]}"
        cost += r["face"]
        out.append(
            f"  {t['cusip']:>11}  {t['maturity'].isoformat():>10}  "
            f"{r['face']:>13,.0f}  {r['coupon']:>6.2%}  {span}"
        )
    out.append(f"  buildable cost ${cost:,.0f} for years "
               f"{base_year + 1}-{base_year + buildable}")
    tail = spec.years - buildable
    if tail > 0:
        out.append(
            f"  + {tail} tail year(s) {base_year + buildable + 1}-"
            f"{base_year + spec.years}: no TIPS issued that long yet — "
            "bridge with 30y TIPS rolled at future auctions."
        )
    return "\n".join(out)


def format_ladder(
    spec: "LadderSpec", label: str = "", base_year: int | None = None,
    cusips: "list | None" = None,
) -> str:
    """Human-readable buy list for a ladder. With `cusips` (from match_cusips,
    aligned to the rungs) it adds a real-security column and flags gap years."""
    import datetime

    if base_year is None:
        base_year = datetime.date.today().year
    rows = ladder_rows(spec)
    out = []
    head = f"TIPS ladder{': ' + label if label else ''}"
    acct = "taxable" if getattr(spec, "taxable", False) else "tax-deferred"
    out.append(
        f"{head} — ${spec.cost:,.0f} cost -> ${spec.annual:,.0f}/yr real for "
        f"{spec.years}y (cost-weighted real yield {spec.real_yield:.2%}, held {acct})"
    )
    if cusips is None:
        out.append(f"  {'matures':>8}  {'real face $':>13}  {'coupon':>7}")
        for r in rows:
            out.append(
                f"  {base_year + r['year']:>8}  {r['face']:>13,.0f}  {r['coupon']:>6.2%}"
            )
    else:
        out.append(
            f"  {'matures':>8}  {'real face $':>13}  {'CUSIP':>11}  "
            f"{'actual mat.':>11}  {'coupon':>7}"
        )
        gaps = 0
        for r, t in zip(rows, cusips):
            if t is None:
                gaps += 1
                out.append(
                    f"  {base_year + r['year']:>8}  {r['face']:>13,.0f}  "
                    f"{'— none —':>11}  {'':>11}  {'':>7}"
                )
            else:
                out.append(
                    f"  {base_year + r['year']:>8}  {r['face']:>13,.0f}  "
                    f"{t['cusip']:>11}  {t['maturity'].isoformat():>11}  "
                    f"{t['coupon']:>6.2%}"
                )
        if gaps:
            out.append(
                f"  ({gaps} rung(s) have no maturing TIPS — fill with an adjacent "
                "maturity, or bridge/hold cash for that year)"
            )
    out.append(
        f"  {'TOTAL':>8}  {sum(r['face'] for r in rows):>13,.0f}   "
        f"(each year delivers ${spec.annual:,.0f} real)"
    )
    return "\n".join(out)
