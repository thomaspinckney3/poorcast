"""US federal tax computation for taxable portfolios (2026 law, Rev. Proc. 2025-32).

Annual tax on investment income, vectorized across simulation paths:
  - interest -> ordinary brackets (10..37%)
  - qualified dividends + net long-term gains -> 0/15/20% brackets, stacked on
    top of ordinary taxable income
  - standard deduction applied to ordinary income first, remainder shields
    preferential income
  - 3.8% NIIT on net investment income above the MAGI threshold

Bracket edges and the standard deduction are indexed to each path's cumulative
inflation (as the law indexes them to chained CPI); the NIIT threshold is NOT
indexed (statutory), so it bites more over time - as under current law.

Assumes investment income is the taxpayer's only income and all dividends are
qualified. State tax, the $3,000 loss offset against ordinary income, and the
short/long-term distinction are not modeled.
"""

from __future__ import annotations

import numpy as np

# (lower edge, rate); upper edge = next entry's lower edge
BRACKETS = {
    "single": {
        "ordinary": [
            (0, 0.10), (12_400, 0.12), (50_400, 0.22), (105_700, 0.24),
            (201_775, 0.32), (256_225, 0.35), (640_600, 0.37),
        ],
        "ltcg": [(0, 0.0), (49_450, 0.15), (545_500, 0.20)],
        "std_deduction": 16_100,
        "niit_threshold": 200_000,
    },
    "married": {
        "ordinary": [
            (0, 0.10), (24_800, 0.12), (100_800, 0.22), (211_400, 0.24),
            (403_550, 0.32), (512_450, 0.35), (768_700, 0.37),
        ],
        "ltcg": [(0, 0.0), (98_900, 0.15), (613_700, 0.20)],
        "std_deduction": 32_200,
        "niit_threshold": 250_000,
    },
}
NIIT_RATE = 0.038


def _bracket_tax(income: np.ndarray, brackets, scale: np.ndarray) -> np.ndarray:
    """Tax on `income` under (edge, rate) brackets with edges scaled per path."""
    tax = np.zeros_like(income)
    for i, (lo, rate) in enumerate(brackets):
        hi = brackets[i + 1][0] if i + 1 < len(brackets) else np.inf
        tax += rate * np.clip(income - lo * scale, 0.0, (hi - lo) * scale)
    return tax


def annual_tax(
    interest: np.ndarray,
    preferential: np.ndarray,
    filing: str,
    price_level: np.ndarray | float = 1.0,
    state_rate: float = 0.0,
) -> np.ndarray:
    """Federal (+ flat state) tax owed on a year's investment income (nominal).

    interest: ordinary-rate income (Treasury coupons, T-bills) - federally
    taxed, STATE-EXEMPT. preferential: qualified dividends + net long-term
    gains - federally preferential, state-taxable at the flat state_rate.
    Muni interest never reaches this function (exempt from both, assuming
    own-state bonds). price_level: cumulative inflation factor for indexing.
    State tax is not deducted federally (standard deduction assumed).
    """
    b = BRACKETS[filing]
    interest = np.asarray(interest, dtype=float)
    preferential = np.asarray(preferential, dtype=float)
    scale = np.broadcast_to(np.asarray(price_level, dtype=float), interest.shape)

    std = b["std_deduction"] * scale
    taxable_ord = np.maximum(interest - std, 0.0)
    deduction_left = np.maximum(std - interest, 0.0)
    taxable_pref = np.maximum(preferential - deduction_left, 0.0)

    tax = _bracket_tax(taxable_ord, b["ordinary"], scale)

    # Preferential income stacks on top of ordinary taxable income: each LTCG
    # rate applies to the slice of [taxable_ord, taxable_ord + taxable_pref]
    # that falls inside that bracket.
    total = taxable_ord + taxable_pref
    ltcg = b["ltcg"]
    for i, (lo, rate) in enumerate(ltcg):
        hi = ltcg[i + 1][0] if i + 1 < len(ltcg) else np.inf
        lo_edge = np.maximum(lo * scale, taxable_ord)
        hi_edge = np.minimum(hi * scale, total)
        tax += rate * np.maximum(hi_edge - lo_edge, 0.0)

    # NIIT: all income here is investment income; MAGI = gross investment
    # income; the threshold is statutory and NOT inflation-indexed.
    magi = interest + preferential
    nii_taxed = np.minimum(magi, np.maximum(magi - b["niit_threshold"], 0.0))
    tax += NIIT_RATE * nii_taxed

    # Flat state income tax on dividends and gains (states give them no
    # preferential rate); Treasury interest is constitutionally state-exempt.
    tax += state_rate * preferential
    return tax
