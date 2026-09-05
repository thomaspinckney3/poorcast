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

Assumes all dividends are qualified; `other_ordinary` carries any
non-investment ordinary income (IRA distributions, pensions). A flat state
rate applies to dividends, gains, and other ordinary income (Treasury
interest is state-exempt). The $3,000 loss offset against ordinary income
and the short/long-term distinction are not modeled.
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

# IRS Uniform Lifetime Table (2022+): age -> distribution period in years.
# RMD for a year = prior Dec 31 balance / period at the age attained that year.
RMD_START_AGE = 73  # SECURE 2.0 (rises to 75 for those reaching 73 after 2032)
RMD_TABLE = {
    72: 27.4, 73: 26.5, 74: 25.5, 75: 24.6, 76: 23.7, 77: 22.9, 78: 22.0,
    79: 21.1, 80: 20.2, 81: 19.4, 82: 18.5, 83: 17.7, 84: 16.8, 85: 16.0,
    86: 15.2, 87: 14.4, 88: 13.7, 89: 12.9, 90: 12.2, 91: 11.5, 92: 10.8,
    93: 10.1, 94: 9.5, 95: 8.9, 96: 8.4, 97: 7.8, 98: 7.3, 99: 6.8,
    100: 6.4, 101: 6.0, 102: 5.6, 103: 5.2, 104: 4.9, 105: 4.6, 106: 4.3,
    107: 4.1, 108: 3.9, 109: 3.7, 110: 3.5, 111: 3.4, 112: 3.3, 113: 3.1,
    114: 3.0, 115: 2.9, 116: 2.8, 117: 2.7, 118: 2.5, 119: 2.3, 120: 2.0,
}


def rmd_period(age: int) -> float:
    """Uniform Lifetime Table distribution period for the age attained this year."""
    return RMD_TABLE[min(max(age, 72), 120)]


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
    other_ordinary: np.ndarray | float = 0.0,
) -> np.ndarray:
    """Federal (+ flat state) tax owed on a year's investment income (nominal).

    interest: ordinary-rate income (Treasury coupons, T-bills) - federally
    taxed, STATE-EXEMPT. preferential: qualified dividends + net long-term
    gains - federally preferential, state-taxable at the flat state_rate.
    other_ordinary: non-investment ordinary income (IRA distributions,
    pensions) - stacked through the ordinary brackets, state-taxable, raises
    MAGI for NIIT purposes but is not itself investment income.
    Muni interest never reaches this function (exempt from both, assuming
    own-state bonds). price_level: cumulative inflation factor for indexing.
    State tax is not deducted federally (standard deduction assumed).
    """
    b = BRACKETS[filing]
    interest = np.asarray(interest, dtype=float)
    preferential = np.asarray(preferential, dtype=float)
    shape = np.broadcast_shapes(
        interest.shape, preferential.shape, np.shape(other_ordinary)
    )
    interest = np.broadcast_to(interest, shape)
    preferential = np.broadcast_to(preferential, shape)
    other_ordinary = np.broadcast_to(np.asarray(other_ordinary, dtype=float), shape)
    scale = np.broadcast_to(np.asarray(price_level, dtype=float), shape)

    ordinary = interest + other_ordinary
    std = b["std_deduction"] * scale
    taxable_ord = np.maximum(ordinary - std, 0.0)
    deduction_left = np.maximum(std - ordinary, 0.0)
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

    # NIIT: investment income above the MAGI threshold. other_ordinary raises
    # MAGI but is not investment income; the threshold is statutory and NOT
    # inflation-indexed.
    invest = interest + preferential
    magi = invest + other_ordinary
    nii_taxed = np.minimum(invest, np.maximum(magi - b["niit_threshold"], 0.0))
    tax += NIIT_RATE * nii_taxed

    # Flat state income tax on dividends, gains, and ordinary non-investment
    # income (states give them no preferential rate); Treasury interest is
    # constitutionally state-exempt.
    tax += state_rate * (preferential + other_ordinary)
    return tax
