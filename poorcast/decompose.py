"""Decompose historical US equity returns into their sources.

    total return ~ dividend yield + inflation + real EPS growth + P/E change
    real EPS growth = margin expansion (CP/GDP) + underlying (output etc.)

Endpoints use 5-year averages (single-year earnings and multiples are too
noisy), so contributions are measured between the window midpoints. Data:
Shiller monthly P/D/E, FRED corporate profits (CP) and GDP, plus the panel's
own us_equities series for the actual total return.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd

from .data import SHILLER_URL, _download, fetch_fred, load_panel


def _shiller_pde(refresh: bool = False) -> pd.DataFrame:
    blob = _download(SHILLER_URL, "shiller_ie_data.xls", refresh, ua="Mozilla/5.0")
    df = pd.read_excel(io.BytesIO(blob), sheet_name="Data", header=None, engine="xlrd")
    rows = df[0].astype(str).str.fullmatch(r"\d{4}\.\d{2}")
    df = df[rows]
    idx = pd.PeriodIndex([f"{d[:4]}-{d[5:7]}" for d in df[0].astype(str)], freq="M")
    out = pd.DataFrame(
        {
            "P": pd.to_numeric(df[1], errors="coerce").to_numpy(),
            "D": pd.to_numeric(df[2], errors="coerce").to_numpy(),
            "E": pd.to_numeric(df[3], errors="coerce").to_numpy(),
            "CPI": pd.to_numeric(df[4], errors="coerce").to_numpy(),
        },
        index=idx,
    )
    return out


def equity_return_decomposition(
    start_window: tuple[str, str] = ("1960-01", "1964-12"),
    end_window: tuple[str, str] | None = None,
    refresh: bool = False,
) -> dict:
    """Annualized (log) contributions between the two 5-year windows' midpoints."""
    sh = _shiller_pde(refresh)
    if end_window is None:
        last = sh["E"].dropna().index.max()
        end_window = (str(last - 59), str(last))

    def avg(s, w):
        return s[pd.Period(w[0], "M") : pd.Period(w[1], "M")].mean()

    def mid(w):
        a, b = pd.Period(w[0], "M"), pd.Period(w[1], "M")
        return pd.Period(ordinal=(a.ordinal + b.ordinal + 1) // 2, freq="M")

    mid_a, mid_b = mid(start_window), mid(end_window)
    years = (mid_b.ordinal - mid_a.ordinal) / 12

    e_real = sh["E"] * sh["CPI"].iloc[-1] / sh["CPI"]
    pe = sh["P"] / sh["E"]
    eps_growth = float(np.log(avg(e_real, end_window) / avg(e_real, start_window)) / years)
    multiple = float(np.log(avg(pe, end_window) / avg(pe, start_window)) / years)
    inflation = float(np.log(avg(sh["CPI"], end_window) / avg(sh["CPI"], start_window)) / years)
    div_yield = float((sh["D"] / sh["P"])[start_window[0] : end_window[1]].mean())

    margin = (fetch_fred("CP", refresh) / fetch_fred("GDP", refresh)).dropna()
    margin_growth = float(np.log(avg(margin, end_window) / avg(margin, start_window)) / years)

    eq = load_panel()["us_equities"]
    actual = float(np.log1p(eq[mid_a:mid_b]).mean() * 12)

    total = div_yield + inflation + eps_growth + multiple
    return {
        "window": f"{start_window[0]}..{end_window[1]} (midpoint to midpoint, {years:.0f}y)",
        "actual_total_return": actual,
        "dividend_yield": div_yield,
        "inflation": inflation,
        "real_eps_growth": eps_growth,
        "margin_expansion": margin_growth,
        "underlying_growth": eps_growth - margin_growth,
        "multiple_expansion": multiple,
        "sum_of_components": total,
        "residual": actual - total,
        "pe_start": float(avg(pe, start_window)),
        "pe_end": float(avg(pe, end_window)),
    }


def print_decomposition(d: dict) -> None:
    print(f"US equity return sources, {d['window']}")
    print(f"  P/E (5y-averaged): {d['pe_start']:.1f} -> {d['pe_end']:.1f}\n")
    print(f"  actual total return                {d['actual_total_return']:+.2%}/yr")
    print(f"    dividend yield                   {d['dividend_yield']:+.2%}")
    print(f"    inflation                        {d['inflation']:+.2%}")
    print(f"    real EPS growth                  {d['real_eps_growth']:+.2%}")
    print(f"      margin expansion (CP/GDP)      {d['margin_expansion']:+.2%}")
    print(f"      underlying (output etc.)       {d['underlying_growth']:+.2%}")
    print(f"    P/E multiple expansion           {d['multiple_expansion']:+.2%}")
    print(f"    (components sum {d['sum_of_components']:+.2%}, residual "
          f"{d['residual']:+.2%}: log/arithmetic cross-terms and S&P-vs-total-market wedge)")
