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

from .data import fetch_fred, load_panel, shiller_month_index, shiller_workbook

SHILLER_CAPE_COL = 12  # Shiller's own P/E10 ("CAPE") column in ie_data.xls


def _shiller_pde(refresh: bool = False) -> pd.DataFrame:
    blob = shiller_workbook(refresh)
    df = pd.read_excel(io.BytesIO(blob), sheet_name="Data", header=None, engine="xlrd")
    rows, idx = shiller_month_index(df[0])
    df = df[rows]
    cols = {
        "P": pd.to_numeric(df[1], errors="coerce").to_numpy(),
        "D": pd.to_numeric(df[2], errors="coerce").to_numpy(),
        "E": pd.to_numeric(df[3], errors="coerce").to_numpy(),
        "CPI": pd.to_numeric(df[4], errors="coerce").to_numpy(),
    }
    if df.shape[1] > SHILLER_CAPE_COL:
        cols["CAPE"] = pd.to_numeric(df[SHILLER_CAPE_COL], errors="coerce").to_numpy()
    return pd.DataFrame(cols, index=idx)


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


def shiller_pe_series(refresh: bool = False) -> pd.Series:
    """Monthly valuation state for conditioned sampling and P/E paths: the
    Shiller CAPE (real price over trailing 10-year average real earnings) -
    his own published column when the workbook has it, else computed the
    same way. (Earlier versions used a 5-year average, which runs well below
    the quoted CAPE.)"""
    sh = _shiller_pde(refresh)
    if "CAPE" in sh and sh["CAPE"].notna().sum() > 120:
        return sh["CAPE"].dropna().rename("shiller_pe")
    return cape_from_pde(sh)


def cape_from_pde(sh: pd.DataFrame, years: int = 10) -> pd.Series:
    """Real price over trailing `years`-year average real earnings."""
    scale = sh["CPI"].iloc[-1] / sh["CPI"]
    p_real = sh["P"] * scale
    e_real = (sh["E"] * scale).rolling(12 * years).mean()
    return (p_real / e_real).dropna().rename("shiller_pe")


def current_cape(refresh: bool = False) -> tuple[float, "pd.Period"]:
    """Latest CAPE and its month."""
    s = shiller_pe_series(refresh)
    return float(s.iloc[-1]), s.index[-1]
