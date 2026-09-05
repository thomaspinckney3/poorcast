"""Reconstruct pre-1986 international developed-market equity returns (USD, monthly).

Method (per country, then GDP-weighted composite):
  1. Monthly local-currency price returns from the OECD MEI share-price index
     (via FRED, monthly averages, 1955+ for most countries).
  2. Add a smooth dividend accrual from the JST macrohistory database's annual
     dividend/price ratio.
  3. Anchor each calendar year so the twelve months compound exactly to the JST
     annual local-currency equity total return (observed, not modeled). The MEI
     index supplies the within-year shape; JST supplies the annual truth.
  4. Convert to USD with monthly exchange rates: official Bretton Woods parities
     (with the documented devaluation/revaluation steps) before 1971, FRED
     monthly market rates from 1971.
  5. Weight countries by prior-year nominal GDP in USD (JST GDP, own FX).

Countries: Japan, UK, Germany, France, Switzerland, Netherlands, Italy,
Australia - the large majority of developed ex-US market cap in this era.
Each country joins the composite when its monthly index begins (Japan 1949
via the Nikkei; Switzerland and France 1955; Italy and Netherlands 1957; UK
December 1957; Australia 1958; Germany 1960), so the 1955-59 composite is a
subset of the eight, GDP-weighted over whoever has data that month.

Caveats, stated once and honestly: GDP weights are not market-cap weights
(the UK equity market was larger, relative to GDP, than most); MEI indices and
FRED FX are monthly averages, which slightly smooths turning points; dividends
accrue smoothly instead of seasonally. Annual totals are pinned to observed
data; the approximations only shape months within a year and the country mix.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd

from .data import CACHE_DIR, _download, fetch_fred

JST_URL = "https://www.macrohistory.net/app/download/9834512469/JSTdatasetR6.dta"

COUNTRIES = {
    # country: (JST name, MEI share-price FRED id, FX FRED id, fx_is_usd_per_lc)
    "japan": ("Japan", "SPASTT01JPM661N", "EXJPUS", False),
    "uk": ("UK", "SPASTT01GBM661N", "EXUSUK", True),
    "germany": ("Germany", "SPASTT01DEM661N", "EXGEUS", False),
    "france": ("France", "SPASTT01FRM661N", "EXFRUS", False),
    "switzerland": ("Switzerland", "SPASTT01CHM661N", "EXSZUS", False),
    "netherlands": ("Netherlands", "SPASTT01NLM661N", "EXNEUS", False),
    "italy": ("Italy", "SPASTT01ITM661N", "EXITUS", False),
    "australia": ("Australia", "SPASTT01AUM661N", "EXUSAL", True),
}

# Official USD-per-local-currency parities before the 1971 float, as a list of
# (first month in force, rate). Sources: IMF par values; steps are the known
# devaluations/revaluations (GBP Nov 1967; DEM Mar 1961, Oct 1969; FRF Aug 1957,
# Dec 1958, Aug 1969; NLG Mar 1961). Australia held its USD rate through the
# 1967 sterling move. The franc is carried in NEW francs throughout (100 old =
# 1 new from January 1960) so the redenomination is not a 100x "return".
PARITIES = {
    "japan": [("1955-01", 1 / 360.0)],
    "uk": [("1955-01", 2.80), ("1967-11", 2.40)],
    "germany": [("1955-01", 1 / 4.20), ("1961-03", 1 / 4.00), ("1969-10", 1 / 3.66)],
    "france": [("1955-01", 1 / 3.50), ("1957-08", 1 / 4.20), ("1959-01", 1 / 4.93706),
               ("1969-08", 1 / 5.55419)],
    "switzerland": [("1955-01", 1 / 4.37282)],
    "netherlands": [("1955-01", 1 / 3.80), ("1961-03", 1 / 3.62)],
    "italy": [("1955-01", 1 / 625.0)],
    "australia": [("1955-01", 1.12)],
}


# Observed MSCI EAFE annual total returns (USD, gross dividends), percent.
# Source: "Annual Returns of Asset Classes 1970-2009", Wharton course data,
# https://finance.wharton.upenn.edu/~acmack/ret2009.pdf ("Int'l stocks" column;
# 1985 +56.73 / 1986 +69.94 / 2008 -43.06 identify it as MSCI EAFE gross TR).
# The reconstructed composite is re-anchored to these where available, so from
# 1970 on only the within-year monthly shape comes from the reconstruction.
EAFE_ANNUAL_USD = {
    1970: -10.5123, 1971: 31.2065, 1972: 37.5993, 1973: -14.1658,
    1974: -22.1477, 1975: 37.1001, 1976: 3.7407, 1977: 19.4246,
    1978: 34.3005, 1979: 6.1833, 1980: 24.4291, 1981: -1.0325,
    1982: -0.8596, 1983: 24.6092, 1984: 7.8649, 1985: 56.7250,
}


def _load_jst(refresh: bool = False) -> pd.DataFrame:
    blob = _download(JST_URL, "jst_macrohistory.dta", refresh, ua="Mozilla/5.0 (X11; Linux x86_64)")
    return pd.read_stata(io.BytesIO(blob))


def _monthly_fx(key: str, refresh: bool = False) -> pd.Series:
    """USD per local currency, monthly, parity table pre-1971 + FRED from 1971."""
    _, _, fred_id, usd_per_lc = COUNTRIES[key]
    market = fetch_fred(fred_id, refresh)
    if not usd_per_lc:
        market = 1.0 / market
    steps = PARITIES[key]
    start = pd.Period(steps[0][0], freq="M")
    idx = pd.period_range(start, market.index.min() - 1, freq="M")
    fixed = pd.Series(np.nan, index=idx)
    for first, rate in steps:
        fixed[fixed.index >= pd.Period(first, freq="M")] = rate
    return pd.concat([fixed, market]).sort_index()


def _nikkei_monthly(refresh: bool = False) -> pd.Series:
    """Month-end Nikkei 225 closes (FRED daily series, 1949+). Point-to-point,
    unlike the monthly-average MEI indices."""
    blob = _download(
        "https://fred.stlouisfed.org/graph/fredgraph.csv?id=NIKKEI225",
        "fred_NIKKEI225.csv",
        refresh,
    )
    df = pd.read_csv(io.BytesIO(blob), na_values=".")
    df.columns = ["date", "level"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna()
    monthly = df.groupby(df["date"].dt.to_period("M"))["level"].last()
    return monthly.astype(float)


def _japan_local_returns(jst_country: pd.DataFrame, refresh: bool = False) -> pd.Series:
    """Japan monthly local total returns: observed month-end Nikkei price moves
    plus JST dividend accrual. No annual anchoring - JST's Japan equity series
    is annual-average based (it shows 1990 as -13% when the market fell 39%
    point-to-point), so anchoring to it would smear turning points; the Nikkei
    itself is already observed point-to-point data."""
    price_ret = _nikkei_monthly(refresh).pct_change().dropna()
    dp = jst_country.set_index("year")["eq_dp"].ffill()
    accrual = dp.reindex(price_ret.index.year).to_numpy()
    accrual = (1 + np.nan_to_num(accrual, nan=float(dp.iloc[-1]))) ** (1 / 12) - 1
    return price_ret + accrual


def _anchored_local_returns(
    price_index: pd.Series, jst_country: pd.DataFrame
) -> pd.Series:
    """Monthly local-currency total returns whose calendar-year compounding
    matches the JST annual total return exactly (pro-rated for partial years)."""
    price_ret = price_index.pct_change().dropna()
    dp = jst_country.set_index("year")["eq_dp"].ffill()
    eq_tr = jst_country.set_index("year")["eq_tr"]

    out = {}
    for year, target in eq_tr.dropna().items():
        months = price_ret[price_ret.index.year == year]
        if len(months) == 0:
            continue
        accrual = (1 + dp.get(year, dp.iloc[-1])) ** (1 / 12) - 1
        raw = (1 + months + accrual).to_numpy()
        n = len(months)
        target_n = (1 + target) ** (n / 12)  # pro-rate a partial first year
        k = (target_n / raw.prod()) ** (1 / n)
        for period, gross in zip(months.index, raw * k):
            out[period] = gross - 1
    return pd.Series(out).sort_index()


def reconstruct_intl(
    last_year: int = 1983, first_year: int = 1955, refresh: bool = False
) -> pd.Series:
    """GDP-weighted developed ex-US USD monthly total returns through last_year."""
    jst = _load_jst(refresh)

    usd_returns = {}
    gdp_usd = {}
    for key, (jst_name, mei_id, _, _) in COUNTRIES.items():
        country = jst[jst["country"] == jst_name]
        if key == "japan":
            local = _japan_local_returns(country, refresh)
        else:
            local = _anchored_local_returns(fetch_fred(mei_id, refresh), country)
        fx = _monthly_fx(key, refresh)
        fx_ret = fx.pct_change()
        total = ((1 + local) * (1 + fx_ret.reindex(local.index)) - 1).dropna()
        usd_returns[key] = total[
            (total.index.year >= first_year) & (total.index.year <= last_year)
        ]

        # JST's nominal `gdp` column has inconsistent units across countries
        # (trillions of yen, millions of francs, ...), so weight by Maddison
        # real GDP instead: per-capita real GDP x population, consistent units.
        c = country.set_index("year")
        gdp_usd[key] = (c["rgdpmad"] * c["pop"]).dropna()

    returns = pd.DataFrame(usd_returns)
    weights_by_year = pd.DataFrame(gdp_usd)

    out = {}
    for period, row in returns.iterrows():
        if period.year - 1 not in weights_by_year.index:
            continue
        w = weights_by_year.loc[period.year - 1]  # prior-year GDP: known ex ante
        w = w[row.notna()].dropna()
        if w.sum() == 0 or row.isna().all():
            continue
        w = w / w.sum()
        out[period] = float((row[w.index] * w).sum())
    composite = pd.Series(out, name="intl_equities").sort_index()
    return _anchor_to_eafe(composite)


def _anchor_to_eafe(composite: pd.Series) -> pd.Series:
    """Scale each calendar year with a published EAFE annual return so the
    months compound exactly to the observed number (full years only)."""
    adjusted = composite.copy()
    for year, pct in EAFE_ANNUAL_USD.items():
        months = composite[composite.index.year == year]
        if len(months) != 12:
            continue
        gross = (1 + months).to_numpy()
        k = ((1 + pct / 100.0) / gross.prod()) ** (1 / 12)
        adjusted[months.index] = gross * k - 1
    return adjusted


def intl_dividend_yield(refresh: bool = False) -> pd.Series:
    """Monthly dividend accrual for the international composite: mean of the
    eight countries' JST dividend/price ratios, divided by 12. Annual data held
    constant within each year; the last observation carries forward."""
    jst = _load_jst(refresh)
    names = [v[0] for v in COUNTRIES.values()]
    dp = (
        jst[jst["country"].isin(names)]
        .pivot_table(index="year", columns="country", values="eq_dp")
        .mean(axis=1)
        .dropna()
    )
    idx = pd.period_range(f"{int(dp.index.min())}-01", f"{int(dp.index.max())}-12", freq="M")
    monthly = pd.Series(
        dp.reindex([p.year for p in idx]).to_numpy() / 12, index=idx,
        name="income_intl_equities",
    )
    return monthly


def validation_report(refresh: bool = False) -> list[dict]:
    """Out-of-sample check: extend the reconstruction past its 1985 use-by date
    (where the EAFE anchor no longer applies) and compare it against the two
    observed series the panel actually uses from 1986 on."""
    from .data import fetch_aqr_global_ex_us, fetch_developed_ex_us, fetch_us_factors

    recon = reconstruct_intl(last_year=1995, refresh=refresh)
    us = fetch_us_factors(refresh)
    aqr = (fetch_aqr_global_ex_us(refresh) + us["rf"]).dropna()
    dev = fetch_developed_ex_us(refresh)

    reports = []
    for label, actual, start, end in [
        ("AQR Global ex USA", aqr, "1986-01", "1990-06"),
        ("French Developed ex US", dev, "1990-07", "1995-12"),
    ]:
        r = recon[pd.Period(start, freq="M") : pd.Period(end, freq="M")]
        both = pd.concat([r.rename("recon"), actual.rename("actual")], axis=1).dropna()
        ann_r = (1 + both["recon"]).prod() ** (12 / len(both)) - 1
        ann_a = (1 + both["actual"]).prod() ** (12 / len(both)) - 1
        reports.append(
            {
                "versus": label,
                "window": f"{both.index.min()}..{both.index.max()}",
                "months": len(both),
                "monthly_corr": float(both["recon"].corr(both["actual"])),
                "annualized_recon": float(ann_r),
                "annualized_actual": float(ann_a),
                "annualized_gap": float(ann_r - ann_a),
                "tracking_error_ann": float(
                    (both["recon"] - both["actual"]).std() * np.sqrt(12)
                ),
            }
        )
    return reports
