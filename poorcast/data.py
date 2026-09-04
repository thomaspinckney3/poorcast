"""Data layer: fetch historical monthly returns from primary sources and build a joint panel.

Sources (all free, fetched directly from the publisher):
  - Ken French data library: US market total return (CRSP, 1926+), size portfolios
    (small cap, 1926+), Developed ex US market (1990+), 1-month T-bill (cash).
  - AQR data library ("Betting Against Beta" dataset): Global ex USA market excess
    return, monthly 1982+. Used to extend international coverage before 1990.
  - FRED: GS10 (10-year Treasury constant-maturity yield, 1953+) from which bond
    total returns are computed; CPIAUCSL (CPI, monthly inflation).

The result is a single monthly DataFrame (PeriodIndex, freq='M') with one column
per asset class plus 'inflation', saved to data/returns.csv. All values are simple
monthly returns in USD (0.01 = 1%).
"""

from __future__ import annotations

import io
import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_DIR = DATA_DIR / "cache"
PANEL_PATH = DATA_DIR / "returns.csv"
CUSTOM_DIR = DATA_DIR / "custom"

FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
AQR_BAB_URL = (
    "https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/"
    "Betting-Against-Beta-Equity-Factors-Monthly.xlsx"
)
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"

ASSET_DESCRIPTIONS = {
    "us_equities": "US total stock market (CRSP value-weighted, via Ken French)",
    "us_small_cap": "US small caps (bottom 30% by market cap, value-weighted, via Ken French)",
    "intl_equities": "International developed ex-US (reconstructed 8-country composite "
    "1960-85 anchored to observed EAFE/JST annuals; AQR Global ex USA 1986-90; "
    "Ken French Developed ex US 1990+)",
    "us_bonds_10yr": "10-year US Treasuries (total return derived from FRED GS10 yields)",
    "muni_bonds": "Municipal bonds (returns derived from Bond Buyer GO-20 yields "
    "1953-2007, observed MUB ETF total returns 2007+; income exempt from federal "
    "and state tax)",
    "cash": "1-month US T-bills (via Ken French)",
}


def _download(url: str, cache_name: str, refresh: bool = False, ua: str = "curl/8.5.0") -> bytes:
    # FRED's CDN times out browser-like Python requests but serves curl UAs;
    # AQR's CDN wants a browser UA. Hence the per-source `ua`.
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = CACHE_DIR / cache_name
    if cached.exists() and not refresh:
        return cached.read_bytes()
    req = urllib.request.Request(url, headers={"User-Agent": ua})
    with urllib.request.urlopen(req, timeout=60) as resp:
        blob = resp.read()
    cached.write_bytes(blob)
    return blob


def _french_csv(zip_name: str, refresh: bool = False) -> str:
    blob = _download(FRENCH_BASE + zip_name, zip_name, refresh)
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        inner = zf.namelist()[0]
        return zf.read(inner).decode("latin-1")


def _parse_french_monthly(text: str, columns: list[str]) -> pd.DataFrame:
    """Parse the first monthly table of a Ken French CSV (rows keyed YYYYMM)."""
    rows = {}
    for line in text.splitlines():
        cells = [c.strip() for c in line.split(",")]
        if re.fullmatch(r"\d{6}", cells[0] or ""):
            rows[cells[0]] = [float(c) for c in cells[1 : len(columns) + 1]]
        elif rows and not re.fullmatch(r"\d{6}", cells[0] or ""):
            break  # first monthly table ended (annual tables etc. follow)
    df = pd.DataFrame.from_dict(rows, orient="index", columns=columns)
    df.index = pd.PeriodIndex(pd.to_datetime(df.index, format="%Y%m"), freq="M")
    df = df.replace([-99.99, -999], np.nan)
    return df / 100.0  # French data is in percent


def fetch_us_factors(refresh: bool = False) -> pd.DataFrame:
    """US market excess return and risk-free rate, monthly 1926+ -> columns mkt_rf, rf."""
    text = _french_csv("F-F_Research_Data_Factors_CSV.zip", refresh)
    df = _parse_french_monthly(text, ["mkt_rf", "smb", "hml", "rf"])
    return df[["mkt_rf", "rf"]]


def fetch_us_small_cap(refresh: bool = False) -> pd.Series:
    """Value-weighted return of the bottom-30%-by-size portfolio, monthly 1926+."""
    text = _french_csv("Portfolios_Formed_on_ME_CSV.zip", refresh)
    # The first monthly table is the value-weighted one; its header row names the
    # portfolio columns. 'Lo 30' is the small-cap 30%.
    header = None
    for line in text.splitlines():
        cells = [c.strip() for c in line.split(",")]
        if header is None and "Lo 30" in cells:
            header = cells
            continue
        if header is not None and re.fullmatch(r"\d{6}", cells[0] or ""):
            break
    if header is None:
        raise ValueError("could not locate 'Lo 30' header in Portfolios_Formed_on_ME")
    df = _parse_french_monthly(text[text.index(",".join(header)) :], header[1:])
    return df["Lo 30"].rename("us_small_cap")


def fetch_developed_ex_us(refresh: bool = False) -> pd.Series:
    """Developed ex US market total return (Mkt-RF + RF), monthly 1990-07+."""
    text = _french_csv("Developed_ex_US_3_Factors_CSV.zip", refresh)
    df = _parse_french_monthly(text, ["mkt_rf", "smb", "hml", "rf"])
    return (df["mkt_rf"] + df["rf"]).rename("intl_equities")


def fetch_aqr_global_ex_us(refresh: bool = False) -> pd.Series:
    """AQR Global ex USA market excess return, monthly 1982+ (decimal, excess of T-bill)."""
    import openpyxl

    blob = _download(AQR_BAB_URL, "aqr_bab_monthly.xlsx", refresh, ua="Mozilla/5.0 (X11; Linux x86_64)")
    wb = openpyxl.load_workbook(io.BytesIO(blob), read_only=True)
    ws = wb["MKT"]
    rows = list(ws.iter_rows(values_only=True))
    hdr_i = next(i for i, r in enumerate(rows) if r[0] == "DATE")
    hdr = [str(h).strip() if h else "" for h in rows[hdr_i]]
    col = next(i for i, h in enumerate(hdr) if h.lower().startswith("global ex"))
    out = {}
    for r in rows[hdr_i + 1 :]:
        if r[0] is None:
            continue
        val = r[col]
        if val is None or val == "":
            continue
        out[pd.Period(pd.to_datetime(str(r[0])), freq="M")] = float(val)
    return pd.Series(out, name="global_ex_us_excess").sort_index()


# Tax character of each asset's income, for taxable-account modeling:
# dividends (equities) vs ordinary interest (bonds, bills).
INCOME_CLASS = {
    "us_equities": "dividend",  # federal preferential rate; state-taxable
    "us_small_cap": "dividend",
    "intl_equities": "dividend",
    "us_bonds_10yr": "interest",  # federal ordinary rate; STATE-EXEMPT (Treasury)
    "cash": "interest",
    "muni_bonds": "muni",  # exempt from federal and (own-state assumption) state
}

SHILLER_URL = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"


def fetch_shiller_dividend_yield(refresh: bool = False) -> pd.Series:
    """Monthly S&P dividend yield from Shiller's ie_data (D is a 12-month rate,
    so the monthly accrual is D/12 divided by price)."""
    blob = _download(SHILLER_URL, "shiller_ie_data.xls", refresh, ua="Mozilla/5.0")
    df = pd.read_excel(io.BytesIO(blob), sheet_name="Data", header=None, engine="xlrd")
    date_col = df[0].astype(str)
    rows = date_col.str.fullmatch(r"\d{4}\.\d{2}")
    df = df[rows]
    # 1871.1 means October (Shiller's fractional format); zero-pad handled by regex
    idx = pd.PeriodIndex(
        [f"{d[:4]}-{d[5:7]}" for d in df[0].astype(str)], freq="M"
    )
    p = pd.to_numeric(df[1], errors="coerce")
    d = pd.to_numeric(df[2], errors="coerce")
    out = pd.Series((d / 12 / p).to_numpy(), index=idx, name="shiller_div_yield")
    return out.dropna()


def fetch_fred(series: str, refresh: bool = False) -> pd.Series:
    blob = _download(FRED_CSV.format(series=series), f"fred_{series}.csv", refresh)
    df = pd.read_csv(io.BytesIO(blob), na_values=".")
    df.columns = ["date", series]
    df.index = pd.PeriodIndex(pd.to_datetime(df["date"]), freq="M")
    return df[series].astype(float)


def fetch_yahoo_monthly(symbol: str, refresh: bool = False) -> pd.DataFrame:
    """Monthly adjusted-close total returns and distribution yields for an ETF
    (Yahoo v8 chart API). Columns: ret, div_yield. First and current (partial)
    months dropped."""
    import json

    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        "?range=30y&interval=1mo&events=div"
    )
    blob = _download(url, f"yahoo_{symbol}.json", refresh, ua="Mozilla/5.0")
    data = json.loads(blob)["chart"]["result"][0]
    ts = pd.to_datetime(data["timestamp"], unit="s")
    idx = pd.PeriodIndex(ts, freq="M")
    adj = pd.Series(data["indicators"]["adjclose"][0]["adjclose"], index=idx)
    close = pd.Series(data["indicators"]["quote"][0]["close"], index=idx)
    adj = adj[~adj.index.duplicated(keep="last")].dropna()
    close = close[~close.index.duplicated(keep="last")].dropna()
    divs = pd.Series(0.0, index=adj.index)
    for d in (data.get("events", {}).get("dividends", {}) or {}).values():
        per = pd.Period(pd.to_datetime(d["date"], unit="s"), freq="M")
        if per in divs.index:
            divs[per] += d["amount"]
    out = pd.DataFrame(
        {"ret": adj.pct_change(), "div_yield": divs / close}
    ).dropna()
    current = pd.Period(pd.Timestamp.now(), freq="M")
    return out[out.index < current]


def fetch_muni_returns(refresh: bool = False) -> tuple[pd.Series, pd.Series]:
    """Monthly muni total returns and income yields: derived from Bond Buyer
    GO-20 yields (FRED MSLB20, 1953-2007), observed MUB ETF from 2007 on."""
    yields = fetch_fred("MSLB20", refresh)  # percent, monthly, ends 2016-09
    # Priced at the index's actual 20-year maturity: earning 20-year yield
    # carry on a shorter-priced bond would systematically flatter the series.
    # The cost of consistency is a duration break at the 2007 MUB splice
    # (MUB runs ~6y duration), documented in the README.
    derived = bond_returns_from_yields(yields, maturity_years=20).rename("muni_bonds")
    derived_income = (yields.shift(1) / 100 / 12).dropna()
    mub = fetch_yahoo_monthly("MUB", refresh)
    splice = mub.index.min()
    ret = pd.concat([derived[derived.index < splice], mub["ret"]]).sort_index()
    income = pd.concat(
        [derived_income[derived_income.index < splice], mub["div_yield"]]
    ).sort_index()
    return ret.rename("muni_bonds"), income.rename("income_muni_bonds")


def bond_returns_from_yields(yields: pd.Series, maturity_years: int = 10) -> pd.Series:
    """Monthly total returns of a constant-maturity par bond from a yield series.

    Each month buy an N-year annual-coupon par bond (coupon = last month's
    yield); a month later value it exactly: the dirty price discounts every
    remaining cash flow (coupons at 11/12, 1+11/12, ..., principal at
    N - 1/12 years) at this month's yield, which factors as
    (1+y)^(1/12) x the at-issue price at the new yield. Return = price - 1.
    (An earlier simple-accrual approximation ran ~15-20bp/yr hot.)
    """
    y = yields / 100.0
    coupon = y.shift(1)
    n = maturity_years
    # at-issue price of bond with annual coupon c, yield y, maturity n years
    with np.errstate(invalid="ignore"):
        p0 = coupon / y * (1 - (1 + y) ** -n) + (1 + y) ** -n
        ret = (1 + y) ** (1 / 12) * p0 - 1
    return ret.rename("us_bonds_10yr").dropna()


def build_panel(refresh: bool = False) -> pd.DataFrame:
    """Fetch everything and assemble the monthly joint panel."""
    us = fetch_us_factors(refresh)
    us_eq = (us["mkt_rf"] + us["rf"]).rename("us_equities")
    cash = us["rf"].rename("cash")
    small = fetch_us_small_cap(refresh)

    dev = fetch_developed_ex_us(refresh)
    aqr_excess = fetch_aqr_global_ex_us(refresh)
    aqr_total = (aqr_excess + us["rf"]).dropna().rename("intl_equities")
    from .reconstruct import reconstruct_intl  # deferred: avoids circular import

    recon = reconstruct_intl(last_year=1985, refresh=refresh)
    # Splice, preferring observed data: reconstruction through 1985 (AQR's
    # 1984-85 aggregate is effectively Canada-only), AQR 1986 to mid-1990,
    # French Developed ex US from 1990-07 on.
    aqr_start = pd.Period("1986-01", freq="M")
    intl = pd.concat(
        [
            recon[recon.index < aqr_start],
            aqr_total[(aqr_total.index >= aqr_start) & (aqr_total.index < dev.index.min())],
            dev,
        ]
    ).sort_index()
    intl = _apply_custom_override("intl_equities", intl)

    gs10 = fetch_fred("GS10", refresh)
    bonds = bond_returns_from_yields(gs10)
    munis, muni_income = fetch_muni_returns(refresh)

    cpi = fetch_fred("CPIAUCSL", refresh)
    inflation = cpi.pct_change().rename("inflation").dropna()

    # Income-yield columns (monthly accrual rates) for taxable-account modeling.
    # Bonds/cash decompose exactly from the yield data; equity dividend yields
    # are observed (Shiller monthly for US, also used as the small-cap proxy;
    # JST country dividend/price for international), forward-filled past each
    # source's end since yields move slowly.
    div_yield = fetch_shiller_dividend_yield(refresh)
    from .reconstruct import intl_dividend_yield  # deferred: avoids circular import

    income = {
        "income_us_equities": div_yield,
        "income_us_small_cap": div_yield.rename("income_us_small_cap"),
        "income_intl_equities": intl_dividend_yield(refresh),
        "income_us_bonds_10yr": (gs10.shift(1) / 100 / 12),
        "income_muni_bonds": muni_income,
        "income_cash": us["rf"],
    }

    panel = pd.concat(
        [us_eq, small, intl, bonds, munis, cash, inflation], axis=1
    ).sort_index()
    for name, series in income.items():
        panel[name] = series.reindex(panel.index).ffill()
    panel.index.name = "month"

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_csv(PANEL_PATH)
    return panel


def _apply_custom_override(asset: str, series: pd.Series) -> pd.Series:
    """If data/custom/<asset>.csv exists (columns: month,return with YYYY-MM rows,
    decimal returns), it replaces the built-in series wherever it has values."""
    path = CUSTOM_DIR / f"{asset}.csv"
    if not path.exists():
        return series
    custom = pd.read_csv(path)
    custom.index = pd.PeriodIndex(custom.iloc[:, 0], freq="M")
    override = custom.iloc[:, 1].astype(float)
    merged = series.reindex(series.index.union(override.index))
    merged.loc[override.index] = override
    return merged.rename(series.name)


def load_panel(refresh: bool = False) -> pd.DataFrame:
    if PANEL_PATH.exists() and not refresh:
        df = pd.read_csv(PANEL_PATH)
        df.index = pd.PeriodIndex(df["month"], freq="M")
        return df.drop(columns=["month"])
    return build_panel(refresh)


def coverage(panel: pd.DataFrame) -> pd.DataFrame:
    """First/last month with data for each column."""
    rows = []
    for col in panel.columns:
        s = panel[col].dropna()
        rows.append({"series": col, "first": str(s.index.min()), "last": str(s.index.max())})
    return pd.DataFrame(rows).set_index("series")
