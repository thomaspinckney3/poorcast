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
    "1955-85 anchored to observed EAFE/JST annuals; AQR Global ex USA 1986-90; "
    "Ken French Developed ex US 1990+)",
    "us_bonds_10yr": "10-year US Treasuries (total return derived from FRED GS10 yields; "
    "the Fed long-term composite LTGOVTBD before 1953)",
    "us_bonds_20yr": "Long US Treasuries (20-year constant maturity; total return "
    "derived from FRED GS20, GS30 level-adjusted across the 1987-93 gap when the "
    "20-year was not issued, the Fed long-term composite LTGOVTBD before 1953)",
    "muni_bonds": "Municipal bonds (returns derived from Bond Buyer GO-20 yields "
    "1953-2007, NBER high-grade muni yields 1937-52, a Treasury-ratio proxy before; "
    "observed MUB ETF total returns 2007+; income exempt from federal and state tax)",
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
    "us_bonds_20yr": "interest",
    "cash": "interest",
    "muni_bonds": "muni",  # exempt from federal and (own-state assumption) state
}

# Shiller's data moved from Yale to shillerdata.com in 2023 (the Yale file
# froze at September 2023). The spreadsheet link on the new site carries a
# version token, so the fetch scrapes the page for the current link and
# falls back to the last known one.
SHILLER_PAGE = "https://shillerdata.com/"
SHILLER_URL = (
    "https://img1.wsimg.com/blobby/go/e5e77e0b-59d1-44d9-ab25-4763ac982e53/"
    "downloads/70fec4f5-727f-4e53-b5f1-179af109c5fa/ie_data.xls"
)


def shiller_link_from_page(html: str) -> str | None:
    """The ie_data.xls download link on shillerdata.com, or None."""
    m = re.search(r'href="((?:https?:)?//[^"]*?/ie_data\.xls[^"]*)"', html)
    if not m:
        return None
    url = m.group(1).replace("&amp;", "&")
    return "https:" + url if url.startswith("//") else url


def shiller_workbook(refresh: bool = False) -> bytes:
    """Shiller's ie_data.xls (monthly S&P price, dividends, earnings, CPI,
    long rate, and his CAPE), cached as shiller_ie_data.xls."""
    cached = CACHE_DIR / "shiller_ie_data.xls"
    if cached.exists() and not refresh:
        return cached.read_bytes()
    url = SHILLER_URL
    try:
        req = urllib.request.Request(
            SHILLER_PAGE, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            page = resp.read().decode("utf-8", "replace")
        url = shiller_link_from_page(page) or url
    except Exception:
        pass  # fall back to the last known link
    return _download(url, "shiller_ie_data.xls", True, ua="Mozilla/5.0 (X11; Linux x86_64)")


def shiller_month_index(col: pd.Series) -> tuple[np.ndarray, pd.PeriodIndex]:
    """Decode Shiller's fractional-year date column (1871.01 .. 1871.12).

    Excel hands the column back as floats, so October is 1871.1 - a text
    match on two decimals silently drops every October. Decode numerically:
    returns (mask of data rows, their months)."""
    v = pd.to_numeric(col, errors="coerce")
    year = np.floor(v)
    month = np.round((v - year) * 100)
    ok = v.notna() & (year >= 1800) & (year <= 2200) & (month >= 1) & (month <= 12)
    idx = pd.PeriodIndex(
        [
            pd.Period(year=int(y), month=int(m), freq="M")
            for y, m in zip(year[ok], month[ok])
        ],
        freq="M",
    )
    return ok.to_numpy(), idx


def fetch_shiller_dividend_yield(refresh: bool = False) -> pd.Series:
    """Monthly S&P dividend yield from Shiller's ie_data (D is a 12-month rate,
    so the monthly accrual is D/12 divided by price)."""
    blob = shiller_workbook(refresh)
    df = pd.read_excel(io.BytesIO(blob), sheet_name="Data", header=None, engine="xlrd")
    rows, idx = shiller_month_index(df[0])
    df = df[rows]
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


def splice_yields(primary: pd.Series, secondary: pd.Series, offset: float) -> pd.Series:
    """`primary` where it has data, else `secondary + offset` (a level
    adjustment fitted on the overlap by the caller). Percent yields."""
    out = primary.reindex(primary.index.union(secondary.index))
    fill = (secondary + offset).reindex(out.index)
    return out.where(out.notna(), fill).sort_index()


def treasury_yield_history(refresh: bool = False) -> pd.Series:
    """10-year Treasury yield, percent: FRED GS10 (1953-04+) extended back to
    1925 with the Fed's long-term government composite (LTGOVTBD), which sits
    ~0.10 pt above GS10 on the 1953-55 overlap. Yields were pegged from
    1942 to the March 1951 Accord, so that stretch is artificially calm."""
    gs10 = fetch_fred("GS10", refresh)
    lt = fetch_fred("LTGOVTBD", refresh)
    both = pd.concat([lt.rename("lt"), gs10.rename("gs")], axis=1).dropna()
    both = both[: pd.Period("1955-12", freq="M")]
    offset = float((both["gs"] - both["lt"]).mean()) if len(both) else 0.0
    return splice_yields(gs10, lt, offset).rename("GS10")


def long_treasury_yield_history(refresh: bool = False) -> pd.Series:
    """20-year Treasury yield, percent. FRED GS20 (1953-04+) is the spine; the
    20-year was not issued between 1987 and 1993, and that gap is filled with
    GS30 level-adjusted to GS20 on their overlap. Extended back to 1925 with
    the Fed's long-term government composite (LTGOVTBD), the same series that
    extends the 10-year.

    Twenty years rather than thirty keeps one constant maturity across the
    whole history: GS30 itself only starts in 1977 and was discontinued
    2002-2006, so a 30-year spine would carry two gaps instead of one.
    """
    gs20 = fetch_fred("GS20", refresh)
    gs30 = fetch_fred("GS30", refresh)
    both = pd.concat([gs30.rename("g30"), gs20.rename("g20")], axis=1).dropna()
    offset = float((both["g20"] - both["g30"]).mean()) if len(both) else 0.0
    spine = splice_yields(gs20, (gs30 + offset).dropna(), 0.0)
    lt = fetch_fred("LTGOVTBD", refresh)
    o = pd.concat([lt.rename("lt"), spine.rename("s")], axis=1).dropna()
    o = o[: pd.Period("1955-12", freq="M")]
    off2 = float((o["s"] - o["lt"]).mean()) if len(o) else 0.0
    return splice_yields(spine, lt, off2).rename("GS20")


def muni_yield_history(refresh: bool = False) -> pd.Series:
    """Bond Buyer 20-bond yield, percent (FRED MSLB20, 1953-01..2016-09),
    extended back to 1925: the NBER Macrohistory high-grade municipal series
    (M13043USM156NNBR, 1937-1966; ~0.22 pt below Bond Buyer 20 on the
    1953-66 overlap, level-adjusted) for 1937-52, and before 1937 the long
    Treasury yield scaled by the 1937-52 average muni/Treasury ratio - a
    proxy, since no free monthly muni series reaches the 1920s."""
    bb = fetch_fred("MSLB20", refresh)
    nber = fetch_fred("M13043USM156NNBR", refresh)
    both = pd.concat([nber.rename("nb"), bb.rename("bb")], axis=1).dropna()
    offset = float((both["bb"] - both["nb"]).mean()) if len(both) else 0.0
    muni = splice_yields(bb, nber, offset)
    tsy = treasury_yield_history(refresh)
    o = pd.concat([muni.rename("m"), tsy.rename("t")], axis=1).dropna()
    o = o[pd.Period("1937-01", freq="M"): pd.Period("1952-12", freq="M")]
    ratio = float((o["m"] / o["t"]).mean()) if len(o) else 0.9
    pre = (tsy[: pd.Period("1936-12", freq="M")] * ratio)
    return splice_yields(muni, pre, 0.0).rename("MSLB20")


def fetch_muni_returns(refresh: bool = False) -> tuple[pd.Series, pd.Series]:
    """Monthly muni total returns and income yields: derived from Bond Buyer
    GO-20 yields (extended to 1925, see muni_yield_history) through 2007,
    observed MUB ETF from 2007 on."""
    yields = muni_yield_history(refresh)  # percent, monthly
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



def house_price_index(refresh: bool = False) -> pd.Series:
    """Monthly US house price index (nominal), 1890 to date.

    Case-Shiller national (FRED CSUSHPINSA) is the monthly spine from 1987.
    FHFA (USSTHPI, quarterly) covers 1975-86 and the Jorda-Schularick-Taylor
    annual series covers 1890-1974; both are interpolated geometrically to
    monthly and level-adjusted at each splice.

    Interpolation invents within-period smoothness, which would matter for an
    asset that is rebalanced or sold, because the path decides what gets
    realised. It does not matter for a residence held to the end of the
    horizon: only the cumulative return is consumed, and interpolation
    preserves each period's endpoints exactly. What the splice does buy is
    joint sampling - the housing months come from the same historical months
    as every other series, so a bad housing outcome lands in the same drawn
    path as the market conditions that produced it.
    """
    import io as _io

    cs = fetch_fred("CSUSHPINSA", refresh).dropna()
    fh = fetch_fred("USSTHPI", refresh).dropna()

    from .reconstruct import _load_jst

    jst = _load_jst(refresh)
    us = jst[jst["country"] == "USA"].set_index("year")["hpnom"].dropna()
    ann = pd.Series(
        us.to_numpy(),
        index=pd.PeriodIndex([f"{int(y)}-12" for y in us.index], freq="M"),
    )

    def to_monthly(idx: pd.Series) -> pd.Series:
        """Geometric interpolation onto a monthly grid, endpoints preserved."""
        full = pd.period_range(idx.index[0], idx.index[-1], freq="M")
        return np.exp(
            np.log(idx.astype(float)).reindex(full).interpolate("index")
        ).rename("hpi")

    fh_m, ann_m = to_monthly(fh), to_monthly(ann)

    def splice(base: pd.Series, earlier: pd.Series) -> pd.Series:
        """Scale `earlier` to meet `base` at their first common month."""
        common = base.index.intersection(earlier.index)
        if len(common) == 0:
            return base
        at = common.min()
        scaled = earlier * (float(base.loc[at]) / float(earlier.loc[at]))
        return pd.concat([scaled[scaled.index < at], base]).sort_index()

    return splice(splice(cs, fh_m), ann_m).rename("house_price_index")


def fetch_house_returns(refresh: bool = False) -> pd.Series:
    """Monthly nominal capital-gain returns on US housing.

    Capital gain only, not total return: an owner-occupier consumes the rent
    yield by living there, so it never accrues to the estate.
    """
    return house_price_index(refresh).pct_change().dropna().rename("us_housing")

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

    gs10 = treasury_yield_history(refresh)
    bonds = bond_returns_from_yields(gs10)
    housing = fetch_house_returns(refresh)
    gs20 = long_treasury_yield_history(refresh)
    long_bonds = bond_returns_from_yields(gs20, maturity_years=20).rename(
        "us_bonds_20yr"
    )
    munis, muni_income = fetch_muni_returns(refresh)

    # Seasonally adjusted CPI from 1947; the unadjusted index (1913+) before
    # that, so the 1926-46 months carry some seasonal noise (sd 0.50%/mo vs
    # 0.44 for the adjusted series on their overlap).
    cpi_sa = fetch_fred("CPIAUCSL", refresh)
    cpi_nsa = fetch_fred("CPIAUCNS", refresh)
    inflation = pd.concat([
        cpi_nsa.pct_change().dropna()[: cpi_sa.index.min() - 1],
        cpi_sa.pct_change().dropna(),
    ]).sort_index().rename("inflation")

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
        "income_us_bonds_20yr": (gs20.shift(1) / 100 / 12),
        "income_muni_bonds": muni_income,
        "income_cash": us["rf"],
    }

    # data/custom/<asset>.csv overrides any built-in series where it has values.
    series = [
        _apply_custom_override(str(s.name), s)
        for s in (us_eq, small, intl, bonds, long_bonds, munis, cash)
    ]
    panel = pd.concat(series + [inflation, housing], axis=1).sort_index()
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
