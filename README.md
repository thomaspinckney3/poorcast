# poorcast

Monte Carlo portfolio forecasting driven by **actual market history**. Give it an
asset mix and (optionally) a withdrawal strategy; it resamples real historical
monthly returns — jointly across assets and inflation, so correlations and
regimes like the 1970s survive — and simulates thousands of outcomes with
**quarterly rebalancing**.

## Quick start

```bash
python -m venv .venv && .venv/bin/pip install -e .
.venv/bin/poorcast fetch          # download historical data from source (~1 min)
.venv/bin/poorcast assets         # list asset classes and coverage

.venv/bin/poorcast run \
    --allocation us_equities=50,intl_equities=20,us_bonds_10yr=25,cash=5 \
    --initial 1000000 --withdraw 3% --horizons 20,30,40
```

Each run prints success rates and terminal-wealth percentiles per horizon and
writes a chart (percentile fan, solvency curve, terminal-wealth distribution)
to `out/forecast_<N>y.png`.

## Asset classes

| name | coverage | source |
|---|---|---|
| `us_equities` | 1926+ | CRSP value-weighted total market (Ken French library) |
| `us_small_cap` | 1926+ | bottom 30% by market cap, value-weighted (Ken French) |
| `intl_equities` | 1960+ | reconstructed 8-country composite (1960–85), AQR Global ex USA (1986–90), Ken French Developed ex US (1990+) |
| `us_bonds_10yr` | 1953+ | 10-yr Treasury total return derived from FRED GS10 yields |
| `muni_bonds` | 1953+ | Bond Buyer GO-20 yields priced at 10y maturity (vol-matched to MUB) through 2007, observed MUB ETF total returns after |
| `cash` | 1926+ | 1-month T-bill (Ken French) |

US CPI (FRED `CPIAUCSL`) is carried alongside and sampled jointly, driving
inflation-adjusted withdrawals and real-dollar reporting.

All returns are monthly, USD, total return. Data is fetched straight from the
publishers by `poorcast fetch` and cached under `data/`; nothing is bundled.

**How pre-1986 international is built.** MSCI EAFE begins Dec 1969 and MSCI
does not allow free download of pre-1997 history, so the 1960–1985 segment is
reconstructed from primary national data (see `poorcast/reconstruct.py`):
monthly local-currency price indices for Japan, UK, Germany, France,
Switzerland, Netherlands, Italy and Australia (OECD MEI via FRED; month-end
Nikkei closes for Japan), dividend accrual and annual local total-return
anchors from the Jordà-Schularick-Taylor macrohistory database, USD conversion
via official Bretton Woods parities (with the documented devaluation steps)
before 1971 and FRED monthly rates after, GDP-weighted. Calendar years
1970–1985 are additionally anchored to published MSCI EAFE annual USD total
returns, so from 1970 on only the *within-year monthly shape* is
reconstructed; annual totals are observed data. Run `poorcast validate-intl`
for the out-of-sample check against observed 1986–1995 series (monthly
correlation 0.81–0.88, annualized gap +1.5 to +2.9%/yr — the reconstruction
runs slightly hot; the smoothing comes from OECD indices being monthly
averages). Known caveats: GDP weights are not market-cap weights, and the
eight countries are most but not all of EAFE. If you have licensed EAFE data,
drop it in `data/custom/intl_equities.csv` (`month,return` rows like
`1970-01,0.023`) and rebuild with `poorcast fetch` — it overrides the built-in
series wherever it has values.

## How the simulation works

- **Sampling** – circular block bootstrap over the historical months (default
  24-month blocks, `--block`). The same sampled months are used for every asset
  and for inflation, preserving cross-asset correlation and serial structure.
  `--block 1` gives classic independent-month resampling.
- **`--mode historical`** – instead of resampling, runs every actual start month
  in the record as one path ("what if I had retired in May 1968?").
- **Monthly steps** – withdrawal/contribution at the start of each month
  (pro-rata across holdings), then market returns, then rebalancing back to
  target weights at each quarter end.
- **Withdrawal strategies**
  - `--withdraw 3%` — the classic percent rule: 3% of the *initial* balance per
    year, inflation-adjusted along each path, in monthly installments.
  - `--withdraw 40000` — fixed dollars per year, inflation-adjusted.
  - `--withdraw 3% --withdraw-strategy percent-of-balance` — 3% of the *current*
    balance, recomputed yearly (spending floats with the market; never depletes).
  - `--tips-ladder 30000` — buy a TIPS ladder at t=0 paying $30k/yr real for the
    whole horizon, held to maturity outside the portfolio (so it needs no
    historical TIPS return series — only the purchase-date real yield,
    `--tips-ladder-yield`, default 2%). Its cost comes off the starting balance
    and its income off the withdrawal; if the residual portfolio depletes,
    income falls to the ladder floor instead of $0. Bracket the yield with 0-2%
    — ladders bought at low real yields are poor value.
  - `--flex [FLOOR_PCT]` — belt-tightening in down markets (fixed-real only):
    each month the withdrawal is scaled by current real balance / initial
    balance, capped at the full target and floored at FLOOR_PCT% of it
    (default 75). Costs little income in the median case but sharply cuts
    tail-risk of depletion.
- **Contributions** – `--contribute 2000` adds $2,000/month in today's dollars
  (grown with inflation) for accumulation scenarios.
- **Taxes** – by default the portfolio is **fully taxable under 2026 US
  federal brackets** (Rev. Proc. 2025-32), settled annually: interest (bond
  coupons, T-bills) through the ordinary brackets, qualified dividends and
  net realized long-term gains through the 0/15/20% brackets stacked on top,
  standard deduction, and 3.8% NIIT. `--filing married` switches from the
  single-filer default; bracket edges and the deduction are indexed to each
  path's simulated inflation (as the law indexes them), while the NIIT
  threshold stays nominal (statutory). Realized gains come from true
  average-cost basis tracking — rebalancing trades, withdrawal liquidations,
  and tax payments all realize gains, netted with loss carryforward.
  `--cost-basis 0.6` starts with embedded gains (basis = 60% of value).
  `--state-tax 5` (the default) adds a flat state income tax on dividends
  and realized gains — Treasury interest is constitutionally state-exempt,
  and `muni_bonds` income is exempt from both levels (own-state assumption);
  muni *capital gains* are taxed normally. `--tax-deferred` turns taxes off
  (IRA/401k); `--tax-rate 15 --tax-ordinary 24` overrides brackets with flat
  rates settled quarterly.
  **Withdrawals spend income first**: each month's dividends/interest fund
  the withdrawal as cash (no sale, no realized gain); only the shortfall
  sells holdings. Unspent income is reinvested (with its basis stepped up).
  Income components are observed data: Shiller monthly dividend yields for
  US equities (also the small-cap proxy), JST dividend/price for
  international, GS10 coupon accrual for bonds. Not modeled: state taxes,
  other (non-portfolio) income, short-term gain rates, the $3,000 loss
  offset, step-up at death; terminal wealth is pre-liquidation.
- **Return-source assumptions** – `poorcast decompose` splits historical US
  equity returns into dividends, inflation, real EPS growth (margin expansion
  vs underlying, via NIPA CP/GDP) and P/E multiple expansion (Shiller data).
  `--multiple-expansion 0` reruns the simulation assuming no future multiple
  expansion: sampled US equity returns are shifted by (assumed − historical
  ≈ 0.5%/yr); negative values model multiple compression. Applies to
  `us_equities`/`us_small_cap` only.
- Results are reported in **real (today's) dollars** by default; `--nominal`
  switches that off. `--seed` makes runs reproducible.

## Caveats

This is a research toy, not financial advice. Bootstrapping assumes the future
is drawn from the same distribution as 1960–present; it ignores taxes, fees,
and mean reversion beyond the block length. The bond series is derived from
constant-maturity yields via a standard pricing approximation. Success rates
above ~95% are not distinguishable from each other given history this short.

## Development

```bash
.venv/bin/python -m pytest tests/
```
