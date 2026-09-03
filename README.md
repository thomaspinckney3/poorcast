# poorcast

Monte Carlo retirement and portfolio forecasting driven by **actual market
history**, not normal-distribution assumptions. Give it an asset mix and
(optionally) a withdrawal strategy; it resamples real historical monthly
returns — jointly across assets and inflation, so correlations and regimes
like the 1970s survive — and simulates thousands of possible futures, with
realistic rebalancing and US taxes.

Use it to answer questions like:

- *Can I retire on this portfolio with a 4% withdrawal rate?*
- *How much does cutting spending in down markets improve my odds?*
- *What if I buy a TIPS ladder to cover my floor expenses?*
- *Which asset mix maximizes my chance of never running out?*

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/thomaspinckney3/poorcast.git
cd poorcast
python -m venv .venv && .venv/bin/pip install -e .
.venv/bin/poorcast fetch     # download historical data from source (~1 min)
```

`fetch` pulls monthly return history (1926+) straight from the publishers (Ken
French library, FRED, AQR, Shiller) and caches it under `data/`; nothing is
bundled with the repo. Run `poorcast assets` to see the asset classes and
their coverage.

## Quick start

A classic question: $1M, 60/35/5 stocks/bonds/cash, withdrawing an
inflation-adjusted 4% a year — does the money last 30 years?

```bash
.venv/bin/poorcast run \
    --allocation us_equities=60,us_bonds_10yr=35,cash=5 \
    --initial 1000000 --withdraw 4% --horizons 30 --seed 42
```

```
=== 30-year horizon ===
  60% us_equities, 35% us_bonds_10yr, 5% cash · start $1,000,000 · withdrawing 4.0%/yr
  of initial (inflation-adjusted) · taxable, federal brackets (single filer, 2026 law,
  indexed) + 5% state · quarterly rebalancing · 10,000 bootstrap paths (block=24mo) ·
  history 1960-01–2026-06
  Success rate (never depleted): 92.1%
  Failed paths (7.9%): earliest failure year 13, median year 25, 90% fail after year 20
  Median total income withdrawn (real): $1.2M
  Median total taxes paid (real): $176k
  First-5-years market: explains 20% of terminal-wealth variance; failure rate 29.2%
  after a worst-quintile start vs 7.9% overall; 74% of failures began with one
  Terminal wealth (real):
    5th pct           $0
    25th pct       $605k
    median        $1.37M
    75th pct       $2.3M
    95th pct      $4.35M
  Median real growth rate: 1.33%/yr
  Chance of ending below start (real): 38.4%
```

Each run also writes a chart (percentile fan, solvency curve, terminal-wealth
distribution) to `out/forecast_<N>y.png` unless you pass `--no-charts`.

**Reading the output.** *Success rate* is the fraction of simulated paths that
never depleted. *Terminal wealth* percentiles are in today's dollars (real);
`$0` at the 5th percentile means at least 5% of paths ran out. The
*first-5-years* line quantifies **sequence-of-returns risk**: bad markets
early in retirement matter far more than bad markets late. Taxes default to
realistic federal brackets plus 5% state tax on a fully taxable account — pass
`--tax-deferred` if the money is in an IRA/401k (this alone often adds several
points of success rate).

## Recipes

**Still saving, not withdrawing** — accumulate $2,000/month (today's dollars,
grown with inflation) for 25 years:

```bash
poorcast run --allocation us_equities=80,us_bonds_10yr=20 \
    --initial 100000 --contribute 2000 --horizons 25
```

**Withdraw fixed dollars instead of a percentage** — $40,000/yr,
inflation-adjusted:

```bash
poorcast run --allocation us_equities=60,us_bonds_10yr=40 \
    --withdraw 40000 --horizons 30
```

**Tighten the belt in bad markets** — same 4% target, but scale withdrawals
down (never below 75% of target) when the real balance is under water. Costs
little in the median case, sharply cuts the risk of ruin:

```bash
poorcast run --allocation us_equities=60,us_bonds_10yr=40 \
    --withdraw 4% --flex --horizons 30
```

**Spend a percent of the current balance** — spending floats with the market
and can shrink, but the portfolio never fully depletes:

```bash
poorcast run --allocation us_equities=70,us_bonds_10yr=30 \
    --withdraw 3.5% --withdraw-strategy percent-of-balance --horizons 30
```

**Put a floor under essential spending with a TIPS ladder** — buy a ladder at
t=0 paying $30k/yr real for the whole horizon; its cost comes off the starting
balance and its income off the withdrawal. If the residual portfolio depletes,
income falls to the ladder floor instead of $0:

```bash
poorcast run --allocation us_equities=70,us_bonds_10yr=30 \
    --initial 2000000 --withdraw 80000 --tips-ladder 30000 --horizons 30
```

Bracket `--tips-ladder-yield` with 0–2 (%) to see regime sensitivity, or use
`--tips-ladder-curve` to price it off today's actual TIPS yield curve (FRED).
By default the ladder is taxable (coupons + inflation accrual taxed as
ordinary income); `--tips-ladder-deferred` holds it in an IRA.

**"What if I had retired in 1968?"** — instead of resampling, run every actual
historical start month as one path:

```bash
poorcast run --allocation us_equities=60,us_bonds_10yr=40 \
    --withdraw 4% --mode historical --horizons 30
```

**Find the best allocation for your situation** — search a structured grid of
mixes for the one that maximizes success probability under all your other
settings, then run it (takes a few minutes per horizon):

```bash
poorcast run --optimize --initial 1500000 --withdraw 60000 --horizons 30
```

**Glidepath** — drift linearly from a starting mix to an ending mix, e.g. a
rising-equity "bond tent" over the first 10 years:

```bash
poorcast run --allocation us_equities=40,us_bonds_10yr=60 \
    --glide-to us_equities=70,us_bonds_10yr=30 --glide-years 10 \
    --withdraw 4% --horizons 30
```

**Assume stocks won't keep getting more expensive** — historically, P/E
multiple expansion contributed ~0.5%/yr to US equity returns. Rerun assuming
that tailwind is gone (or negative, for multiple compression):

```bash
poorcast run --allocation us_equities=100 --withdraw 4% \
    --multiple-expansion 0 --horizons 30
```

Other useful knobs: `--filing married`, `--state-tax 0`, `--cost-basis 0.5`
(embedded gains), `--rebalance 12` (annual), `--start 1926-01` (sample deeper
history, US-only assets), `--sims 50000`, `--seed 1` (reproducible),
`--nominal` (report nominal dollars). `poorcast run --help` lists everything.

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
inflation-adjusted withdrawals and real-dollar reporting. All returns are
monthly, USD, total return. Default sampling starts in 1960 (when everything
overlaps); the sampler automatically narrows the window to the months where
every asset in *your* allocation has data.

**Custom data:** drop `data/custom/<asset>.csv` (`month,return` rows like
`1970-01,0.023`) and rerun `poorcast fetch` — it overrides the built-in series
wherever it has values.

### How pre-1986 international is built

MSCI EAFE begins Dec 1969 and MSCI does not allow free download of pre-1997
history, so the 1960–1985 segment is reconstructed from primary national data
(see `poorcast/reconstruct.py`): monthly local-currency price indices for
Japan, UK, Germany, France, Switzerland, Netherlands, Italy and Australia
(OECD MEI via FRED; month-end Nikkei closes for Japan), dividend accrual and
annual local total-return anchors from the Jordà-Schularick-Taylor
macrohistory database, USD conversion via official Bretton Woods parities
(with the documented devaluation steps) before 1971 and FRED monthly rates
after, GDP-weighted. Calendar years 1970–1985 are additionally anchored to
published MSCI EAFE annual USD total returns, so from 1970 on only the
*within-year monthly shape* is reconstructed; annual totals are observed data.

Run `poorcast validate-intl` for the out-of-sample check against observed
1986–1995 series (monthly correlation 0.81–0.88, annualized gap +1.5 to
+2.9%/yr — the reconstruction runs slightly hot; the smoothing comes from OECD
indices being monthly averages). Known caveats: GDP weights are not market-cap
weights, and the eight countries are most but not all of EAFE. If you have
licensed EAFE data, use the custom-data override above.

## How the simulation works

- **Sampling** — circular block bootstrap over the historical months (default
  24-month blocks, `--block`). The same sampled months are used for every
  asset and for inflation, preserving cross-asset correlation and serial
  structure (momentum, multi-year regimes). `--block 1` gives classic
  independent-month resampling. `--mode historical` runs every actual start
  month instead.
- **Monthly steps** — withdrawal/contribution at the start of each month
  (pro-rata across holdings), then market returns, then rebalancing back to
  target weights every `--rebalance` months (default quarterly).
- **Withdrawals spend income first** — each month's dividends and interest
  fund the withdrawal as cash (no sale, no realized gain); only the shortfall
  sells holdings. Unspent income is reinvested with its basis stepped up.
- **Real dollars** — results are reported in today's dollars by default
  (`--nominal` switches that off), using each path's own sampled inflation.

## Taxes

By default the portfolio is **fully taxable under 2026 US federal brackets**
(Rev. Proc. 2025-32), settled annually: interest (bond coupons, T-bills)
through the ordinary brackets; qualified dividends and net realized long-term
gains through the 0/15/20% brackets stacked on top; standard deduction; and
3.8% NIIT. `--filing married` switches from the single-filer default. Bracket
edges and the deduction are indexed to each path's simulated inflation (as the
law indexes them), while the NIIT threshold stays nominal (statutory).

Realized gains come from true average-cost basis tracking — rebalancing
trades, withdrawal liquidations, and tax payments themselves all realize
gains, netted with loss carryforward. `--cost-basis 0.6` starts with embedded
gains (basis = 60% of value). `--state-tax 5` (the default) adds a flat state
income tax on dividends and realized gains — Treasury interest is
constitutionally state-exempt, and `muni_bonds` income is exempt at both
levels (own-state assumption); muni *capital gains* are taxed normally.

`--tax-deferred` turns taxes off entirely (IRA/401k). `--tax-rate 15
--tax-ordinary 24` overrides the brackets with flat rates settled quarterly.

Income components are observed data: Shiller monthly dividend yields for US
equities (also the small-cap proxy), JST dividend/price for international,
GS10 coupon accrual for bonds. Not modeled: non-portfolio income, short-term
gain rates, the $3,000 loss offset against ordinary income, step-up at death;
terminal wealth is pre-liquidation.

## Commands

| command | what it does |
|---|---|
| `poorcast fetch` | download/refresh historical data from the publishers |
| `poorcast assets` | list asset classes and their data coverage |
| `poorcast run` | run a simulation (see `run --help` for all flags) |
| `poorcast decompose` | split historical US equity returns into dividends, inflation, real EPS growth (margin vs underlying), and P/E multiple expansion |
| `poorcast validate-intl` | out-of-sample check of the pre-1986 international reconstruction |

## Python API

The CLI is a thin wrapper over `poorcast.simulate.simulate(panel, SimConfig)`.
Some features are only reachable from Python — notably
`SimConfig.allocation_rule`, a hook for state-dependent allocation strategies
called at each rebalance. `poorcast/strategies.py` ships several: a funded-ratio
(LDI-style) rule, Bernstein's "stop playing when you've won" ratchet, and
market-signal rules (trend, volatility — use those only with
`mode="historical"`, since resampling destroys the serial structure they rely
on).

```python
from poorcast import data
from poorcast.simulate import SimConfig, Withdrawal, simulate
from poorcast.strategies import ASSETS, funded_ratio_rule

panel = data.load_panel()
cfg = SimConfig(
    allocation={a: [0.5, 0.2, 0.25, 0.05][i] for i, a in enumerate(ASSETS)},
    allocation_rule=funded_ratio_rule(),
    withdrawal=Withdrawal("fixed_real", rate=0.04),
    years=30,
    seed=42,
)
result = simulate(panel, cfg)
print(f"success: {result.success_rate:.1%}")
```

## Caveats

This is a research toy, not financial advice. Bootstrapping assumes the future
is drawn from the same distribution as 1960–present; it ignores fund fees and
mean reversion beyond the block length. The bond and muni series are derived
from constant-maturity yields via a standard pricing approximation. Success
rates above ~95% are not distinguishable from each other given history this
short.

## Development

```bash
.venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest tests/
```
