# poorcast

Monte Carlo retirement and portfolio forecasting driven by **actual market
history**, not normal-distribution assumptions. Give it an asset mix and
(optionally) a withdrawal strategy; it resamples real historical monthly
returns — jointly across assets and inflation, so correlations and regimes
like the 1970s survive — and simulates thousands of possible futures, with
realistic rebalancing and US taxes.

Use it to answer questions like:

- *Can I retire on this portfolio with a 4% withdrawal rate?*
- *How much do things change when Social Security starts at 67?*
- *How much does cutting spending in down markets improve my odds?*
- *What if I buy a TIPS ladder to cover my floor expenses?*
- *Is my 401k enough, given that withdrawals get taxed and RMDs kick in?*
- *Will this 529 cover four years of tuition starting at 18?*
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
  Success rate (never depleted): 91.7%
  Failed paths (8.3%): earliest failure year 13, median year 25, 90% fail after year 19
  Median total income withdrawn (real): $1.2M
  Median total taxes paid (real): $169k
  First-5-years market: explains 22% of terminal-wealth variance among surviving
  paths; failure rate 29.6% after a worst-quintile start vs 8.3% overall; 71%
  of failures began with one
  Terminal wealth (real):
    5th pct           $0
    25th pct       $565k
    median         $1.3M
    75th pct      $2.25M
    95th pct      $4.33M
  Median real growth rate: 0.89%/yr
  Chance of ending below start (real): 39.8%
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

**Retiring before Social Security starts** — retire at 62 on the portfolio
alone, then $30k/yr of (after-tax) Social Security from 67 offsets the
withdrawals. This routinely moves success rates more than any allocation
change:

```bash
poorcast run --allocation us_equities=60,us_bonds_10yr=35,cash=5 \
    --withdraw 4% --age 62 --income 30000@67 --horizons 30
```

`--income` is treated as after-tax (a fair approximation for Social Security
at modest incomes); use `--pension` for streams taxed as ordinary income. Both
repeat: `--income 30000@67 --pension 12000@65`. Any income beyond that month's
spending is invested.

**The money is in a 401k/IRA** — traditional accounts tax nothing inside, but
every withdrawal is ordinary income, and required minimum distributions start
at 73 (both need `--age`):

```bash
poorcast run --allocation us_equities=60,us_bonds_10yr=40 \
    --withdraw 50000 --age 65 --account traditional --horizons 30
```

`--account roth` is tax-free (the old `--tax-deferred` flag now means this).

**Spending that changes with age** — the "retirement smile": spend more in the
go-go years, less later, plus a new roof at 70:

```bash
poorcast run --allocation us_equities=60,us_bonds_10yr=40 --age 65 \
    --withdraw "90000:65-75,70000:75+" --expense 50000@70 --horizons 30
```

Schedule segments are `AMOUNT:FROM-TO` or `AMOUNT:FROM+`; the numbers are ages
when `--age` is given, otherwise **years from the start of the simulation**
(the same rule applies to every `@N` anchor and account schedule). Amounts may
be percents of the initial balance. `--expense` adds one-time real outlays. For
a smooth glide instead of steps, `--spend-decline 1` shrinks real spending 1%
a year, compounding (Blanchett's measured "retirement smile" downslope is
roughly that); `--spend-decline 1@75` starts the decline at 75. It composes
with schedules and `--flex`.

**College savings (529)** — $20k saved for a 3-year-old, $500/month
contributions, four years of $40k tuition from 18; qualified 529 withdrawals
are tax-free:

```bash
poorcast run --allocation us_equities=70,us_bonds_10yr=30 --account 529 \
    --initial 20000 --age 3 --contribute 500 --withdraw "40000:18-22" --horizons 19
```

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

**Or hold the ladder as an allocation**: the reserved asset name
`tips_ladder` in any allocation buys rungs with that share of the balance at
t=0 — a purchase-time cost share, not a maintained weight (rungs amortize
and never rebalance; the other assets renormalize around them). Rung income
offsets withdrawals; remaining principal is carried in reported balances at
par but can't be drawn early. In a household this makes **ladder asset
location** first-class:

```toml
[[account]]
type = "roth"
balance = 500_000
allocation = { tips_ladder = 100 }   # rung payouts tax-free

[tips_ladder]
curve = true      # or yield = 2.0; years = N (default: the horizon)
```

Taxation follows the holding account: taxable = phantom income; traditional
= payouts are RMD-countable ordinary-income distributions (RMDs are owed on
the rungs' value too); Roth = free. Not combinable with `--tips-ladder`,
glidepaths, or 529 accounts.

Rungs beyond the curve's 30-year point cannot be bought today; **the default
assumes they are bridged** — extra 30-year TIPS held (duration-scaled) and
rolled into the long rungs as those maturities are auctioned — which locks
approximately today's forward real rates. `--tips-ladder-tail PCT` (or
`tail_yield` in `[tips_ladder]`) prices the tail at an assumed future roll
yield instead, for the unbridged case (the 2010–26 DFII30 median ≈ 1.0 is
the conservative bracket).

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

**Valuation scenarios**: `--pe-path "30@0,20@10,30@40"` (or a `pe_path`
table in config) drives the multiple-expansion component of US equity
returns along an assumed P/E path — piecewise-linear in log-P/E, net of the
historical rate. Adding `--pe-conditioned` goes further than a mean shift:
bootstrap blocks are drawn from historical months whose Shiller P/E
resembles the assumed level at that point of the path (Gaussian kernel,
`--pe-bandwidth`, default 0.15 log units), and equity returns are
re-centered so the path's multiple change replaces the sampled regimes'
own. Conditioned runs inherit the *dynamics* (volatility, correlations,
inflation regime) of comparable valuation eras — which also means they
inherit those eras' company: since 1960, high-CAPE months are almost
entirely 1995–2021, so a conditioned high-valuation scenario largely
excludes 1970s-style stagflation. That is history's honest answer, not a
neutral one. `--adjust asset=-1.1,...` (or `[adjustments]`) shifts any
asset's mean, e.g. anchoring bond returns at today's yields.

Other useful knobs: `--fees 1` (annual management/expense drag in %; historical
returns are index returns, so the default is free investing — a 1% advisor fee
visibly moves 30-year success rates), `--filing married`, `--state-tax 0`,
`--cost-basis 0.5` (embedded gains), `--rebalance 12` (annual),
`--start 1926-01` (sample deeper history, US-only assets), `--sims 50000`,
`--seed 1` (reproducible), `--nominal` (report nominal dollars). Most age-based flags combine freely —
Social Security plus a spending schedule plus a traditional IRA is a normal
run. `poorcast run --help` lists everything.

## Config files

A real plan doesn't fit comfortably on a command line. `poorcast run --config
plan.toml` reads everything from a TOML file whose keys mirror the flags:

```toml
# Our retirement plan, revisited 2026-09
age = 62
initial = 1_500_000
horizons = [25, 30, 35]
fees = 0.08                  # blended expense ratio, %/yr

[allocation]                 # percents, must sum to 100
us_equities = 55
intl_equities = 15
us_bonds_10yr = 25
cash = 5

[withdrawal]
amount = "4%"                # "4%" or 60000; or use `schedule` instead:
# schedule = [
#   { amount = 90_000, from = 65, to = 75 },   # go-go years
#   { amount = 70_000, from = 75 },            # open-ended
# ]
flex = 75                    # optional belt-tightening floor (%)
decline = { rate = 1, from = 75 }   # or `decline = 1`: real spending -1%/yr

[[income]]                   # after-tax streams (Social Security); repeatable
annual = 30_000
at = 67

[[pension]]                  # taxed as ordinary income
annual = 12_000
at = 65

[[expense]]                  # one-time outlays, today's dollars
amount = 50_000
at = 70                      # the roof

[taxes]
account = "taxable"          # taxable | traditional | roth | 529
filing = "married"
state = 5
cost_basis = 0.6

[simulation]                 # all optional
sims = 10_000
seed = 42

[output]
charts = true
```

Explicit command-line flags override the file, so what-ifs don't require
editing the plan: `poorcast run --config plan.toml --withdraw 3.5%`. The
repeatable flags (`--income`, `--pension`, `--expense`) *add* to the file's
streams rather than replacing them. A `[tips_ladder]` section (`annual`,
`yield`, `curve`, `deferred`) and a `[glide]` section (`to`, `years`) cover
the remaining features; unknown keys are hard errors with a did-you-mean
hint, so a typo can't silently skew a forecast.

## Multi-account households

Real households hold several accounts with different tax treatments. Repeated
`[[account]]` sections (config-file and Python API only) simulate them
*jointly* — same market paths, one household spending policy:

```toml
[[account]]
type = "taxable"
balance = 1_500_000
cost_basis = 0.6
allocation = { us_equities = 60, muni_bonds = 35, cash = 5 }

[[account]]
type = "traditional"        # the 401k/IRA
balance = 400_000
allocation = { us_equities = 60, us_bonds_10yr = 35, cash = 5 }

[[account]]
type = "roth"
balance = 250_000
allocation = { us_equities = 60, us_bonds_10yr = 35, cash = 5 }
```

How it behaves:

- **Withdrawals waterfall** through the accounts — taxable first, then
  traditional, then Roth/529 by default (`withdraw_order = ["roth",
  "taxable", ...]` overrides). Percent withdrawals are percents of the
  *combined* starting balance; contributions and surplus income land in the
  taxable account.
- **Taxes are settled jointly**, as on a real return: taxable-account
  interest, dividends, and realized gains stack with traditional-account
  distributions through one set of brackets and one standard deduction, and
  the bill is paid from the taxable account.
- **RMD dollars actually move**: a required distribution beyond spending is
  sold from the IRA, taxed, and reinvested in the taxable account (basis =
  value) — replacing the single-account mode's deemed-distribution
  approximation.
- **Asset location works**: each account holds its own allocation (munis in
  taxable, Treasuries in tax-advantaged, as above), rebalanced independently.
- **Early withdrawals are penalized**: draws before age 59½ from a
  traditional account, or from a Roth beyond its contribution basis (the
  account's `cost_basis` × balance), incur the 10% penalty, settled annually
  (on by default when `--age` is given; `--no-early-penalty` or
  `early_penalty = false` under `[taxes]` disables it). Ordinary tax on
  early Roth earnings is not modeled — only the penalty.
- **Accounts can carry their own draw schedule** — `schedule = [{amount =
  111_111, from = 59, to = 68}]` on an `[[account]]` drains that account on
  its own timetable (ages, anchored by `age`), on top of the household
  withdrawal policy; shortfalls fall back to the waterfall. On a 529 these
  scheduled draws are **qualified** (tuition): tax- and penalty-free — so
  college and retirement can share one simulation.
- **Unscheduled 529 draws are non-qualified** — the waterfall reaching a 529
  means it's funding living expenses — so each such distribution splits
  pro-rata between contributions (`cost_basis`) and earnings at the
  draw-date ratio, and the earnings are taxed as ordinary income plus the
  10% penalty, at any age. Not modeled: scholarship/death/disability
  exceptions, state deduction recapture, the Roth-rollover escape hatch.
- The report adds median terminal wealth per account; success still means
  the *household* never depleted (every account empty).

At most one taxable and one traditional account. Glidepaths, allocation
rules, and `--tips-ladder` remain single-account features for now.

**Household optimization**: with `[[account]]` sections, `--optimize`
searches a declared space of (household equity share × total ladder
dollars), keeping the account structure fixed — each account's liquid
sleeve is rescaled preserving its own asset proportions, ladder dollars
fill traditional accounts first then taxable, and 529s are left alone:

```toml
[optimize]
equity = [40, 80, 10]                  # min, max, step (%)
ladder = [0, 4_000_000, 1_000_000]     # min, max, step ($)
```

Candidates screen under common random numbers, leaders refine across
several seeds, and the report shows the whole frontier (success ± sd,
5th-percentile/median terminal, income floor) — ladder size is a
risk-preference dial, so the tradeoff is the answer. Composes with
scenario assumptions: `poorcast run --config plan.toml --optimize
--pe-path "30@0,20@10,30@40"` optimizes under a valuation cycle.

## Asset classes

| name | coverage | source |
|---|---|---|
| `us_equities` | 1926+ | CRSP value-weighted total market (Ken French library) |
| `us_small_cap` | 1926+ | bottom 30% by market cap, value-weighted (Ken French) |
| `intl_equities` | 1960+ | reconstructed 8-country composite (1960–85), AQR Global ex USA (1986–90), Ken French Developed ex US (1990+) |
| `us_bonds_10yr` | 1953+ | 10-yr Treasury total return derived from FRED GS10 yields |
| `muni_bonds` | 1953+ | Bond Buyer GO-20 yields priced at their 20y maturity through 2007, observed MUB ETF total returns after |
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

**Account types** (`--account`): `taxable` (default) applies all of the
above. `traditional` (IRA/401k) taxes nothing inside the account; instead
every distribution — spending withdrawals, and the tax payments themselves —
is taxed as ordinary income through the same brackets (with the standard
deduction), and RMDs are enforced from age 73 per the IRS Uniform Lifetime
Table: required dollars beyond spending are deemed distributed and taxed,
while staying invested (approximating reinvestment in a taxable account whose
own future tax drag is ignored — a mild flattery late in life). `roth` and
`529` (qualified use) are tax-free. `--pension` income is taxed as ordinary
income in whichever regime is active and raises MAGI for NIIT purposes.

`--tax-rate 15 --tax-ordinary 24` overrides the brackets with flat rates
settled quarterly (for traditional accounts, the ordinary rate applies to
distributions annually).

Income components are observed data: Shiller monthly dividend yields for US
equities (also the small-cap proxy), JST dividend/price for international,
GS10 coupon accrual for bonds. Not modeled: Social Security benefit taxation
(`--income` is treated as fully after-tax), short-term gain rates, the $3,000
loss offset against ordinary income, step-up at death; terminal wealth is
pre-liquidation.

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
Everything the flags expose is a `SimConfig`/`Withdrawal` field (`accounts`
as `Account` tuples, `age`, `income` as `IncomeStream` tuples, `expenses`,
`fee_annual`, `Withdrawal.schedule`/`decline`/`flex_floor`, ...), and the API
supports
things the CLI can't — e.g. `SimResult.months` gives each path's sampled
historical months, so a second account can be grown on the *same* market
paths for correlated multi-account estimates. Some features are only
reachable from Python — notably `SimConfig.allocation_rule`, a hook for
state-dependent allocation strategies called at each rebalance. `poorcast/strategies.py` ships several: a funded-ratio
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
is drawn from the same distribution as 1960–present; it ignores mean reversion
beyond the block length, and fees default to zero (set `--fees` to yours).
Yield-derived bond returns use exact repricing of an aged annual-coupon par
bond. The muni series is self-consistent (GO-20 yields priced at their
20-year maturity) but changes duration at the 2007 splice onto MUB (~6y) —
pre-2007 munis are correspondingly more volatile than the fund era. The bond and muni series are derived
from constant-maturity yields via a standard pricing approximation. Success
rates above ~95% are not distinguishable from each other given history this
short.

## Development

```bash
.venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest tests/
```
