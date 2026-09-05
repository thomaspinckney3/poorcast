"""poorcast command-line interface.

Examples:
  poorcast fetch
  poorcast assets
  poorcast run --allocation us_equities=60,intl_equities=20,us_bonds_10yr=15,cash=5 \\
               --initial 1000000 --withdraw 3% --horizons 20,30,40
  poorcast run --allocation us_equities=100 --contribute 2000 --horizons 25 --no-charts
  poorcast run --allocation us_equities=60,us_bonds_10yr=40 --withdraw 4% --mode historical
  poorcast run --config plan.toml
  poorcast run --config plan.toml --withdraw 3.5%   # what-if: flags beat the file
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import data as data_mod
from .simulate import Account, IncomeStream, SimConfig, Withdrawal, simulate


def parse_allocation(text: str) -> dict[str, float]:
    alloc = {}
    for part in text.split(","):
        if "=" not in part:
            raise argparse.ArgumentTypeError(
                f"bad allocation entry {part!r}; expected asset=percent"
            )
        k, v = part.split("=", 1)
        alloc[k.strip()] = float(v) / 100.0
    total = sum(alloc.values())
    if abs(total - 1.0) > 1e-6:
        raise argparse.ArgumentTypeError(
            f"allocation percentages sum to {total * 100:g}, expected 100"
        )
    return alloc


def parse_withdrawal(text: str | None, strategy: str) -> Withdrawal:
    if text is None:
        return Withdrawal("none")
    text = text.strip()
    if text.endswith("%"):
        rate = float(text[:-1]) / 100.0
        kind = "percent_of_balance" if strategy == "percent-of-balance" else "fixed_real"
        return Withdrawal(kind, rate=rate)
    if strategy == "percent-of-balance":
        raise argparse.ArgumentTypeError(
            "percent-of-balance strategy needs a percentage, e.g. --withdraw 3%"
        )
    return Withdrawal("fixed_real", amount=float(text.replace("_", "").replace(",", "")))


def _dollars(text: str) -> float:
    return float(text.replace("_", "").replace(",", ""))


def parse_at_age(text: str) -> tuple[float, int | None]:
    """'30000@67' -> (30000.0, 67); '30000' -> (30000.0, None)."""
    if "@" in text:
        amt, at = text.split("@", 1)
        return _dollars(amt), int(at)
    return _dollars(text), None


def parse_schedule(text: str, age: int, initial: float) -> tuple[tuple[int, float], ...]:
    """Age-varying spending: '80000:65-75,60000:75+' -> ((start_month, annual), ...).

    Each segment is AMOUNT:FROM-TO or AMOUNT:FROM+ (FROM inclusive, TO
    exclusive). The numbers are ages anchored by `age`; pass age=0 to treat
    them as years from the start of the simulation. AMOUNT may be a percent
    of the initial balance ('4%'). Segments may not overlap; uncovered
    stretches spend nothing.
    """
    segs = []
    for part in text.split(","):
        if ":" not in part:
            raise ValueError(
                f"bad schedule entry {part!r}; expected AMOUNT:FROM-TO or AMOUNT:FROM+"
            )
        amt_s, rng = part.rsplit(":", 1)
        amt = (
            float(amt_s[:-1]) / 100.0 * initial
            if amt_s.endswith("%")
            else _dollars(amt_s)
        )
        if rng.endswith("+"):
            a, b = int(rng[:-1]), None
        elif "-" in rng:
            a_s, b_s = rng.split("-", 1)
            a, b = int(a_s), int(b_s)
            if b <= a:
                raise ValueError(f"empty age range {rng!r}")
        else:
            raise ValueError(f"bad age range {rng!r}; expected FROM-TO or FROM+")
        segs.append(((a - age) * 12, None if b is None else (b - age) * 12, amt))
    segs.sort(key=lambda s: s[0])
    sched: list[tuple[int, float]] = []
    prev_end: int | None = None
    for i, (start_m, end_m, amt) in enumerate(segs):
        if i > 0 and (prev_end is None or start_m < prev_end):
            raise ValueError("overlapping withdrawal schedule segments")
        if end_m is not None and end_m <= 0:
            prev_end = end_m
            continue  # segment lies entirely before the starting age
        start = max(start_m, 0)
        if sched and prev_end is not None and start > max(prev_end, 0):
            sched.append((max(prev_end, 0), 0.0))  # gap: no spending
        sched.append((start, amt))
        prev_end = end_m
    if not sched:
        raise ValueError("withdrawal schedule lies entirely before the starting age")
    if prev_end is not None and max(prev_end, 0) > sched[-1][0]:
        sched.append((max(prev_end, 0), 0.0))  # bounded last segment: stop after
    return tuple(sched)


def parse_pe_path(text: str) -> list[tuple[float, float]]:
    """'30@0,20@10,30@40' -> [(year, pe), ...] sorted, validated."""
    points = []
    for part in text.split(","):
        pe_s, yr_s = part.split("@", 1)
        pe, yr = float(pe_s), float(yr_s)
        if pe <= 0 or yr < 0:
            raise ValueError(f"bad P/E path point {part!r}")
        points.append((yr, pe))
    points.sort()
    if len(points) < 2:
        raise ValueError("a P/E path needs at least two PE@YEAR points")
    months = [int(round(y * 12)) for y, _ in points]
    if any(b <= a for a, b in zip(months, months[1:])):
        raise ValueError("P/E path points must be at least one month apart")
    return points


def pe_path_rates(points: list[tuple[float, float]], n_months: int):
    """Per-month annual multiple-expansion rates: piecewise-linear in log-P/E
    between points, flat (0) before the first and after the last."""
    import numpy as np

    rates = np.zeros(n_months)
    for (y0, p0), (y1, p1) in zip(points, points[1:]):
        m0, m1 = int(round(y0 * 12)), int(round(y1 * 12))
        rates[m0:min(m1, n_months)] = np.log(p1 / p0) / ((m1 - m0) / 12.0)
    return rates


def build_parser(run_defaults: dict | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="poorcast",
        description="Monte Carlo portfolio forecasting driven by actual market history",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples:")[1] if __doc__ else None,
    )
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("fetch", help="download/refresh historical data from source")
    sub.add_parser("assets", help="list asset classes and their data coverage")
    sub.add_parser(
        "decompose",
        help="decompose historical US equity returns: dividends, inflation, real "
        "EPS growth (margin vs underlying), P/E multiple expansion",
    )
    sub.add_parser(
        "validate-intl",
        help="out-of-sample check of the pre-1986 international reconstruction "
        "against observed 1986-1995 data",
    )

    lad = sub.add_parser(
        "ladder",
        help="generate a TIPS ladder buy list (rung face values by maturity)",
    )
    lad.add_argument("--config", metavar="FILE",
                     help="emit the ladder(s) implied by a plan's tips_ladder "
                     "allocations")
    lad.add_argument("--annual", type=float, help="target real income per year")
    lad.add_argument("--cost", type=float, help="total dollars to invest "
                     "(solves the annual income instead)")
    lad.add_argument("--years", type=int, default=30, help="ladder length (default 30)")
    lad.add_argument("--yield", dest="lyield", type=float, default=2.0,
                     metavar="PCT", help="flat real yield %% (default 2.0)")
    lad.add_argument("--curve", action="store_true",
                     help="price off today's FRED TIPS real-yield curve")
    lad.add_argument("--tail", type=float, default=None, metavar="PCT",
                     help="real yield %% for rungs beyond 30y (default: "
                     "bridge-locked at the 30y point)")
    lad.add_argument("--taxable", action="store_true",
                     help="note the ladder as taxable (phantom income)")
    lad.add_argument("--cusips", action="store_true",
                     help="look up outstanding TIPS CUSIPs from TreasuryDirect "
                     "for each rung's maturity (for secondary-market buying); "
                     "flags years with no maturing TIPS")
    lad.add_argument("--price", action="store_true",
                     help="with --cusips, add a MODELED clean price per $100 "
                     "(from each bond's coupon and today's real curve) and "
                     "estimated outlay — an estimate, not a live quote")

    r = sub.add_parser("run", help="run a simulation")
    r.add_argument(
        "--config",
        metavar="FILE",
        default=None,
        help="TOML file of run settings (keys mirror these flags; see the "
        "README). Explicit command-line flags override file values",
    )
    r.add_argument(
        "--allocation",
        type=parse_allocation,
        default=None,
        help="e.g. us_equities=60,intl_equities=20,us_bonds_10yr=15,cash=5 (percents, sum 100)",
    )
    r.add_argument(
        "--optimize",
        action="store_true",
        help="instead of --allocation, search the allocation grid for the mix that "
        "maximizes success probability under all the other settings (withdrawal, "
        "taxes, horizon...), then run it. Takes a few minutes per horizon",
    )
    r.add_argument(
        "--optimize-tolerance",
        type=float,
        default=0.0,
        metavar="PTS",
        help="household --optimize: treat candidates within PTS points of the "
        "best success rate as tied and pick the highest median real estate "
        "among them (default 0 = strict success ranking). The tolerance is "
        "the household's risk preference: with one shared history, success "
        "differences of a point or two may not be real",
    )
    r.add_argument(
        "--optimize-anchor",
        choices=["base", "stress"],
        default="base",
        help="which world the tolerance band is measured in: the run's own "
        "assumptions (base) or the --optimize-stress scenario (stress). Estate "
        "is always judged in the base world",
    )
    r.add_argument(
        "--optimize-stress",
        default=None,
        metavar="PE@YEAR,...",
        help="a second P/E path (same syntax as --pe-path, conditioned when "
        "--pe-conditioned is set) under which every household candidate is "
        "also scored, reported alongside the base results and available as "
        "the tolerance anchor",
    )
    r.add_argument("--initial", type=float, default=1_000_000, help="starting balance (default 1,000,000)")
    r.add_argument(
        "--glide-to",
        type=parse_allocation,
        default=None,
        help="glidepath: drift linearly from --allocation to this mix (same assets), "
        "e.g. a rising-equity 'bond tent'",
    )
    r.add_argument(
        "--glide-equity",
        type=float,
        default=None,
        metavar="PCT",
        help="glidepath by equity share: drift the equity/defensive split "
        "linearly to PCT%% equities over --glide-years, keeping each "
        "bucket's own asset proportions. Works for households too: every "
        "account with a liquid sleeve glides (529s and accounts with their "
        "own glide_to are left alone)",
    )
    r.add_argument(
        "--glide-years",
        type=int,
        default=None,
        help="years over which the glide completes (default: the whole horizon)",
    )
    r.add_argument(
        "--withdraw",
        default=None,
        help="annual withdrawal: '3%%' (percent rule), '40000' (dollars/yr, "
        "inflation-adjusted), or a varying schedule like "
        "'80000:65-75,60000:75+' - the numbers are ages with --age, else "
        "years from the start (amounts may also be percents of initial)",
    )
    r.add_argument(
        "--age",
        type=int,
        default=None,
        help="age at the start of the simulation; enables age-based features: "
        "@AGE in --income/--pension/--expense, withdrawal schedules, and RMDs "
        "for traditional accounts",
    )
    r.add_argument(
        "--account",
        choices=["taxable", "traditional", "roth", "529"],
        default="taxable",
        help="account type (default taxable, with full dividend/gains taxation). "
        "traditional (IRA/401k): nothing taxed inside, every distribution taxed "
        "as ordinary income, RMDs from age 73 (needs --age). roth and 529 "
        "(qualified use): tax-free",
    )
    r.add_argument(
        "--income",
        action="append",
        default=[],
        metavar="ANNUAL[@AGE]",
        help="outside income in today's dollars/yr, treated as after-tax "
        "(approximates Social Security): e.g. --income 30000@67. Offsets "
        "withdrawals; any surplus is invested. Repeatable; @N is an age "
        "with --age, else a year from the start",
    )
    r.add_argument(
        "--pension",
        action="append",
        default=[],
        metavar="ANNUAL[@AGE]",
        help="like --income but taxed as ordinary income (pension, annuity "
        "payout). Repeatable",
    )
    r.add_argument(
        "--expense",
        action="append",
        default=[],
        metavar="AMOUNT@AGE",
        help="one-time expense in today's dollars (a roof, a car, a wedding): "
        "e.g. --expense 50000@70. Repeatable; @N is an age with --age, else "
        "a year from the start",
    )
    r.add_argument(
        "--withdraw-strategy",
        choices=["fixed-real", "percent-of-balance"],
        default="fixed-real",
        help="fixed-real: %% of initial balance, inflation-adjusted (the classic 3%%/4%% rule). "
        "percent-of-balance: %% of current balance, recomputed yearly (never fully depletes)",
    )
    r.add_argument(
        "--gross",
        action="store_true",
        help="treat the withdrawal target as the TOTAL budget, gross of "
        "taxes: taxes paid in one year reduce the next year's spendable "
        "amount (default: the target is pure consumption; taxes are drawn "
        "from the portfolio on top). Fixed-real withdrawals only",
    )
    r.add_argument(
        "--spend-decline",
        default=None,
        metavar="PCT[@AGE]",
        help="real spending declines PCT%%/yr, compounding - the observed "
        "'retirement smile' downslope (Blanchett measured roughly 1): e.g. "
        "--spend-decline 1, or 1@75 to start the decline at 75 (needs --age). "
        "Composes with schedules and --flex; fixed-real withdrawals only",
    )
    r.add_argument(
        "--flex",
        nargs="?",
        const=75.0,
        default=None,
        type=float,
        metavar="FLOOR_PCT",
        help="reduce withdrawals in down markets: scale the monthly withdrawal by "
        "current real balance / initial balance, never below FLOOR_PCT%% of target "
        "(default 75). Only with the fixed-real strategy",
    )
    r.add_argument(
        "--tips-ladder",
        type=float,
        default=None,
        metavar="ANNUAL",
        help="buy a TIPS ladder at t=0 paying this many real dollars/yr for the whole "
        "horizon, held to maturity outside the portfolio. Its cost comes off the "
        "starting balance and its income off the withdrawal. Requires a fixed-real "
        "--withdraw",
    )
    r.add_argument(
        "--tips-ladder-yield",
        type=float,
        default=2.0,
        metavar="PCT",
        help="flat real yield (%%) the ladder is bought at (default 2.0; bracket "
        "with 0-2 to see regime sensitivity)",
    )
    r.add_argument(
        "--tips-ladder-curve",
        action="store_true",
        help="price the ladder off today's TIPS real yield curve (FRED DFII "
        "5/7/10/20/30y, interpolated) instead of a flat --tips-ladder-yield",
    )
    r.add_argument(
        "--tips-ladder-tail",
        type=float,
        default=None,
        metavar="PCT",
        help="real yield (%%) for ladder rungs beyond the curve's 30y point - "
        "such rungs can't be bought today and must be rolled into at future "
        "long real yields. Default: today's 30y yield (bridge-lock "
        "approximation); the 2010-2026 DFII30 median ~1.0 is the "
        "conservative bracket",
    )
    r.add_argument(
        "--tips-ladder-deferred",
        action="store_true",
        help="hold the ladder in a tax-deferred account. Default: taxable — "
        "coupons and the annual inflation accrual (phantom income) are taxed "
        "as federal ordinary income, paid from the portfolio",
    )
    r.add_argument(
        "--contribute",
        type=float,
        default=0.0,
        help="monthly contribution in today's dollars (grows with inflation)",
    )
    r.add_argument(
        "--filing",
        choices=["single", "married"],
        default="single",
        help="filing status for the default federal-bracket taxation (default single)",
    )
    r.add_argument(
        "--state-tax",
        type=float,
        default=5.0,
        metavar="PCT",
        help="flat state income tax (%%) on dividends and realized gains "
        "(default 5; Treasury interest is state-exempt, muni income exempt "
        "from both levels). 0 to disable",
    )
    r.add_argument(
        "--no-early-penalty",
        action="store_true",
        help="skip the 10%% early-withdrawal penalty modeled on pre-59.5 "
        "traditional draws and roth earnings draws (on by default when "
        "--age is given)",
    )
    r.add_argument(
        "--tax-deferred",
        action="store_true",
        help="alias for --account roth: no taxes modeled (use --account "
        "traditional for an IRA/401k whose withdrawals are taxed)",
    )
    r.add_argument(
        "--tax-rate",
        type=float,
        default=0.0,
        metavar="PCT",
        help="override the federal brackets with a flat long-term capital-gains + "
        "qualified-dividend rate (%%), settled quarterly",
    )
    r.add_argument(
        "--tax-ordinary",
        type=float,
        default=None,
        metavar="PCT",
        help="ordinary-income rate (%%) for interest (bond coupons, T-bills). "
        "Default: same as --tax-rate",
    )
    r.add_argument(
        "--cost-basis",
        type=float,
        default=1.0,
        metavar="FRACTION",
        help="starting cost basis as a fraction of the starting balance "
        "(default 1.0 = no embedded gains; retirees often hold appreciated "
        "positions, e.g. 0.5)",
    )
    r.add_argument(
        "--fees",
        type=float,
        default=0.0,
        metavar="PCT",
        help="annual management/expense fee (%%) applied to all holdings as a "
        "return drag - fund expense ratios or an advisor fee, e.g. 0.1 for "
        "10bp index funds, 1 for a typical advisor (default 0)",
    )
    r.add_argument(
        "--horizons",
        default="30",
        help="comma-separated horizons in years (default 30), e.g. 10,20,30",
    )
    r.add_argument("--sims", type=int, default=10_000, help="number of Monte Carlo paths (default 10000)")
    r.add_argument(
        "--mode",
        choices=["bootstrap", "historical"],
        default="bootstrap",
        help="bootstrap: block-resampled history (default). historical: every actual start month",
    )
    r.add_argument(
        "--block",
        type=int,
        default=24,
        help="bootstrap block length in months (default 24; 1 = fully independent months)",
    )
    r.add_argument(
        "--multiple-expansion",
        type=float,
        default=None,
        metavar="PCT_PER_YR",
        help="assumed future P/E multiple-expansion contribution to US equity "
        "returns (%%/yr). Sampled us_equities/us_small_cap returns are shifted "
        "by (assumed - historical); e.g. 0 removes the historical tailwind, "
        "negative models multiple compression. Default: leave history as is",
    )
    r.add_argument(
        "--pe-path",
        default=None,
        metavar="PE@YEAR,...",
        help="assumed P/E valuation path for US equities, e.g. "
        "'30@0,20@10,30@40': piecewise-linear in log-P/E (constant multiple "
        "expansion per segment, relative to the historical rate), held flat "
        "after the last point. Years are from the start of the simulation. "
        "Supersedes any multiple-expansion setting",
    )
    r.add_argument(
        "--pe-conditioned",
        action="store_true",
        help="with --pe-path: instead of a mean shift, draw bootstrap blocks "
        "from historical months whose Shiller P/E resembles the assumed "
        "level at that point of the path (Gaussian kernel in log-P/E), and "
        "re-center US equity returns so the path's multiple expansion "
        "replaces the sampled regimes' own. Blocks then carry the "
        "volatility/correlation/inflation dynamics of comparable eras",
    )
    r.add_argument(
        "--pe-bandwidth",
        type=float,
        default=0.15,
        metavar="LOGW",
        help="kernel width for --pe-conditioned, in log-P/E units "
        "(default 0.15 = roughly +/-15%%)",
    )
    r.add_argument(
        "--adjust",
        default=None,
        metavar="ASSET=PCT,...",
        help="additive annual return adjustment per asset (%%/yr), e.g. "
        "us_bonds_10yr=-1.1,muni_bonds=-1.8 to anchor bond returns at "
        "today's yields instead of the sampled-history average (which "
        "includes the 1982-2020 yield decline). Stacks with "
        "--multiple-expansion; replaces (does not merge with) a config "
        "[adjustments] table",
    )
    r.add_argument(
        "--rebalance",
        type=int,
        default=3,
        metavar="MONTHS",
        help="months between rebalances (default 3 = quarterly; 12 = annual, "
        "a large value = never)",
    )
    r.add_argument("--start", default="1955-01", help="earliest history to sample (default 1955-01, when every asset has data)")
    r.add_argument("--end", default=None, help="latest history to sample (default: all)")
    r.add_argument("--seed", type=int, default=None, help="random seed for reproducibility")
    r.add_argument("--nominal", action="store_true", help="report nominal dollars instead of real")
    r.add_argument("--out", default="out", help="directory for charts (default ./out)")
    r.add_argument("--no-charts", action="store_true", help="skip chart generation")
    if run_defaults:
        r.set_defaults(**run_defaults)
    return p


def _held_assets(cfg: SimConfig, panel) -> list[str]:
    """Panel assets held anywhere in the run: the config-level allocation
    and every account's (households may have no config-level allocation;
    the reserved tips_ladder name is not a panel column)."""
    held: list[str] = []
    for alloc in [cfg.allocation or {}] + [a.allocation or {} for a in cfg.accounts or ()]:
        for k in alloc:
            if k in panel.columns and k not in held:
                held.append(k)
    return held


def _config_path(argv: list[str]) -> str | None:
    for i, tok in enumerate(argv):
        if tok == "--config" and i + 1 < len(argv):
            return argv[i + 1]
        if tok.startswith("--config="):
            return tok.split("=", 1)[1]
    return None



def _run_ladder(args) -> int:
    """poorcast ladder: print the rung-by-rung buy list for a ladder, either
    from explicit --annual/--cost + pricing, or from a plan config's
    tips_ladder allocations."""
    from .ladder import (build_ladder, build_ladder_curve, current_real_curve,
                         format_ladder, format_ladder_gap_adjusted,
                         outstanding_tips)

    curve = current_real_curve() if args.curve else None
    tail = None if args.tail is None else args.tail / 100.0
    tips = None
    if args.cusips:
        try:
            tips = outstanding_tips()
        except Exception as e:
            print(f"error: could not fetch TIPS list from TreasuryDirect: {e}")
            return 2
        import datetime
        print(f"Matching against {len(tips)} outstanding TIPS "
              f"(maturities {tips[0]['maturity'].year}-{tips[-1]['maturity'].year}, "
              f"as of {datetime.date.today().isoformat()}).\n")

    def one(annual=None, cost=None, years=None, taxable=False, label=""):
        import datetime
        yrs = years or args.years
        if curve is not None:
            unit = build_ladder_curve(1.0, yrs, curve, taxable=taxable, tail_yield=tail)
            a = annual if annual is not None else cost / unit.cost
            spec = build_ladder_curve(a, yrs, curve, taxable=taxable, tail_yield=tail)
        else:
            unit = build_ladder(1.0, yrs, args.lyield / 100.0, taxable=taxable)
            a = annual if annual is not None else cost / unit.cost
            spec = build_ladder(a, yrs, args.lyield / 100.0, taxable=taxable)
        if tips is not None:
            pcurve = current_real_curve() if args.price else None
            price_curve = ({m: pcurve[m] for m in pcurve} if pcurve else None)
            print(format_ladder_gap_adjusted(
                spec, tips, datetime.date.today().year, label=label,
                price_curve=price_curve))
        else:
            print(format_ladder(spec, label=label))
        return spec

    if args.config:
        from .config import ConfigError, load_config

        try:
            cfg = load_config(args.config)
        except ConfigError as e:
            print(f"error: {e}")
            return 2
        accounts = cfg.get("accounts")
        if not accounts:
            print("error: --config needs [[account]] sections with a tips_ladder "
                  "allocation (or use --annual/--cost directly)")
            return 2
        # Config pricing overrides the flag defaults.
        if cfg.get("tips_ladder_curve"):
            curve = current_real_curve()
        elif "tips_ladder_yield" in cfg:
            args.lyield = cfg["tips_ladder_yield"]
        if "tips_ladder_tail" in cfg:
            tail = cfg["tips_ladder_tail"] / 100.0
        lyrs = cfg.get("ladder_years") or int(str(cfg.get("horizons", "30")).split(",")[0])
        found = False
        for a in accounts:
            wl = (a.get("allocation") or {}).get("tips_ladder", 0.0)
            if wl > 0:
                found = True
                one(cost=wl * a["balance"], years=lyrs,
                    taxable=(a["kind"] == "taxable"),
                    label=f"{a['kind']} account, ${wl * a['balance']:,.0f}")
                print()
        if not found:
            print("no tips_ladder allocations found in the config")
            return 2
        return 0

    if (args.annual is None) == (args.cost is None):
        print("error: give exactly one of --annual or --cost (or --config)")
        return 2
    one(annual=args.annual, cost=args.cost, taxable=args.taxable)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    # --config values become argparse defaults, so explicit flags override.
    run_defaults = None
    cfg_path = _config_path(argv)
    if cfg_path is not None:
        from .config import ConfigError, load_config

        try:
            run_defaults = load_config(cfg_path)
        except ConfigError as e:
            print(f"error: {e}")
            return 2
    args = build_parser(run_defaults).parse_args(argv)

    if args.command == "fetch":
        panel = data_mod.build_panel(refresh=True)
        print(f"Wrote {data_mod.PANEL_PATH}")
        print(data_mod.coverage(panel).to_string())
        return 0

    if args.command == "decompose":
        from .decompose import equity_return_decomposition, print_decomposition

        print_decomposition(equity_return_decomposition())
        return 0

    if args.command == "validate-intl":
        from .reconstruct import validation_report

        print(
            "Reconstruction extended past 1985 (no EAFE anchor there) vs the two\n"
            "observed series the panel uses from 1986 on:\n"
        )
        for rep in validation_report():
            print(f"  vs {rep['versus']} ({rep['window']}, {rep['months']} months)")
            print(f"    monthly correlation:  {rep['monthly_corr']:.3f}")
            print(
                f"    annualized return:    {rep['annualized_recon']:+.2%} recon "
                f"vs {rep['annualized_actual']:+.2%} observed "
                f"(gap {rep['annualized_gap']:+.2%})"
            )
            print(f"    tracking error:       {rep['tracking_error_ann']:.2%}/yr\n")
        return 0

    if args.command == "ladder":
        return _run_ladder(args)

    panel = data_mod.load_panel()

    if args.command == "assets":
        cov = data_mod.coverage(panel)
        print("Asset classes (monthly total returns, USD):\n")
        for name, desc in data_mod.ASSET_DESCRIPTIONS.items():
            row = cov.loc[name]
            print(f"  {name:<15} {row['first']} .. {row['last']}   {desc}")
        row = cov.loc["inflation"]
        print(f"  {'inflation':<15} {row['first']} .. {row['last']}   US CPI (via FRED), sampled jointly")
        print("\nCustom override: put data/custom/<asset>.csv (month,return) to extend a series.")
        return 0

    # Multi-account household (config-file [[account]] sections only).
    accounts = None
    if getattr(args, "accounts", None):
        specs = []
        for a in args.accounts:
            a = dict(a)
            sched_text = a.pop("schedule", None)
            if sched_text is not None:
                try:
                    a["schedule"] = parse_schedule(sched_text, args.age or 0, a["balance"])
                except ValueError as e:
                    print(f"error: {e}")
                    return 2
            specs.append(Account(**a))
        accounts = tuple(specs)
        if args.glide_to is not None:
            print("error: a household glides per account: put glide_to on each "
                  "[[account]] (with [glide] years), or use --glide-equity / "
                  "[glide] equity for a household-wide equity glide")
            return 2
        if args.tips_ladder is not None:
            print("error: --tips-ladder is a single-account feature; drop the "
                  "[[account]] sections to use it")
            return 2
        if args.optimize and not getattr(args, "optimize_grid", None):
            print("error: household optimization needs an [optimize] section "
                  "declaring the search space, e.g. equity = [40, 80, 10] "
                  "and/or ladder = [0, 4_000_000, 1_000_000]")
            return 2
        if args.allocation is None and any(a.allocation is None for a in accounts):
            print("error: every [[account]] needs an allocation (or give a "
                  "top-level [allocation])")
            return 2
        args.initial = sum(a.balance for a in accounts)
    elif (args.allocation is None) == (not args.optimize):
        print("error: provide exactly one of --allocation or --optimize")
        return 2
    glide_equity = getattr(args, "glide_equity", None)
    any_glide = (
        args.glide_to is not None or glide_equity is not None
        or any(a.allocation_end is not None for a in accounts or ())
    )
    if args.optimize and any_glide:
        print("error: --optimize searches static allocations; drop the glidepath")
        return 2
    if glide_equity is not None:
        from dataclasses import replace

        from .optimize import household_bucket_templates, rescale_equity

        if not 0 <= glide_equity <= 100:
            print("error: --glide-equity is a percent (0-100)")
            return 2
        try:
            if accounts is not None:
                agg_eq, agg_de = household_bucket_templates(accounts, args.allocation)
                glided = []
                for a in accounts:
                    alloc = a.allocation or args.allocation or {}
                    liquid = {k: v for k, v in alloc.items() if k != "tips_ladder"}
                    if a.kind == "529" or a.allocation_end is not None or not liquid:
                        glided.append(a)
                        continue
                    glided.append(replace(a, allocation_end=rescale_equity(
                        alloc, glide_equity / 100.0, agg_eq, agg_de)))
                accounts = tuple(glided)
            else:
                if args.glide_to is not None:
                    print("error: give --glide-to or --glide-equity, not both")
                    return 2
                args.glide_to = rescale_equity(args.allocation, glide_equity / 100.0)
        except ValueError as e:
            print(f"error: {e}")
            return 2

    return_adjustments = None
    adjust_extra: dict = {}  # --adjust / [adjustments]: kept apart from the
    # multiple-expansion haircut, which any P/E path scenario supersedes
    if args.multiple_expansion is not None and args.pe_path is None:
        from .decompose import equity_return_decomposition

        hist = equity_return_decomposition()["multiple_expansion"]
        delta = args.multiple_expansion / 100.0 - hist
        return_adjustments = {"us_equities": delta, "us_small_cap": delta}
        print(
            f"Multiple-expansion assumption: {args.multiple_expansion:+.2f}%/yr vs "
            f"historical {hist * 100:+.2f}%/yr -> US equity returns adjusted "
            f"{delta * 100:+.2f}%/yr"
        )
    if args.adjust:
        try:
            if isinstance(args.adjust, dict):  # from a config [adjustments] table
                extra = {k: float(v) / 100.0 for k, v in args.adjust.items()}
            else:
                extra = {
                    k.strip(): float(v) / 100.0
                    for k, v in (pair.split("=", 1) for pair in args.adjust.split(","))
                }
        except ValueError:
            print(f"error: bad --adjust entry in {args.adjust!r}; expected ASSET=PCT,...")
            return 2
        unknown = [
            k for k in extra
            if k not in panel.columns or k.startswith("income_") or k == "inflation"
        ]
        if unknown:
            print(f"error: unknown asset(s) in adjustments: {unknown}")
            return 2
        held: set = set()
        for alloc in [args.allocation or {}] + [
            a.allocation or {} for a in (accounts or ())
        ]:
            held |= set(alloc)
        idle = [k for k in extra if k not in held]
        if idle and not args.optimize:
            print(f"note: adjustment asset(s) {idle} are in no allocation and "
                  "have no effect on this run")
        adjust_extra = dict(extra)
        return_adjustments = dict(return_adjustments or {})
        for k, v in extra.items():
            return_adjustments[k] = return_adjustments.get(k, 0.0) + v
    pe_points = None
    stress_points = None
    hist_me = 0.0
    if args.pe_path is not None or getattr(args, "optimize_stress", None):
        from .decompose import equity_return_decomposition

        hist_me = equity_return_decomposition()["multiple_expansion"]
    if args.pe_path is not None:
        if args.multiple_expansion is not None:
            print("note: --pe-path supersedes the multiple-expansion setting")
        try:
            pe_points = parse_pe_path(args.pe_path)
        except ValueError as e:
            print(f"error: {e}")
            return 2
        pes = " -> ".join(f"{p:g} at year {y:g}" for y, p in pe_points)
        print(
            f"P/E path: {pes} (vs historical multiple expansion "
            f"{hist_me * 100:+.2f}%/yr)"
        )
    if getattr(args, "optimize_stress", None):
        if not (args.optimize and accounts is not None):
            print("error: --optimize-stress applies to household --optimize runs")
            return 2
        try:
            stress_points = parse_pe_path(args.optimize_stress)
        except ValueError as e:
            print(f"error: {e}")
            return 2
        pes = " -> ".join(f"{p:g} at year {y:g}" for y, p in stress_points)
        print(f"Optimizer stress scenario: P/E {pes}")
    opt_tol = getattr(args, "optimize_tolerance", 0.0) or 0.0
    opt_anchor = getattr(args, "optimize_anchor", "base") or "base"
    if not 0 <= opt_tol < 100:
        print("error: --optimize-tolerance is in success-rate points (0-100)")
        return 2
    if opt_anchor == "stress" and stress_points is None:
        print("error: --optimize-anchor stress needs --optimize-stress")
        return 2
    if opt_tol > 0 and not (args.optimize and accounts is not None):
        print("error: --optimize-tolerance applies to household --optimize runs")
        return 2

    # Account type and age-based features.
    if accounts is not None:
        if args.tax_deferred:
            print("error: --tax-deferred conflicts with [[account]] sections")
            return 2
        no_tax = all(a.kind in ("roth", "529") for a in accounts)
        needs_age = any(a.kind == "traditional" for a in accounts)
    else:
        if args.tax_deferred:
            if args.account == "traditional":
                print("error: --tax-deferred (no taxes) conflicts with --account traditional")
                return 2
            if args.account == "taxable":
                args.account = "roth"
        no_tax = args.account in ("roth", "529")
        needs_age = args.account == "traditional"
    if needs_age and args.age is None:
        print("error: traditional accounts need --age (RMDs are age-based)")
        return 2

    def start_month(at_age: int | None, flag: str) -> int:
        # With --age the @N anchors are ages; without it, years from start.
        if at_age is None:
            return 0
        if args.age is None:
            return max(at_age * 12, 0)
        return max((at_age - args.age) * 12, 0)

    streams: list[IncomeStream] = []
    expenses: list[tuple[int, float]] = []
    try:
        for text in args.income:
            amt, at = parse_at_age(text)
            streams.append(IncomeStream(amt, start_month(at, "--income")))
        for text in args.pension:
            amt, at = parse_at_age(text)
            streams.append(IncomeStream(amt, start_month(at, "--pension"), taxable=True))
        for text in args.expense:
            amt, at = parse_at_age(text)
            if at is None:
                raise ValueError(
                    "--expense needs AMOUNT@N (an age with --age, else a year)"
                )
            expenses.append((start_month(at, "--expense"), amt))
    except ValueError as e:
        print(f"error: {e}")
        return 2
    # Tax-free accounts drop the tax settings - unless a pension needs the
    # regime to be taxed through.
    strip_tax = no_tax and not args.pension

    if args.withdraw and ":" in args.withdraw:
        if args.withdraw_strategy != "fixed-real":
            print("error: withdrawal schedules only work with the fixed-real strategy")
            return 2
        try:
            withdrawal = Withdrawal(
                "fixed_real",
                schedule=parse_schedule(args.withdraw, args.age or 0, args.initial),
            )
        except ValueError as e:
            print(f"error: {e}")
            return 2
    else:
        withdrawal = parse_withdrawal(args.withdraw, args.withdraw_strategy)
    if args.gross:
        if withdrawal.kind != "fixed_real":
            print("error: --gross requires a fixed-real withdrawal")
            return 2
        withdrawal.gross_of_tax = True
    if args.spend_decline is not None:
        if withdrawal.kind != "fixed_real":
            print("error: --spend-decline requires a fixed-real withdrawal")
            return 2
        try:
            rate, at = parse_at_age(args.spend_decline)
            withdrawal.decline = rate / 100.0
            withdrawal.decline_start_month = start_month(at, "--spend-decline")
        except ValueError as e:
            print(f"error: {e}")
            return 2
    if args.flex is not None:
        if withdrawal.kind != "fixed_real":
            print("error: --flex requires a fixed-real withdrawal (e.g. --withdraw 4%)")
            return 2
        withdrawal.flex_floor = args.flex / 100.0
    horizons = [int(h) for h in str(args.horizons).split(",")]
    real = not args.nominal

    from .report import print_summary, save_report  # defer matplotlib import

    if args.tips_ladder is not None and (
        withdrawal.kind != "fixed_real" or withdrawal.schedule is not None
    ):
        print(
            "error: --tips-ladder requires a plain fixed-real withdrawal "
            "(e.g. --withdraw 4%), not a schedule"
        )
        return 2

    # Allocation-based TIPS ladders (the 'tips_ladder' asset name).
    all_allocs = [args.allocation or {}] + (
        [a.allocation or {} for a in accounts] if accounts else []
    )
    has_lad_alloc = any(a and "tips_ladder" in a for a in all_allocs)
    ladder_curve = None
    if has_lad_alloc:
        if args.tips_ladder is not None:
            print("error: use either --tips-ladder or a tips_ladder allocation, not both")
            return 2
        if args.tips_ladder_curve:
            from .ladder import current_real_curve

            ladder_curve = current_real_curve()

    for years in horizons:
        run_withdrawal = withdrawal
        run_initial = args.initial
        ladder = None
        if args.tips_ladder is not None:
            from dataclasses import replace

            from .ladder import build_ladder, build_ladder_curve, current_real_curve

            # A traditional account's ladder is bought with IRA money, so it
            # lives inside the IRA: the engine taxes its payouts as
            # distributions rather than as phantom income.
            ladder_taxable = (
                not no_tax and not args.tips_ladder_deferred
                and args.account != "traditional"
            )
            if args.tips_ladder_curve:
                ladder = build_ladder_curve(
                    args.tips_ladder, years, current_real_curve(), taxable=ladder_taxable
                )
            else:
                ladder = build_ladder(
                    args.tips_ladder, years, args.tips_ladder_yield / 100.0,
                    taxable=ladder_taxable,
                )
            if ladder.cost >= args.initial:
                print(
                    f"error: a {years}y ${args.tips_ladder:,.0f}/yr ladder costs "
                    f"${ladder.cost:,.0f} at {args.tips_ladder_yield}% real — more than "
                    f"the ${args.initial:,.0f} starting balance"
                )
                return 2
            run_initial = args.initial - ladder.cost
            target = withdrawal.amount or withdrawal.rate * args.initial
            run_withdrawal = replace(
                withdrawal, rate=0.0, amount=max(target - ladder.annual, 0.0)
            )
        def scenario_fields(points) -> dict:
            """SimConfig fields that drive US equity valuation along a P/E
            path: conditioned block sampling with re-centering when
            --pe-conditioned is set, otherwise a per-month mean shift net of
            the historical multiple expansion. A path supersedes any
            --multiple-expansion haircut (the path IS the valuation
            assumption), so it starts from the --adjust extras alone."""
            if not points:
                return dict(return_adjustments=return_adjustments)
            if args.pe_conditioned:
                import numpy as np

                from .decompose import shiller_pe_series

                rates = pe_path_rates(points, years * 12)
                levels = points[0][1] * np.exp(np.concatenate(
                    [[0.0], np.cumsum(rates[:-1] / 12.0)]
                ))
                return dict(
                    return_adjustments=dict(adjust_extra) or None,
                    state_series=shiller_pe_series(),
                    state_path=levels,
                    state_bandwidth=args.pe_bandwidth,
                    state_adjust_assets=("us_equities", "us_small_cap"),
                )
            arr = pe_path_rates(points, years * 12) - hist_me
            adj = dict(adjust_extra)
            for a in ("us_equities", "us_small_cap"):
                adj[a] = adj.get(a, 0.0) + arr
            return dict(return_adjustments=adj)

        scenario = scenario_fields(pe_points)
        run_adjustments = scenario.pop("return_adjustments")
        state_kw = scenario
        cfg = SimConfig(
            allocation=(
                args.allocation
                if accounts is not None
                else args.allocation or {"us_equities": 1.0}  # replaced by --optimize
            ),
            accounts=accounts,
            withdraw_order=tuple(getattr(args, "withdraw_order", None) or ()) or None,
            allocation_end=args.glide_to,
            glide_years=args.glide_years,
            initial=run_initial,
            years=years,
            n_sims=args.sims,
            withdrawal=run_withdrawal,
            ladder=ladder,
            account=args.account,
            age=args.age,
            early_penalty=not args.no_early_penalty,
            ladder_yield=args.tips_ladder_yield / 100.0,
            ladder_curve=ladder_curve,
            ladder_years=getattr(args, "ladder_years", None),
            ladder_tail_yield=(
                None if args.tips_ladder_tail is None
                else args.tips_ladder_tail / 100.0
            ),
            income=tuple(streams) or None,
            expenses=tuple(expenses) or None,
            tax_rate=0.0 if strip_tax else args.tax_rate / 100.0,
            tax_ordinary=(
                None if strip_tax or args.tax_ordinary is None else args.tax_ordinary / 100.0
            ),
            # Default: fully taxable under federal brackets (as distribution
            # taxation for traditional accounts). Flat rates or a tax-free
            # account switch that off.
            tax_brackets=(
                None
                if strip_tax or args.tax_rate or args.tax_ordinary is not None
                else args.filing
            ),
            cost_basis_start=args.cost_basis,
            rebalance_months=args.rebalance,
            state_tax=0.0 if strip_tax else args.state_tax / 100.0,
            return_adjustments=run_adjustments,
            **state_kw,
            fee_annual=args.fees / 100.0,
            contribution_monthly=args.contribute,
            block_months=args.block,
            mode=args.mode,
            sample_start=args.start,
            sample_end=args.end,
            seed=args.seed,
        )
        if args.optimize and accounts is not None:
            from dataclasses import replace

            from .optimize import optimize_household

            grid = args.optimize_grid
            e_vals = L_vals = None
            if "equity" in grid:
                lo, hi, st = grid["equity"]
                e_vals, x = [], lo
                while x <= hi + st * 1e-9:
                    e_vals.append(x / 100.0)
                    x += st
            if "ladder" in grid:
                lo, hi, st = grid["ladder"]
                L_vals, x = [], lo
                while x <= hi + st * 1e-9:
                    L_vals.append(x)
                    x += st
            n_cand = len(e_vals or [1]) * len(L_vals or [1])
            stress_cfg = None
            if stress_points is not None:
                sf = scenario_fields(stress_points)
                stress_cfg = replace(
                    cfg, return_adjustments=sf.pop("return_adjustments"),
                    state_series=sf.get("state_series"), state_path=sf.get("state_path"),
                    state_bandwidth=sf.get("state_bandwidth", cfg.state_bandwidth),
                    state_adjust_assets=sf.get("state_adjust_assets"),
                )
            print(f"\nSearching household allocations for the {years}-year "
                  f"horizon ({n_cand} candidates, then refinement)...")
            best, board, screened = optimize_household(
                panel, cfg, e_vals, L_vals, progress=lambda s: print(s, flush=True),
                stress=stress_cfg, success_tolerance=opt_tol / 100.0,
                anchor=opt_anchor, return_screen=True,
            )
            has_stress = stress_cfg is not None
            print("\nRefined candidates (success mean ± sd across seeds"
                  + (" · stress success" if has_stress else "")
                  + " · real terminal p5/median · floor):")
            for row in board:
                line = "  %-28s %6.2f%% ± %4.2f" % (
                    row["label"], row["success"] * 100, row["success_sd"] * 100)
                if has_stress:
                    line += " · stress %6.2f%%" % (row["stress_success"] * 100)
                line += " · $%.2fM / $%.1fM · $%s/yr" % (
                    row["p5"] / 1e6, row["median"] / 1e6, f"{row['floor']:,.0f}")
                print(line)
            if opt_tol > 0:
                from .optimize import tolerance_picks

                print(f"\nPick by success tolerance ({opt_anchor}-anchored band, "
                      "highest median estate within the band; from the screen):")
                for t, row in tolerance_picks(screened, opt_anchor).items():
                    line = f"  {t * 100:.0f} pt{'s' if t > 0.01 else ' '}: {row['label']:<28} " \
                           f"success {row['success']:.1%}"
                    if has_stress:
                        line += f", stress {row['stress_success']:.1%}"
                    line += f", estate ${row['median'] / 1e6:.1f}M, floor ${row['floor']:,.0f}/yr"
                    print(line)
                chosen = next(r["label"] for r in board if r["accounts"] == best)
                print(f"Selected at {opt_tol:g} pt{'s' if opt_tol != 1 else ''}: {chosen}")
            else:
                print(f"Selected: {board[0]['label']}")
            cfg = replace(cfg, accounts=best)
        elif args.optimize:
            from dataclasses import replace

            from .optimize import optimize

            print(f"\nSearching allocations for the {years}-year horizon "
                  "(175 candidates, then refinement)...")
            best, board = optimize(panel, cfg, progress=lambda s: print(s, flush=True))
            print("\nTop allocations (success mean ± sd across seeds · real terminal p5/median/sd):")
            for row in board[:5]:
                print(
                    "  %-44s %6.2f%% ± %4.2f · $%.2fM / $%.2fM / $%.2fM"
                    % (row["label"], row["success"] * 100, row["success_sd"] * 100,
                       row["terminal_p5"] / 1e6, row["terminal_median"] / 1e6,
                       row["terminal_sd"] / 1e6)
                )
            print("Selected: " + ", ".join(f"{k}={v * 100:.1f}%" for k, v in best.items()))
            cfg = replace(cfg, allocation=best)

        result = simulate(panel, cfg)
        print_summary(result, real=real)
        import pandas as pd

        if result.window.min() > pd.Period(args.start, freq="M"):
            limiting = max(
                _held_assets(cfg, panel),
                key=lambda a: panel[a].dropna().index.min(),
            )
            print(
                f"  Note: sampling starts {result.window.min()} (not {args.start}) because "
                f"'{limiting}' has no earlier data. Run 'poorcast assets' for coverage."
            )
        if not args.no_charts:
            tag = f"forecast_{years}y"
            path = save_report(result, Path(args.out), tag, real=real)
            print(f"  Charts: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
