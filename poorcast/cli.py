"""poorcast command-line interface.

Examples:
  poorcast fetch
  poorcast assets
  poorcast run --allocation us_equities=60,intl_equities=20,us_bonds_10yr=15,cash=5 \\
               --initial 1000000 --withdraw 3% --horizons 20,30,40
  poorcast run --allocation us_equities=100 --contribute 2000 --horizons 25 --no-charts
  poorcast run --allocation us_equities=60,us_bonds_10yr=40 --withdraw 4% --mode historical
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import data as data_mod
from .simulate import SimConfig, Withdrawal, simulate


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


def build_parser() -> argparse.ArgumentParser:
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

    r = sub.add_parser("run", help="run a simulation")
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
    r.add_argument("--initial", type=float, default=1_000_000, help="starting balance (default 1,000,000)")
    r.add_argument(
        "--glide-to",
        type=parse_allocation,
        default=None,
        help="glidepath: drift linearly from --allocation to this mix (same assets), "
        "e.g. a rising-equity 'bond tent'",
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
        help="annual withdrawal: '3%%' (percent rule) or '40000' (dollars/yr, inflation-adjusted)",
    )
    r.add_argument(
        "--withdraw-strategy",
        choices=["fixed-real", "percent-of-balance"],
        default="fixed-real",
        help="fixed-real: %% of initial balance, inflation-adjusted (the classic 3%%/4%% rule). "
        "percent-of-balance: %% of current balance, recomputed yearly (never fully depletes)",
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
        "--tax-deferred",
        action="store_true",
        help="treat the portfolio as tax-deferred (IRA/401k): no taxes modeled",
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
        "--rebalance",
        type=int,
        default=3,
        metavar="MONTHS",
        help="months between rebalances (default 3 = quarterly; 12 = annual, "
        "a large value = never)",
    )
    r.add_argument("--start", default="1960-01", help="earliest history to sample (default 1960-01)")
    r.add_argument("--end", default=None, help="latest history to sample (default: all)")
    r.add_argument("--seed", type=int, default=None, help="random seed for reproducibility")
    r.add_argument("--nominal", action="store_true", help="report nominal dollars instead of real")
    r.add_argument("--out", default="out", help="directory for charts (default ./out)")
    r.add_argument("--no-charts", action="store_true", help="skip chart generation")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

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

    if (args.allocation is None) == (not args.optimize):
        print("error: provide exactly one of --allocation or --optimize")
        return 2
    if args.optimize and args.glide_to is not None:
        print("error: --optimize searches static allocations; drop --glide-to")
        return 2

    return_adjustments = None
    if args.multiple_expansion is not None:
        from .decompose import equity_return_decomposition

        hist = equity_return_decomposition()["multiple_expansion"]
        delta = args.multiple_expansion / 100.0 - hist
        return_adjustments = {"us_equities": delta, "us_small_cap": delta}
        print(
            f"Multiple-expansion assumption: {args.multiple_expansion:+.2f}%/yr vs "
            f"historical {hist * 100:+.2f}%/yr -> US equity returns adjusted "
            f"{delta * 100:+.2f}%/yr"
        )

    withdrawal = parse_withdrawal(args.withdraw, args.withdraw_strategy)
    if args.flex is not None:
        if withdrawal.kind != "fixed_real":
            print("error: --flex requires a fixed-real withdrawal (e.g. --withdraw 4%)")
            return 2
        withdrawal.flex_floor = args.flex / 100.0
    horizons = [int(h) for h in str(args.horizons).split(",")]
    real = not args.nominal

    from .report import print_summary, save_report  # defer matplotlib import

    if args.tips_ladder is not None and withdrawal.kind != "fixed_real":
        print("error: --tips-ladder requires a fixed-real withdrawal (e.g. --withdraw 4%)")
        return 2

    for years in horizons:
        run_withdrawal = withdrawal
        run_initial = args.initial
        ladder = None
        if args.tips_ladder is not None:
            from dataclasses import replace

            from .ladder import build_ladder, build_ladder_curve, current_real_curve

            ladder_taxable = not args.tax_deferred and not args.tips_ladder_deferred
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
        cfg = SimConfig(
            allocation=args.allocation or {"us_equities": 1.0},  # replaced by --optimize
            allocation_end=args.glide_to,
            glide_years=args.glide_years,
            initial=run_initial,
            years=years,
            n_sims=args.sims,
            withdrawal=run_withdrawal,
            ladder=ladder,
            tax_rate=args.tax_rate / 100.0,
            tax_ordinary=None if args.tax_ordinary is None else args.tax_ordinary / 100.0,
            # Default: fully taxable under federal brackets. Flat rates or
            # --tax-deferred switch that off.
            tax_brackets=(
                None
                if args.tax_deferred or args.tax_rate or args.tax_ordinary is not None
                else args.filing
            ),
            cost_basis_start=args.cost_basis,
            rebalance_months=args.rebalance,
            state_tax=0.0 if args.tax_deferred else args.state_tax / 100.0,
            return_adjustments=return_adjustments,
            contribution_monthly=args.contribute,
            block_months=args.block,
            mode=args.mode,
            sample_start=args.start,
            sample_end=args.end,
            seed=args.seed,
        )
        if args.optimize:
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
            limiting = max(cfg.allocation, key=lambda a: panel[a].dropna().index.min())
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
