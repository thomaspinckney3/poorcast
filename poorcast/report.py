"""Terminal summary and chart output for simulation results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from .simulate import DEFAULT_WITHDRAW_ORDER, SimResult, summarize, total_initial

# Reference dataviz palette (light mode): sequential blue ramp + ink tokens.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE_100 = "#cde2fb"
BLUE_200 = "#9ec5f4"
BLUE_400 = "#3987e5"
BLUE_500 = "#256abf"
BLUE_650 = "#104281"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Helvetica", "Arial"],
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_2,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.axisbelow": True,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "figure.dpi": 150,
    }
)


def _dollars(x: float, _pos=None) -> str:
    ax = abs(x)
    if ax >= 1e9:
        return f"${x / 1e9:.3g}B"
    if ax >= 1e6:
        return f"${x / 1e6:.3g}M"
    if ax >= 1e3:
        return f"${x / 1e3:.3g}k"
    return f"${x:.3g}"


def fan_chart(result: SimResult, ax=None, real: bool = True):
    """Percentile bands of portfolio balance over time (sequential blue ramp)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))
    bal = result.real_balance if real else result.balance
    years = np.arange(bal.shape[1]) / 12.0
    p5, p25, p50, p75, p95 = np.percentile(bal, [5, 25, 50, 75, 95], axis=0)

    ax.fill_between(years, p5, p95, color=BLUE_100, linewidth=0, label="5th–95th pct")
    ax.fill_between(years, p25, p75, color=BLUE_200, linewidth=0, label="25th–75th pct")
    ax.plot(years, p50, color=BLUE_500, linewidth=2, label="Median")

    for y_end, txt in [(p95[-1], "95th"), (p50[-1], "median"), (p5[-1], "5th")]:
        ax.annotate(
            f" {txt}: {_dollars(y_end)}",
            (years[-1], y_end),
            fontsize=8,
            color=INK_2,
            va="center",
        )
    ax.set_xlim(0, years[-1] * 1.14)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_dollars))
    ax.set_xlabel("Years")
    kind = "real (today's dollars)" if real else "nominal"
    ax.set_title(f"Portfolio balance over {result.config.years} years, {kind}")
    ax.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK_2)
    return ax


def survival_chart(result: SimResult, ax=None):
    """Probability the portfolio is not yet depleted, by month."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3.2))
    n_months = result.balance.shape[1] - 1
    dep = result.depleted_month
    alive = np.ones(n_months + 1)
    for m in range(n_months + 1):
        alive[m] = np.mean((dep < 0) | (dep >= m))
    years = np.arange(n_months + 1) / 12.0
    ax.plot(years, alive * 100, color=BLUE_500, linewidth=2)
    ax.annotate(
        f" {alive[-1] * 100:.1f}%",
        (years[-1], alive[-1] * 100),
        fontsize=9,
        color=INK_2,
        va="center",
    )
    ax.set_xlim(0, years[-1] * 1.08)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}%"))
    ax.set_xlabel("Years")
    ax.set_title("Chance the portfolio is still solvent")
    return ax


def failure_year_stats(result: SimResult) -> dict | None:
    """Timing of depletion among failed paths, in 1-based years. None if no failures."""
    dep = result.depleted_month
    failed = dep[dep >= 0]
    if len(failed) == 0:
        return None
    years = failed // 12 + 1
    return {
        "share": len(failed) / result.n_paths,
        "earliest": int(years.min()),
        "median": int(np.median(years)),
        "p10": int(np.percentile(years, 10)),  # 90% of failures happen after this
    }


def sequence_risk_stats(result: SimResult) -> dict | None:
    """Attribution of outcomes to the first 5 years' exogenous market return:
    share of terminal-wealth variance explained (R^2 among surviving paths) and
    failure odds after a worst-quintile start. None if not computable."""
    x = result.early_real_market
    if x is None or result.n_paths < 200 or np.std(x) < 1e-9:
        return None
    terminal = result.real_balance[:, -1]
    ok = terminal > 0
    out = {"equal_share": 5 * 12 / (result.balance.shape[1] - 1)}
    if ok.sum() > 100 and np.std(np.log(terminal[ok])) > 1e-9:
        out["terminal_r2"] = float(np.corrcoef(x[ok], np.log(terminal[ok]))[0, 1] ** 2)
    fail = ~ok
    if fail.any():
        q20 = np.quantile(x, 0.2)
        out["fail_rate_bad_start"] = float(fail[x <= q20].mean())
        out["fail_rate_base"] = float(fail.mean())
        out["failures_with_bad_start"] = float((x[fail] <= q20).mean())
    return out if len(out) > 1 else None


def failure_hist(result: SimResult, ax=None):
    """Distribution of the year in which failing paths run out of money."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3.2))
    stats = failure_year_stats(result)
    dep = result.depleted_month
    fail_years = dep[dep >= 0] // 12 + 1
    horizon = result.config.years
    bins = np.arange(0.5, horizon + 1.5)
    weights = np.full(len(fail_years), 100.0 / result.n_paths)
    ax.hist(fail_years, bins=bins, weights=weights, color=BLUE_400,
            edgecolor=SURFACE, linewidth=0.8)
    ax.axvline(stats["median"], color=INK_2, linewidth=1, linestyle="--")
    ax.annotate(
        f"median failure: year {stats['median']}  ",
        (stats["median"], ax.get_ylim()[1] * 1.0),
        fontsize=8,
        color=INK_2,
        ha="right",
        va="top",
    )
    ax.set_xlim(0, horizon + 1)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}%"))
    ax.set_xlabel("Year the money runs out")
    ax.set_ylabel("Share of all paths")
    ax.set_title(f"When failing paths fail ({stats['share']:.1%} of paths deplete)")
    ax.grid(axis="x", visible=False)
    return ax


def terminal_hist(result: SimResult, ax=None, real: bool = True):
    """Distribution of terminal wealth (log-scaled dollars)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3.2))
    bal = result.real_balance if real else result.balance
    terminal = bal[:, -1]
    positive = terminal[terminal > 0]
    depleted_share = 1 - len(positive) / len(terminal)
    if len(positive) == 0:
        ax.set_title("Terminal wealth: every path depleted")
        ax.set_ylabel("Paths")
        return ax

    lo = np.percentile(positive, 0.5)
    hi = np.percentile(positive, 99.5)
    bins = np.geomspace(max(lo, 1), hi, 40)
    ax.hist(positive, bins=bins, color=BLUE_400, edgecolor=SURFACE, linewidth=0.8)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_dollars))
    initial = total_initial(result.config)
    ax.axvline(initial, color=INK_2, linewidth=1, linestyle="--")
    ax.annotate(
        "  starting\n  balance",
        (initial, ax.get_ylim()[1] * 0.82),
        fontsize=8,
        color=INK_2,
    )
    med = float(np.median(terminal))
    title = f"Terminal wealth after {result.config.years} years (median {_dollars(med)}"
    if depleted_share > 0:
        title += f", {depleted_share:.1%} of paths depleted"
    title += ")"
    ax.set_title(title)
    ax.set_ylabel("Paths")
    ax.grid(axis="x", visible=False)
    return ax


def save_report(result: SimResult, out_dir: Path, tag: str, real: bool = True) -> Path:
    """One PNG per horizon: fan chart, survival (if withdrawing), histogram."""
    out_dir.mkdir(parents=True, exist_ok=True)
    withdrawing = result.config.withdrawal.kind != "none"
    has_failures = failure_year_stats(result) is not None
    n_rows = 2 + (1 if withdrawing else 0) + (1 if withdrawing and has_failures else 0)
    heights = [4.2] + [2.6] * (n_rows - 1)
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(8.5, sum(heights) + 1.2), height_ratios=heights
    )
    axes = np.atleast_1d(axes)
    fan_chart(result, axes[0], real=real)
    if withdrawing:
        survival_chart(result, axes[1])
        if has_failures:
            failure_hist(result, axes[2])
    terminal_hist(result, axes[-1], real=real)
    fig.suptitle(_describe(result, wrap=True), fontsize=9, color=INK_2, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = out_dir / f"{tag}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _describe(result: SimResult, wrap: bool = False) -> str:
    cfg = result.config

    def _alloc_str(alloc_dict):
        return ", ".join(f"{int(round(v * 100))}% {k}" for k, v in alloc_dict.items())

    def _when(start_month: int) -> str:
        if start_month <= 0:
            return "start"
        if cfg.age is not None:
            return f"age {cfg.age + start_month // 12}"
        return f"year {start_month // 12}"

    if cfg.accounts:
        parts_a = []
        for a in cfg.accounts:
            desc = f"${a.balance:,.0f} {a.kind}"
            if a.allocation:
                desc += f" ({_alloc_str(a.allocation)})"
            if a.schedule:
                steps = ", ".join(
                    f"${amt:,.0f}/yr from {_when(sm)}" for sm, amt in a.schedule if amt
                )
                desc += f" drawing {steps}"
            parts_a.append(desc)
        alloc = " + ".join(parts_a)
        order = cfg.withdraw_order or DEFAULT_WITHDRAW_ORDER
        kinds = {a.kind for a in cfg.accounts}
        alloc += " · spend " + " then ".join(k for k in order if k in kinds)
    else:
        alloc = _alloc_str(cfg.allocation)
    if cfg.allocation_end is not None:
        end = ", ".join(f"{int(round(v * 100))}% {k}" for k, v in cfg.allocation_end.items())
        span = f"{cfg.glide_years}y" if cfg.glide_years else "full horizon"
        alloc = f"{alloc} gliding to {end} over {span}"
    w = cfg.withdrawal
    if w.kind == "fixed_real":
        if w.schedule:
            steps = ", ".join(
                f"${amt:,.0f}/yr from {_when(sm)}" for sm, amt in w.schedule
            )
            wd = f"withdrawing {steps} (inflation-adjusted)"
        elif not w.amount:
            wd = f"withdrawing {w.rate:.1%}/yr of initial (inflation-adjusted)"
        else:
            wd = f"withdrawing ${w.amount:,.0f}/yr (inflation-adjusted)"
        if w.decline:
            wd += f", spending declining {w.decline:.1%}/yr"
            if w.decline_start_month > 0:
                wd += f" from {_when(w.decline_start_month)}"
        if w.flex_floor is not None:
            wd += f", flexed down to {w.flex_floor:.0%} in down markets"
    elif w.kind == "percent_of_balance":
        wd = f"withdrawing {w.rate:.1%}/yr of current balance"
    else:
        wd = "no withdrawals"
    for s in cfg.income or ():
        kind = "pension" if s.taxable else "income"
        wd += f" · {kind} ${s.annual:,.0f}/yr from {_when(s.start_month)}"
    for em, amt in cfg.expenses or ():
        wd += f" · ${amt:,.0f} expense at {_when(em)}"
    if cfg.account == "traditional":
        regime = (
            f"federal brackets ({cfg.tax_brackets} filer)"
            if cfg.tax_brackets is not None
            else f"{(cfg.tax_ordinary if cfg.tax_ordinary is not None else cfg.tax_rate):.0%} flat"
        )
        state = f" + {cfg.state_tax:.0%} state" if cfg.state_tax else ""
        wd += (
            f" · traditional IRA: distributions taxed as ordinary income "
            f"({regime}{state}), RMDs from 73"
        )
    elif cfg.account in ("roth", "529"):
        wd += " · " + (
            "Roth (tax-free)" if cfg.account == "roth" else "529 (tax-free, qualified use)"
        )
    elif cfg.tax_brackets is not None:
        state = f" + {cfg.state_tax:.0%} state" if cfg.state_tax else ""
        wd += (
            f" · taxable, federal brackets ({cfg.tax_brackets} filer, 2026 law, "
            f"indexed){state}"
            + (f", basis {cfg.cost_basis_start:.0%}" if cfg.cost_basis_start != 1 else "")
        )
    elif cfg.tax_rate > 0 or (cfg.tax_ordinary or 0) > 0:
        ord_rate = cfg.tax_ordinary if cfg.tax_ordinary is not None else cfg.tax_rate
        wd += (
            f" · taxes {cfg.tax_rate:.0%} gains/div, {ord_rate:.0%} interest"
            + (f", basis {cfg.cost_basis_start:.0%}" if cfg.cost_basis_start != 1 else "")
        )
    if cfg.ladder is not None:
        lad = cfg.ladder
        acct = "taxable" if getattr(lad, "taxable", False) else "tax-deferred"
        wd += (
            f" · TIPS ladder floor ${lad.annual:,.0f}/yr "
            f"({lad.years}y at {lad.real_yield:.2%} real, cost ${lad.cost:,.0f}, {acct})"
        )
    elif result.ladder_annual:
        pricing = "today's curve" if cfg.ladder_curve else f"{cfg.ladder_yield:.2%} real"
        wd += (
            f" · TIPS rungs paying ${result.ladder_annual:,.0f}/yr "
            f"({min(cfg.ladder_years or cfg.years, cfg.years)}y at {pricing})"
        )
    kinds_present = (
        {a.kind for a in cfg.accounts} if cfg.accounts else {cfg.account}
    )
    if (
        cfg.early_penalty
        and cfg.age is not None
        and cfg.age < 59.5
        and kinds_present & {"traditional", "roth"}
    ):
        wd += " · 10% penalty on early retirement-account draws (pre-59½)"
    if cfg.accounts and "529" in kinds_present:
        wd += " · unscheduled 529 draws non-qualified (earnings taxed + 10%)"
    mode = (
        f"{result.n_paths:,} bootstrap paths (block={cfg.block_months}mo)"
        if cfg.mode == "bootstrap"
        else f"all {result.n_paths} historical {cfg.years}-year windows"
    )
    if cfg.return_adjustments:
        adjs = ", ".join(
            f"{k} {v * 100:+.2f}%/yr" for k, v in cfg.return_adjustments.items()
        )
        wd += f" · return adjustment: {adjs}"
    if cfg.fee_annual:
        wd += f" · fees {cfg.fee_annual:.2%}/yr"
    rebal = f"rebalancing every {cfg.rebalance_months} months"
    if cfg.rebalance_months == 3:
        rebal = "quarterly rebalancing"
    parts = [
        f"{alloc} · start ${total_initial(cfg):,.0f}",
        f"{wd} · {rebal}",
        f"{mode} · history {result.window.min()}–{result.window.max()}",
    ]
    return "\n".join(parts) if wrap else " · ".join(parts)


def print_summary(result: SimResult, real: bool = True) -> None:
    s = summarize(result, real=real)
    cfg = result.config
    unit = "real" if real else "nominal"
    print(f"\n=== {cfg.years}-year horizon ===")
    print(f"  {_describe(result)}")
    lad_annual = (
        cfg.ladder.annual if cfg.ladder is not None else (result.ladder_annual or 0.0)
    )
    if cfg.withdrawal.kind != "none":
        if lad_annual:
            print(
                f"  Success rate (full income maintained): {s['success_rate']:.1%}"
                f"  — worst case falls back to the ${lad_annual:,.0f}/yr "
                "ladder floor, not $0"
            )
        else:
            print(f"  Success rate (never depleted): {s['success_rate']:.1%}")
        f = failure_year_stats(result)
        if f is not None:
            print(
                f"  Failed paths ({f['share']:.1%}): earliest failure year "
                f"{f['earliest']}, median year {f['median']}, "
                f"90% fail after year {f['p10']}"
            )
        lad_years = (
            cfg.ladder.years if cfg.ladder is not None
            else min(cfg.ladder_years or cfg.years, cfg.years)
        )
        ladder_income = lad_annual * lad_years
        med_wd = float(np.median(result.total_withdrawn_real)) + ladder_income
        label = "incl. ladder" if ladder_income else "real"
        print(f"  Median total income withdrawn ({label}): {_dollars(med_wd)}")
    if result.total_tax_real is not None and result.total_tax_real.any():
        print(
            f"  Median total taxes paid (real): "
            f"{_dollars(float(np.median(result.total_tax_real)))}"
        )
    seq = sequence_risk_stats(result)
    if seq is not None:
        parts = []
        if "terminal_r2" in seq:
            parts.append(
                f"explains {seq['terminal_r2']:.0%} of terminal-wealth variance "
                f"among surviving paths (time share {seq['equal_share']:.0%})"
            )
        if "fail_rate_bad_start" in seq:
            parts.append(
                f"failure rate {seq['fail_rate_bad_start']:.1%} after a "
                f"worst-quintile start vs {seq['fail_rate_base']:.1%} overall; "
                f"{seq['failures_with_bad_start']:.0%} of failures began with one"
            )
        print(f"  First-5-years market: {'; '.join(parts)}")
    print(f"  Terminal wealth ({unit}):")
    for label, key in [
        ("5th pct ", "terminal_p5"),
        ("25th pct", "terminal_p25"),
        ("median  ", "terminal_median"),
        ("75th pct", "terminal_p75"),
        ("95th pct", "terminal_p95"),
    ]:
        print(f"    {label}  {_dollars(s[key]):>10}")
    if result.account_terminal is not None:
        term = result.account_terminal
        if real:
            term = term / result.cum_inflation[:, -1:]
        med = np.median(term, axis=0)
        print(
            "  Median terminal by account: "
            + ", ".join(
                f"{k} {_dollars(v)}" for k, v in zip(result.account_kinds, med)
            )
        )
    if cfg.contribution_monthly == 0:
        print(f"  Median {unit} growth rate: {s['median_cagr']:.2%}/yr")
        print(f"  Chance of ending below start ({unit}): {s['prob_loss']:.1%}")
