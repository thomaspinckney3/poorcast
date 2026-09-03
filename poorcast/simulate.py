"""Monte Carlo engine: joint block-bootstrap of historical months, monthly steps,
quarterly rebalancing, withdrawal strategies.

All asset columns and inflation are sampled *jointly* (the same historical months
for every series in a path), so cross-asset correlations and return/inflation
co-movement (e.g. the 1970s) survive into the simulation. Block sampling keeps
serial structure (momentum, multi-year regimes) at the block length scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Withdrawal:
    """Withdrawal policy, applied monthly.

    kind:
      'none'               – accumulation only.
      'fixed_real'         – the classic X% rule: X% of the *initial* balance per
                             year (or a fixed dollar amount), inflation-adjusted
                             along each path, withdrawn in monthly installments.
      'percent_of_balance' – X% of the *current* balance, recomputed every 12
                             months, withdrawn in monthly installments.

    flex_floor (fixed_real only): tighten the belt in down markets. Each month
    the withdrawal is scaled by current real balance / initial balance, capped
    at 1 and floored at flex_floor (e.g. 0.75 = never below 75% of target).
    Above water you take the full target; under water you take proportionally
    less, but never less than the floor.
    """

    kind: str = "none"
    rate: float = 0.0  # annual, e.g. 0.03
    amount: float = 0.0  # annual dollars, alternative to rate for fixed_real
    flex_floor: float | None = None  # e.g. 0.75; None = always withdraw target
    # Age-varying spending (fixed_real only): sorted (start_month, annual real
    # dollars) steps; each amount applies from its start month until the next
    # step (the last runs to the horizon). Overrides rate/amount when set.
    schedule: tuple[tuple[int, float], ...] | None = None


@dataclass(frozen=True)
class IncomeStream:
    """An inflation-adjusted income stream outside the portfolio (Social
    Security, pension, annuity). Offsets portfolio withdrawals from
    start_month on; any surplus is invested like a contribution. taxable=True
    treats it as ordinary income in the active tax regime (a pension);
    False models an after-tax amount (Social Security, roughly)."""

    annual: float  # real dollars per year
    start_month: int = 0
    taxable: bool = False


@dataclass
class SimConfig:
    allocation: dict[str, float]  # asset -> weight, must sum to 1
    # Optional glidepath: target weights drift linearly from `allocation` to
    # `allocation_end` over the first `glide_years` years, then hold. Same asset
    # set as `allocation`. Rebalancing tracks the moving target.
    allocation_end: dict[str, float] | None = None
    glide_years: int | None = None
    # Optional state-dependent rule. Called at each rebalance with a RuleState;
    # must return target weights of shape (n_paths, n_assets) in the order of
    # `allocation`'s keys. Overrides the static/glidepath targets.
    allocation_rule: "callable | None" = None
    # Informational: a TIPS ladder bought at t=0, held outside the simulated
    # portfolio. The engine ignores it (the CLI already deducted its cost from
    # `initial` and its floor from the withdrawal); reporting uses it to state
    # floor-aware success and income figures.
    ladder: "object | None" = None  # LadderSpec
    # Taxable-account modeling: long-term capital-gains rate applied to gains
    # realized by rebalancing, withdrawals, and tax payments themselves.
    # Average-cost basis per asset; gains netted within each quarter, losses
    # carried forward; tax settled and paid from the portfolio at each
    # rebalance. Dividend/interest tax is NOT modeled (returns are total
    # returns). 0 = tax-deferred account (default).
    tax_rate: float = 0.0
    # Ordinary-income rate for interest (bond coupons, T-bill income). Dividends
    # are taxed at tax_rate (qualified). None = same as tax_rate.
    tax_ordinary: float | None = None
    # US federal bracket mode ('single' or 'married'): progressive 2026 brackets
    # with LTCG stacking, standard deduction, and NIIT, settled annually with
    # inflation-indexed thresholds. Mutually exclusive with the flat rates.
    tax_brackets: str | None = None
    # Flat state income-tax rate on dividends and realized gains (Treasury
    # interest is state-exempt; muni income exempt from both levels).
    state_tax: float = 0.0
    # Additive annual return adjustment per asset (decimal, applied as /12 to
    # every sampled month) - e.g. removing the historical multiple-expansion
    # tailwind from US equities. {'us_equities': -0.005} = -50bp/yr.
    return_adjustments: dict[str, float] | None = None
    # Annual management/expense fee (decimal, e.g. 0.001 = 10bp) applied to
    # every asset as a monthly return drag of fee/12 - fund expense ratios,
    # or an advisor fee. Historical returns are index returns, so 0 models
    # free investing.
    fee_annual: float = 0.0
    cost_basis_start: float = 1.0  # initial basis as fraction of starting value
    # Account type: 'taxable' (default) uses the full basis/income machinery
    # above. 'traditional' (IRA/401k) taxes nothing inside the account but
    # taxes every distribution as ordinary income (through tax_brackets, or a
    # flat tax_ordinary/tax_rate) and enforces RMDs from age 73 (requires
    # `age`). RMD dollars beyond spending are deemed distributed and taxed but
    # stay invested - approximating reinvestment in a taxable account whose
    # own future drag is ignored. 'roth' and '529' (qualified use) are
    # tax-free: no taxes modeled at all.
    account: str = "taxable"
    age: int | None = None  # age at t=0; enables age-based features (RMDs)
    # Income streams outside the portfolio (Social Security, pensions).
    income: tuple[IncomeStream, ...] | None = None
    # One-time real expenses: (month index, today's dollars) - a roof, a car.
    expenses: tuple[tuple[int, float], ...] | None = None
    initial: float = 1_000_000.0
    years: int = 30
    n_sims: int = 10_000
    withdrawal: Withdrawal = field(default_factory=Withdrawal)
    contribution_monthly: float = 0.0  # real (inflation-adjusted) monthly contribution
    block_months: int = 24  # bootstrap block length
    mode: str = "bootstrap"  # or 'historical': every actual overlapping window
    sample_start: str = "1960-01"  # earliest historical month to sample from
    sample_end: str | None = None
    rebalance_months: int = 3  # quarterly
    seed: int | None = None


@dataclass
class RuleState:
    """What an allocation_rule sees at a rebalance point."""

    month: int  # 0-based month index just completed
    n_months: int  # horizon length in months
    balance: np.ndarray  # (n_paths,) nominal balance after this month's returns
    price_level: np.ndarray  # (n_paths,) cumulative inflation factor (starts at 1)
    initial: float
    monthly_withdrawal_real: float  # target real withdrawal per month (0 if none)
    hist_indices: np.ndarray  # (n_paths,) row in the historical window for this month

    @property
    def real_balance(self) -> np.ndarray:
        return self.balance / self.price_level


@dataclass
class SimResult:
    config: SimConfig
    months: np.ndarray  # (n_paths, n_months) indices into the historical window
    balance: np.ndarray  # (n_paths, n_months+1) nominal total balance
    cum_inflation: np.ndarray  # (n_paths, n_months+1) price level, starts at 1
    depleted_month: np.ndarray  # (n_paths,) month index of depletion, -1 if never
    window: pd.PeriodIndex  # historical months the sampler drew from
    total_withdrawn: np.ndarray  # (n_paths,) nominal dollars withdrawn
    total_withdrawn_real: np.ndarray  # (n_paths,) in starting dollars
    total_tax_real: np.ndarray | None = None  # (n_paths,) taxes paid, real
    # (n_paths,) exogenous real market log-return of the first 5 years (initial
    # target weights, before withdrawals/taxes) - for sequence-risk attribution
    early_real_market: np.ndarray | None = None

    @property
    def real_balance(self) -> np.ndarray:
        return self.balance / self.cum_inflation

    @property
    def success_rate(self) -> float:
        return float((self.depleted_month < 0).mean())

    @property
    def n_paths(self) -> int:
        return self.balance.shape[0]


def _historical_matrix(
    panel: pd.DataFrame, assets: list[str], cfg: SimConfig, need_income: bool = False
) -> tuple[np.ndarray, np.ndarray, pd.PeriodIndex, np.ndarray | None]:
    """Rows = historical months in the sample window with data for every series."""
    cols = panel[assets + ["inflation"]]
    window = cols.dropna()
    window = window[window.index >= pd.Period(cfg.sample_start, freq="M")]
    if cfg.sample_end:
        window = window[window.index <= pd.Period(cfg.sample_end, freq="M")]
    if len(window) < 120:
        raise ValueError(
            f"only {len(window)} historical months available for {assets}; "
            "widen the sample window or drop an asset"
        )
    income = None
    if need_income:
        inc_cols = [f"income_{a}" for a in assets]
        missing = [c for c in inc_cols if c not in panel.columns]
        if missing:
            raise ValueError(
                f"panel lacks income columns {missing} needed for tax modeling; "
                "run 'poorcast fetch' to rebuild the data"
            )
        income = panel[inc_cols].reindex(window.index).ffill().fillna(0.0).to_numpy()
    return window[assets].to_numpy(), window["inflation"].to_numpy(), window.index, income


def _sample_months(cfg: SimConfig, t_hist: int, rng: np.random.Generator) -> np.ndarray:
    """Month indices per path: circular block bootstrap, or all historical windows."""
    n_months = cfg.years * 12
    if cfg.mode == "historical":
        n_windows = t_hist - n_months + 1
        if n_windows < 1:
            raise ValueError(
                f"horizon of {cfg.years}y exceeds the {t_hist} months of history"
            )
        starts = np.arange(n_windows)[:, None]
        return starts + np.arange(n_months)[None, :]
    block = max(1, min(cfg.block_months, t_hist))
    n_blocks = -(-n_months // block)  # ceil
    starts = rng.integers(0, t_hist, size=(cfg.n_sims, n_blocks))
    offsets = np.arange(block)
    idx = (starts[:, :, None] + offsets[None, None, :]) % t_hist  # circular
    return idx.reshape(cfg.n_sims, -1)[:, :n_months]


def simulate(panel: pd.DataFrame, cfg: SimConfig) -> SimResult:
    assets = list(cfg.allocation)
    weights = np.array([cfg.allocation[a] for a in assets], dtype=float)
    if abs(weights.sum() - 1.0) > 1e-6:
        raise ValueError(f"allocation weights sum to {weights.sum():.4f}, expected 1")
    missing = [a for a in assets if a not in panel.columns]
    if missing:
        raise ValueError(f"unknown asset(s): {missing}; available: {list(panel.columns)}")

    # Per-month target weights: static, or a linear glide over glide_years.
    n_months_total = cfg.years * 12
    if cfg.allocation_end is not None:
        if set(cfg.allocation_end) != set(assets):
            raise ValueError("allocation_end must use the same assets as allocation")
        w_end = np.array([cfg.allocation_end[a] for a in assets], dtype=float)
        if abs(w_end.sum() - 1.0) > 1e-6:
            raise ValueError(f"allocation_end weights sum to {w_end.sum():.4f}, expected 1")
        glide_m = min((cfg.glide_years or cfg.years) * 12, n_months_total)
        frac = np.minimum(np.arange(n_months_total) / max(glide_m - 1, 1), 1.0)
        target_w = weights[None, :] + frac[:, None] * (w_end - weights)[None, :]
    else:
        target_w = np.tile(weights, (n_months_total, 1))

    if cfg.account not in ("taxable", "traditional", "roth", "529"):
        raise ValueError(
            f"account must be taxable/traditional/roth/529, got {cfg.account!r}"
        )
    trad = cfg.account == "traditional"
    ord_rate = cfg.tax_ordinary if cfg.tax_ordinary is not None else cfg.tax_rate
    if cfg.tax_brackets is not None:
        if cfg.tax_brackets not in ("single", "married"):
            raise ValueError(f"tax_brackets must be 'single' or 'married', got {cfg.tax_brackets!r}")
        if cfg.tax_rate > 0 or (cfg.tax_ordinary or 0) > 0:
            raise ValueError("tax_brackets and flat tax rates are mutually exclusive")
    any_tax_setting = (
        cfg.tax_brackets is not None
        or cfg.tax_rate > 0
        or ord_rate > 0
        or cfg.state_tax > 0
    )
    if cfg.account in ("roth", "529"):
        if any_tax_setting:
            raise ValueError(f"{cfg.account} accounts are tax-free; clear the tax settings")
    if trad:
        if cfg.age is None:
            raise ValueError("traditional accounts need `age` (for RMDs)")
        if cfg.tax_brackets is None and ord_rate <= 0:
            raise ValueError(
                "traditional accounts need tax_brackets or a flat ordinary rate "
                "for distribution taxation"
            )
    taxed = cfg.account == "taxable" and any_tax_setting
    returns_hist, inflation_hist, window, income_hist = _historical_matrix(
        panel, assets, cfg, need_income=taxed
    )
    rng = np.random.default_rng(cfg.seed)
    months = _sample_months(cfg, len(window), rng)
    n_paths, n_months = months.shape

    if not 0 <= cfg.fee_annual < 0.1:
        raise ValueError(f"fee_annual must be in [0, 0.1), got {cfg.fee_annual}")
    path_returns = returns_hist[months]  # (n_paths, n_months, n_assets)
    if cfg.return_adjustments or cfg.fee_annual:
        adj = np.array(
            [(cfg.return_adjustments or {}).get(a, 0.0) for a in assets]
        )
        path_returns = path_returns + (adj - cfg.fee_annual)[None, None, :] / 12.0
    path_inflation = inflation_hist[months]  # (n_paths, n_months)
    e5 = min(60, n_months)
    early_real_market = (
        np.log1p(path_returns[:, :e5, :] @ weights) - np.log1p(path_inflation[:, :e5])
    ).sum(axis=1)
    cum_inflation = np.ones((n_paths, n_months + 1))
    np.cumprod(1 + path_inflation, axis=1, out=cum_inflation[:, 1:])

    w = cfg.withdrawal
    if w.schedule is not None and w.kind != "fixed_real":
        raise ValueError("withdrawal schedules only apply to fixed_real withdrawals")
    if w.kind == "fixed_real":
        # Per-month real spending target: constant, or a step schedule.
        if w.schedule:
            starts = np.array([s for s, _ in w.schedule], dtype=int)
            amts = np.array([a for _, a in w.schedule], dtype=float) / 12.0
            if starts[0] < 0 or (np.diff(starts) <= 0).any():
                raise ValueError("withdrawal schedule months must be increasing and >= 0")
            seg = np.searchsorted(starts, np.arange(n_months), side="right") - 1
            spend_real_m = np.where(seg >= 0, amts[np.maximum(seg, 0)], 0.0)
        else:
            spend_real_m = np.full(n_months, (w.amount or w.rate * cfg.initial) / 12.0)
    elif w.kind == "percent_of_balance":
        monthly_withdrawal = np.full(n_paths, cfg.initial * w.rate / 12.0)
    elif w.kind != "none":
        raise ValueError(f"unknown withdrawal kind {w.kind!r}")

    # Outside income streams and one-time expenses, as per-month real amounts.
    income_real_m = np.zeros(n_months)
    income_taxed_real_m = np.zeros(n_months)
    for stream in cfg.income or ():
        sm = max(int(stream.start_month), 0)
        if sm < n_months:
            income_real_m[sm:] += stream.annual / 12.0
            if stream.taxable:
                income_taxed_real_m[sm:] += stream.annual / 12.0
    expense_real_m = np.zeros(n_months)
    for em, amt in cfg.expenses or ():
        if 0 <= em < n_months:
            expense_real_m[em] += amt
    has_flows = bool(income_real_m.any() or expense_real_m.any())
    if w.flex_floor is not None:
        if w.kind != "fixed_real":
            raise ValueError("flex_floor only applies to fixed_real withdrawals")
        if not 0 < w.flex_floor <= 1:
            raise ValueError(f"flex_floor must be in (0, 1], got {w.flex_floor}")

    holdings = np.tile(weights * cfg.initial, (n_paths, 1))  # (n_paths, n_assets)
    balance = np.empty((n_paths, n_months + 1))
    balance[:, 0] = cfg.initial
    depleted_month = np.full(n_paths, -1, dtype=int)
    total_withdrawn = np.zeros(n_paths)
    total_withdrawn_real = np.zeros(n_paths)

    if not (0 <= cfg.tax_rate < 1 and 0 <= ord_rate < 1):
        raise ValueError(f"tax rates must be in [0, 1), got {cfg.tax_rate}/{ord_rate}")
    basis = holdings * cfg.cost_basis_start  # average-cost basis per asset
    realized = np.zeros(n_paths)  # gains realized since last tax settlement
    loss_carry = np.zeros(n_paths)  # <= 0, carried-forward losses
    div_acc = np.zeros(n_paths)  # dividends received since last settlement
    int_acc = np.zeros(n_paths)  # interest received since last settlement
    # Withdrawals spend the previous month's dividends/interest as cash first
    # (those dollars have basis = value, so spending them realizes no gains);
    # only the shortfall is a gain-realizing sale. Unspent income stays
    # reinvested (its basis step-up already applied).
    income_credit = np.zeros((n_paths, len(assets)))
    total_tax_real = np.zeros(n_paths)
    # Traditional-account state: nominal distributions this tax year (spending
    # withdrawals, later the tax payment itself), and the balance at the start
    # of the year that RMDs are computed from.
    dist_acc = np.zeros(n_paths)
    other_ord_acc = np.zeros(n_paths)  # taxable outside income since last settlement
    year_start_bal = np.full(n_paths, cfg.initial)
    if taxed:
        from .data import INCOME_CLASS

        classes = [INCOME_CLASS.get(a, "dividend") for a in assets]
        interest_mask = np.array([1.0 if c == "interest" else 0.0 for c in classes])
        dividend_mask = np.array([1.0 if c == "dividend" else 0.0 for c in classes])
        # anything else (muni) is tax-exempt income: spendable, never taxed
        path_income = income_hist[months]  # (n_paths, n_months, n_assets)
        # A taxable TIPS ladder throws off federal ordinary income each year:
        # its coupons plus the inflation accrual on remaining principal
        # ("phantom income"), both state-exempt Treasury interest. The tax is
        # paid from the portfolio.
        ladder_coupons = ladder_principal = None
        lad = cfg.ladder
        if lad is not None and getattr(lad, "taxable", False) and getattr(lad, "faces", ()):
            ladder_coupons = lad.coupon_income_real()
            ladder_principal = lad.remaining_principal_real()

    def prorata_flow(scale: np.ndarray) -> None:
        """Apply a pro-rata sale (scale<1) or buy (scale>1) to holdings+basis,
        booking realized gains on the sale portion."""
        nonlocal realized
        if taxed:
            selling = scale < 1
            realized += np.where(
                selling, (1 - scale) * (holdings.sum(1) - basis.sum(1)), 0.0
            )
            buy_add = np.where(scale > 1, scale - 1, 0.0)[:, None] * holdings
            basis[:] = np.where(selling[:, None], basis * scale[:, None], basis + buy_add)
        holdings[:] *= scale[:, None]

    for m in range(n_months):
        total = holdings.sum(axis=1)
        alive = total > 0
        if trad and m % 12 == 0:
            year_start_bal = total.copy()

        # Withdrawal / contribution at the start of the month, pro-rata across
        # holdings so the flow itself doesn't rebalance the portfolio.
        if w.kind == "percent_of_balance" and m > 0 and m % 12 == 0:
            monthly_withdrawal = total * w.rate / 12.0
        if w.kind == "fixed_real":
            spend = spend_real_m[m] * cum_inflation[:, m]
            if w.flex_floor is not None:
                real_balance = total / cum_inflation[:, m]
                spend *= np.clip(real_balance / cfg.initial, w.flex_floor, 1.0)
        elif w.kind == "percent_of_balance":
            spend = monthly_withdrawal.copy()
        else:
            spend = np.zeros(n_paths)
        # One-time expenses add to the need; outside income offsets it. A
        # surplus (income above spending) is invested like a contribution.
        income_invested = 0.0
        if has_flows:
            net = (
                spend
                + expense_real_m[m] * cum_inflation[:, m]
                - income_real_m[m] * cum_inflation[:, m]
            )
            income_invested = np.maximum(-net, 0.0)
            spend = np.maximum(net, 0.0)
        wd = np.where(alive, np.minimum(spend, total), 0.0)
        total_withdrawn += wd
        total_withdrawn_real += wd / cum_inflation[:, m]
        if trad:
            dist_acc += wd
        if taxed or trad:
            other_ord_acc += income_taxed_real_m[m] * cum_inflation[:, m]

        wd_to_sell = wd
        if taxed:
            # Fund the withdrawal from last month's income first: no sale, no
            # realized gain; remove exactly the basis those dollars carry.
            spendable = np.minimum(income_credit, np.minimum(holdings, basis))
            credit_total = spendable.sum(axis=1)
            use = np.minimum(wd, credit_total)
            with np.errstate(invalid="ignore", divide="ignore"):
                frac = np.where(credit_total > 0, use / credit_total, 0.0)
            sold = spendable * frac[:, None]
            holdings -= sold
            basis -= sold
            income_credit[:] = 0.0  # unspent income reverts to reinvested
            wd_to_sell = wd - use
            total = holdings.sum(axis=1)

        flow = (
            cfg.contribution_monthly * cum_inflation[:, m]
            + income_invested
            - wd_to_sell
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            scale = np.where(total > 0, (total + flow) / total, 0.0)
        prorata_flow(scale)
        newly_dead = alive & (holdings.sum(axis=1) <= 1e-9)
        depleted_month[newly_dead] = m
        holdings[newly_dead] = 0.0
        basis[newly_dead] = 0.0

        # Market returns for the month.
        holdings *= 1 + path_returns[:, m, :]
        holdings = np.maximum(holdings, 0.0)

        if taxed:
            # Dividends/interest arrive inside the total return and are
            # reinvested; tax them at each asset's income rate and step the
            # basis up by the reinvested amount (it was already taxed).
            income_amt = holdings * path_income[:, m, :]
            int_acc += income_amt @ interest_mask
            div_acc += income_amt @ dividend_mask
            basis += income_amt
            income_credit = income_amt  # spendable by next month's withdrawal
            if ladder_coupons is not None and m // 12 < len(ladder_coupons):
                t = m // 12
                int_acc += (
                    ladder_coupons[t] / 12 * cum_inflation[:, m + 1]
                    + ladder_principal[t]
                    * (cum_inflation[:, m + 1] - cum_inflation[:, m])
                )

        # Quarterly rebalance back to the (possibly gliding) target weights.
        if (m + 1) % cfg.rebalance_months == 0:
            total = holdings.sum(axis=1)
            if cfg.allocation_rule is not None:
                state = RuleState(
                    month=m,
                    n_months=n_months,
                    balance=total,
                    price_level=cum_inflation[:, m + 1],
                    initial=cfg.initial,
                    monthly_withdrawal_real=(
                        float(spend_real_m[m]) if w.kind == "fixed_real" else 0.0
                    ),
                    hist_indices=months[:, m],
                )
                targets = np.asarray(cfg.allocation_rule(state), dtype=float)
            else:
                targets = target_w[m][None, :]
            new_holdings = total[:, None] * targets
            if taxed:
                sold = np.maximum(holdings - new_holdings, 0.0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    gain_frac = np.where(holdings > 0, 1 - basis / holdings, 0.0)
                realized += (sold * gain_frac).sum(axis=1)
                with np.errstate(invalid="ignore", divide="ignore"):
                    shrink = np.where(holdings > 0, new_holdings / holdings, 0.0)
                basis[:] = np.where(
                    new_holdings < holdings,
                    basis * shrink,
                    basis + (new_holdings - holdings),
                )
            holdings = new_holdings

        # Tax settlement is independent of the rebalance schedule: annually
        # for brackets (a tax year), quarterly for flat rates.
        if taxed and (m + 1) % (12 if cfg.tax_brackets else 3) == 0:
            total = holdings.sum(axis=1)
            # Net gains against carried-forward losses, then pay the tax from
            # the portfolio (itself a pro-rata sale whose gains roll into the
            # next settlement).
            net = realized + loss_carry
            gains = np.maximum(net, 0.0)
            loss_carry = np.minimum(net, 0.0)
            realized = np.zeros(n_paths)
            if cfg.tax_brackets is None:
                tax = (
                    cfg.tax_rate * (gains + div_acc)
                    + ord_rate * int_acc
                    + (ord_rate + cfg.state_tax) * other_ord_acc
                    + cfg.state_tax * (gains + div_acc)
                )
            else:
                from .tax import annual_tax

                tax = annual_tax(
                    int_acc, div_acc + gains, cfg.tax_brackets,
                    cum_inflation[:, m + 1], state_rate=cfg.state_tax,
                    other_ordinary=other_ord_acc,
                )
            div_acc = np.zeros(n_paths)
            int_acc = np.zeros(n_paths)
            other_ord_acc = np.zeros(n_paths)
            tax = np.minimum(tax, total)
            total_tax_real += tax / cum_inflation[:, m + 1]
            with np.errstate(invalid="ignore", divide="ignore"):
                scale_t = np.where(total > 0, (total - tax) / total, 0.0)
            prorata_flow(np.where(total > 0, scale_t, 1.0))

        # Traditional IRA/401k: distributions (spending withdrawals, deemed
        # RMD shortfalls, last year's tax payment) are ordinary income, taxed
        # annually and paid from the portfolio. RMD dollars beyond spending
        # are taxed but stay invested (approximating reinvestment in a
        # taxable account whose own future drag is ignored).
        if trad and (m + 1) % 12 == 0:
            from .tax import RMD_START_AGE, annual_tax, rmd_period

            total = holdings.sum(axis=1)
            age_now = cfg.age + (m + 1) // 12 - 1  # age attained this sim year
            deemed = 0.0
            if age_now >= RMD_START_AGE:
                rmd = year_start_bal / rmd_period(age_now)
                deemed = np.maximum(rmd - dist_acc, 0.0)
            ordinary = dist_acc + deemed + other_ord_acc
            if cfg.tax_brackets is not None:
                tax = annual_tax(
                    0.0, 0.0, cfg.tax_brackets, cum_inflation[:, m + 1],
                    state_rate=cfg.state_tax, other_ordinary=ordinary,
                )
            else:
                tax = (ord_rate + cfg.state_tax) * ordinary
            tax = np.minimum(tax, total)
            total_tax_real += tax / cum_inflation[:, m + 1]
            with np.errstate(invalid="ignore", divide="ignore"):
                scale_t = np.where(total > 0, (total - tax) / total, 1.0)
            holdings *= scale_t[:, None]
            dist_acc = tax.copy()  # paying the tax is itself a distribution
            other_ord_acc = np.zeros(n_paths)

        balance[:, m + 1] = holdings.sum(axis=1)

    return SimResult(
        config=cfg,
        months=months,
        balance=balance,
        cum_inflation=cum_inflation,
        depleted_month=depleted_month,
        window=window,
        total_withdrawn=total_withdrawn,
        total_withdrawn_real=total_withdrawn_real,
        total_tax_real=total_tax_real,
        early_real_market=early_real_market,
    )


def summarize(result: SimResult, real: bool = True) -> dict:
    """Headline numbers for one simulation run."""
    bal = result.real_balance if real else result.balance
    terminal = bal[:, -1]
    pct = np.percentile(terminal, [5, 25, 50, 75, 95])
    years = result.config.years
    initial = result.config.initial
    with np.errstate(divide="ignore"):
        cagr = np.where(terminal > 0, (terminal / initial) ** (1 / years) - 1, np.nan)
    return {
        "n_paths": result.n_paths,
        "success_rate": result.success_rate,
        "terminal_p5": pct[0],
        "terminal_p25": pct[1],
        "terminal_median": pct[2],
        "terminal_p75": pct[3],
        "terminal_p95": pct[4],
        "median_cagr": float(np.nanmedian(cagr)),
        "prob_loss": float((terminal < initial).mean()),
        "sample_window": f"{result.window.min()}..{result.window.max()}",
    }
