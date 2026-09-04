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
    # Smooth real spending decline (fixed_real only): the target shrinks by
    # `decline` per year (e.g. 0.01 = 1%/yr, the observed "retirement smile"
    # downslope), compounding monthly from decline_start_month. Composes with
    # amount/rate/schedule and with flex_floor.
    decline: float = 0.0
    decline_start_month: int = 0


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


@dataclass(frozen=True)
class Account:
    """One account in a multi-account household (SimConfig.accounts).

    kind: 'taxable' | 'traditional' | 'roth' | '529' - same tax semantics as
    SimConfig.account. allocation: this account's target weights (None = the
    config-level allocation), enabling asset location (munis in taxable,
    Treasuries in Roth). cost_basis: starting basis fraction, taxable only.
    """

    kind: str
    balance: float
    allocation: dict[str, float] | None = None
    # taxable: starting cost basis as a fraction of balance. roth: the
    # contribution basis - draws up to it are penalty-free before 59.5,
    # draws beyond it are earnings (default 1.0 = all contributions).
    # 529 (multi-account mode): the contribution fraction for the pro-rata
    # earnings/basis split on non-qualified draws.
    cost_basis: float = 1.0
    # This account's own draw schedule: (start_month, annual real dollars)
    # steps, drawn from this account each month ON TOP of the household
    # withdrawal policy (which is never flexed/declined into it). A 529's
    # scheduled draws are QUALIFIED (tuition): tax- and penalty-free.
    # Scheduled traditional/roth draws are taxed/penalized like any other
    # draw from that account. If the account can't cover its schedule, the
    # shortfall falls back to the household waterfall.
    schedule: tuple[tuple[int, float], ...] | None = None


@dataclass
class SimConfig:
    allocation: dict[str, float] | None = None  # asset -> weight, must sum to 1
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
    # tailwind from US equities. {'us_equities': -0.005} = -50bp/yr. A value
    # may also be a per-month sequence (length years*12) of annual rates, for
    # time-varying paths like a P/E cycle.
    return_adjustments: "dict[str, float] | None" = None
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
    # Multi-account household: when set, `account`, `initial`, and
    # `cost_basis_start` are ignored and each Account carries its own kind,
    # balance, allocation, and basis. Withdrawals waterfall through the
    # accounts in `withdraw_order` (spend taxable first by default);
    # contributions and surplus income land in the taxable account. Taxes are
    # settled jointly (taxable investment income and traditional distributions
    # stack through one set of brackets) and paid from the taxable account
    # when there is one. Traditional RMD dollars beyond spending are actually
    # transferred to the taxable account (basis = value). At most one taxable
    # and one traditional account; glidepaths/allocation rules and the
    # cfg-level ladder are single-account features.
    accounts: tuple[Account, ...] | None = None
    withdraw_order: tuple[str, ...] | None = None  # default taxable->traditional->roth->529
    # TIPS-ladder-as-allocation: the reserved asset name 'tips_ladder' in any
    # allocation buys rungs with that share of the account's balance at t=0
    # (a purchase-time cost share, NOT a maintained weight - rungs amortize
    # and are never rebalanced). Rung income offsets household withdrawals;
    # the remaining principal is carried in reported balances at amortized
    # (par) value but is not drawable by the waterfall. Taxation follows the
    # holding account: taxable = phantom income; traditional = payouts are
    # distributions (RMD-countable, early-penalty applies); roth = tax-free.
    # Priced at ladder_yield (flat, decimal) or ladder_curve ({maturity_years:
    # yield} points, interpolated); term ladder_years (default: the horizon).
    ladder_yield: float = 0.02
    ladder_curve: dict[int, float] | None = None
    ladder_years: int | None = None
    # Valuation-conditioned sampling (bootstrap mode only): block starts are
    # drawn with Gaussian-kernel weights in log-state space, matching each
    # block's historical state (state_series, e.g. Shiller P/E by month) to
    # the assumed state at that point of the simulation (state_path, one
    # level per simulation month). Blocks then carry the DYNAMICS - vol,
    # correlations, inflation regime - of comparable valuation eras, and
    # regimes as long as the path persist beyond the block length. For
    # assets in state_adjust_assets, returns are re-centered so the path's
    # log-state drift replaces the sampled months' conditional drift.
    state_series: "object | None" = None  # pd.Series, monthly PeriodIndex
    state_path: "object | None" = None  # (years*12,) assumed state levels
    state_bandwidth: float = 0.15  # kernel width in log-state units
    state_adjust_assets: tuple[str, ...] | None = None
    age: int | None = None  # age at t=0; enables age-based features (RMDs)
    # 10% early-withdrawal penalty before age 59.5 (needs `age`): applies to
    # traditional draws and to the earnings portion of roth draws (beyond the
    # account's cost_basis x balance of contributions); 529s are assumed
    # qualified. Settled annually, paid from the taxable account when there
    # is one (the payment is not itself a distribution). Ordinary income tax
    # on early roth earnings is NOT modeled - only the penalty.
    early_penalty: bool = True
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
    # Multi-account runs: kinds in config order, and nominal terminal balance
    # per account (n_paths, n_accounts). None for single-account runs.
    account_kinds: tuple[str, ...] | None = None
    account_terminal: np.ndarray | None = None
    # Total real income/yr of allocation-based TIPS ladders, None if none.
    ladder_annual: float | None = None
    # (n_paths,) real dollars of spending the household could not deliver
    # (liquid exhausted while a target remained). Ladder runs only.
    total_unmet_real: np.ndarray | None = None

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


def _slot_weights(
    state_log: np.ndarray, path_log: np.ndarray, block: int, n_blocks: int, bw: float
) -> np.ndarray:
    """(n_blocks, t_hist) block-start weights: Gaussian kernel in log-state."""
    W = np.empty((n_blocks, len(state_log)))
    for b in range(n_blocks):
        target = path_log[min(b * block, len(path_log) - 1)]
        w = np.exp(-0.5 * ((state_log - target) / bw) ** 2)
        w[~np.isfinite(w)] = 0.0
        s = w.sum()
        if s <= 0:
            raise ValueError(
                f"no historical months have state data near the assumed level "
                f"{np.exp(target):.1f} (block {b})"
            )
        W[b] = w / s
    return W


def _sample_months(
    cfg: SimConfig,
    t_hist: int,
    rng: np.random.Generator,
    slot_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Month indices per path: circular block bootstrap (optionally with
    state-conditioned block starts), or all historical windows."""
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
    if slot_weights is None:
        starts = rng.integers(0, t_hist, size=(cfg.n_sims, n_blocks))
    else:
        starts = np.empty((cfg.n_sims, n_blocks), dtype=np.int64)
        for b in range(n_blocks):
            starts[:, b] = rng.choice(t_hist, size=cfg.n_sims, p=slot_weights[b])
    offsets = np.arange(block)
    idx = (starts[:, :, None] + offsets[None, None, :]) % t_hist  # circular
    return idx.reshape(cfg.n_sims, -1)[:, :n_months]


ACCOUNT_KINDS = ("taxable", "traditional", "roth", "529")
DEFAULT_WITHDRAW_ORDER = ("taxable", "traditional", "roth", "529")
LADDER_ASSET = "tips_ladder"  # reserved allocation name: buys held-to-maturity rungs


def total_initial(cfg: SimConfig) -> float:
    """Household starting balance: sum of the accounts, or `initial`."""
    if cfg.accounts:
        return float(sum(a.balance for a in cfg.accounts))
    return cfg.initial


def _schedule_array(
    schedule, n_months: int, what: str = "withdrawal schedule"
) -> np.ndarray:
    """(start_month, annual real dollars) steps -> per-month real amounts."""
    starts = np.array([s for s, _ in schedule], dtype=int)
    amts = np.array([a for _, a in schedule], dtype=float) / 12.0
    if starts[0] < 0 or (np.diff(starts) <= 0).any():
        raise ValueError(f"{what} months must be increasing and >= 0")
    seg = np.searchsorted(starts, np.arange(n_months), side="right") - 1
    return np.where(seg >= 0, amts[np.maximum(seg, 0)], 0.0)


def _weight_vec(alloc: dict[str, float], assets: list[str], label: str) -> np.ndarray:
    w = np.array([alloc.get(a, 0.0) for a in assets], dtype=float)
    if abs(w.sum() - 1.0) > 1e-6:
        raise ValueError(f"{label} weights sum to {w.sum():.4f}, expected 1")
    return w


class _Acct:
    """Per-account simulation state."""

    __slots__ = ("kind", "weights", "holdings", "basis", "income_credit", "taxed")

    def __init__(self, kind, weights, holdings, basis, taxed):
        self.kind = kind
        self.weights = weights
        self.holdings = holdings
        self.basis = basis
        self.taxed = taxed  # taxable basis/income machinery active
        self.income_credit = None


def simulate(panel: pd.DataFrame, cfg: SimConfig) -> SimResult:
    multi = cfg.accounts is not None
    if multi:
        specs = tuple(cfg.accounts)
        if not specs:
            raise ValueError("accounts must be non-empty")
        kinds = [a.kind for a in specs]
        for a in specs:
            if a.kind not in ACCOUNT_KINDS:
                raise ValueError(
                    f"account kind must be one of {ACCOUNT_KINDS}, got {a.kind!r}"
                )
            if a.balance <= 0:
                raise ValueError(f"account balance must be positive, got {a.balance}")
        if kinds.count("taxable") > 1 or kinds.count("traditional") > 1:
            raise ValueError("at most one taxable and one traditional account")
        if cfg.allocation_end is not None or cfg.allocation_rule is not None:
            raise ValueError("glidepaths and allocation rules need single-account mode")
        order = cfg.withdraw_order or DEFAULT_WITHDRAW_ORDER
        bad = [k for k in order if k not in ACCOUNT_KINDS]
        if bad:
            raise ValueError(f"unknown kind(s) in withdraw_order: {bad}")
        rank = {k: i for i, k in enumerate(order)}
        unranked = [k for k in kinds if k not in rank]
        if unranked:
            raise ValueError(f"withdraw_order must cover account kind(s) {unranked}")
        draw_order = sorted(range(len(specs)), key=lambda i: rank[kinds[i]])
        assets: list[str] = []
        for src in [cfg.allocation or {}] + [a.allocation or {} for a in specs]:
            for k in src:
                if k not in assets and k != LADDER_ASSET:
                    assets.append(k)
        for a in specs:
            if a.allocation is None and not cfg.allocation:
                raise ValueError(
                    "an account without its own allocation needs a "
                    "config-level allocation"
                )
    else:
        if not cfg.allocation:
            raise ValueError("allocation is required")
        if cfg.account not in ACCOUNT_KINDS:
            raise ValueError(
                f"account must be taxable/traditional/roth/529, got {cfg.account!r}"
            )
        specs = (
            Account(kind=cfg.account, balance=cfg.initial,
                    allocation=cfg.allocation, cost_basis=cfg.cost_basis_start),
        )
        kinds = [cfg.account]
        draw_order = [0]
        assets = [a for a in cfg.allocation if a != LADDER_ASSET]
    missing = [a for a in assets if a not in panel.columns]
    if missing:
        raise ValueError(f"unknown asset(s): {missing}; available: {list(panel.columns)}")
    # Per-account weights: raw asset weights (used at purchase - the ladder
    # share buys rungs) and liquid-renormalized weights (rebalance targets;
    # rungs are never rebalanced).
    lad_w: list[float] = []
    acct_w_raw: list[np.ndarray] = []
    for s in specs:
        alloc = s.allocation or cfg.allocation
        wl = float(alloc.get(LADDER_ASSET, 0.0))
        if not 0 <= wl <= 1:
            raise ValueError(f"tips_ladder weight must be in [0, 1], got {wl}")
        rest = {k: v for k, v in alloc.items() if k != LADDER_ASSET}
        wv = np.array([rest.get(a, 0.0) for a in assets], dtype=float)
        if abs(wv.sum() + wl - 1.0) > 1e-6:
            raise ValueError(
                f"allocation weights sum to {wv.sum() + wl:.4f}, expected 1"
            )
        lad_w.append(wl)
        acct_w_raw.append(wv)
    has_ladder_alloc = any(wl > 0 for wl in lad_w)
    # Liquid rebalance targets: renormalize by the actual asset-weight sum
    # (dividing by 1-wl would amplify the 1e-6 sum tolerance into a per-
    # rebalance value leak).
    acct_w = []
    for wv, wl in zip(acct_w_raw, lad_w):
        if wl <= 0:
            acct_w.append(wv / (1.0 - wl))
        else:
            s = wv.sum()
            acct_w.append(wv / s if s > 0 else wv)
    if has_ladder_alloc:
        if not assets:
            raise ValueError("at least one non-ladder asset is required")
        if cfg.ladder is not None:
            raise ValueError(
                "use either the external ladder (--tips-ladder) or a "
                "tips_ladder allocation, not both"
            )
        if cfg.allocation_end is not None or cfg.allocation_rule is not None:
            raise ValueError(
                "glidepaths/allocation rules don't support tips_ladder allocations"
            )
        for wl, k in zip(lad_w, kinds):
            if wl > 0 and k == "529":
                raise ValueError("tips_ladder is not supported in 529 accounts")
    initial_total = total_initial(cfg)
    if multi:
        if has_ladder_alloc:
            w_house = sum(s.balance * w for s, w in zip(specs, acct_w_raw))
            hs = w_house.sum()
            weights = w_house / hs if hs > 0 else w_house
        else:
            weights = sum(s.balance * w for s, w in zip(specs, acct_w)) / initial_total
    else:
        weights = acct_w[0]

    # Build the per-account rung ladders bought at t=0.
    acct_ladders: dict[int, object] = {}
    acct_lad_val: dict[int, np.ndarray] = {}
    lyears = 0
    ladder_annual_total = 0.0
    if has_ladder_alloc:
        from .ladder import build_ladder, build_ladder_curve

        if cfg.ladder_years is not None and cfg.ladder_years < 1:
            raise ValueError(f"ladder_years must be >= 1, got {cfg.ladder_years}")
        lyears = cfg.ladder_years or cfg.years
        for i, (s, wl) in enumerate(zip(specs, lad_w)):
            if wl <= 0:
                continue
            cost = wl * s.balance
            tax_flag = kinds[i] == "taxable"
            if cfg.ladder_curve:
                unit = build_ladder_curve(1.0, lyears, cfg.ladder_curve, taxable=tax_flag)
                spec = build_ladder_curve(
                    cost / unit.cost, lyears, cfg.ladder_curve, taxable=tax_flag
                )
            else:
                unit = build_ladder(1.0, lyears, cfg.ladder_yield, taxable=tax_flag)
                spec = build_ladder(
                    cost / unit.cost, lyears, cfg.ladder_yield, taxable=tax_flag
                )
            acct_ladders[i] = spec
            ladder_annual_total += spec.annual

    # Per-month target weights: static, or a linear glide over glide_years
    # (single-account mode; each account in accounts mode holds its own
    # static weights).
    n_months_total = cfg.years * 12
    if not multi and cfg.allocation_end is not None:
        if set(cfg.allocation_end) != set(assets):
            raise ValueError("allocation_end must use the same assets as allocation")
        w_end = np.array([cfg.allocation_end[a] for a in assets], dtype=float)
        if abs(w_end.sum() - 1.0) > 1e-6:
            raise ValueError(f"allocation_end weights sum to {w_end.sum():.4f}, expected 1")
        if cfg.glide_years is not None and cfg.glide_years < 1:
            raise ValueError(f"glide_years must be >= 1, got {cfg.glide_years}")
        glide_m = min((cfg.glide_years or cfg.years) * 12, n_months_total)
        frac = np.minimum(np.arange(n_months_total) / max(glide_m - 1, 1), 1.0)
        target_w = weights[None, :] + frac[:, None] * (w_end - weights)[None, :]
    else:
        target_w = np.tile(weights, (n_months_total, 1))

    tax_i = kinds.index("taxable") if "taxable" in kinds else None
    trad_i = kinds.index("traditional") if "traditional" in kinds else None
    trad = trad_i is not None
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
    has_taxable_income = any(s.taxable for s in cfg.income or ())
    # A household of only tax-free accounts still owes ordinary tax on
    # taxable outside income (pensions), so a tax regime is allowed then.
    pension_only = tax_i is None and not trad and any_tax_setting and has_taxable_income
    if tax_i is None and not trad and any_tax_setting and not has_taxable_income:
        label = cfg.account if not multi else "roth/529-only households"
        raise ValueError(f"{label} accounts are tax-free; clear the tax settings")
    if trad:
        if cfg.age is None:
            raise ValueError("traditional accounts need `age` (for RMDs)")
        if cfg.tax_brackets is None and ord_rate <= 0:
            raise ValueError(
                "traditional accounts need tax_brackets or a flat ordinary rate "
                "for distribution taxation"
            )
    taxed = tax_i is not None and any_tax_setting
    returns_hist, inflation_hist, window, income_hist = _historical_matrix(
        panel, assets, cfg, need_income=taxed
    )
    rng = np.random.default_rng(cfg.seed)
    W = None
    if cfg.state_series is not None:
        if cfg.mode != "bootstrap":
            raise ValueError("state-conditioned sampling needs bootstrap mode")
        if cfg.state_path is None:
            raise ValueError("state_series needs a state_path (assumed levels)")
        path_lvl = np.asarray(cfg.state_path, dtype=float)
        if len(path_lvl) != cfg.years * 12:
            raise ValueError(
                f"state_path has length {len(path_lvl)}, expected {cfg.years * 12}"
            )
        if not cfg.state_bandwidth > 0:
            raise ValueError("state_bandwidth must be positive")
        with np.errstate(invalid="ignore", divide="ignore"):
            state_log = np.log(cfg.state_series.reindex(window).to_numpy(dtype=float))
            path_log = np.log(path_lvl)
        block = max(1, min(cfg.block_months, len(window)))
        n_blocks = -(-(cfg.years * 12) // block)
        W = _slot_weights(state_log, path_log, block, n_blocks, cfg.state_bandwidth)
    months = _sample_months(cfg, len(window), rng, slot_weights=W)
    n_paths, n_months = months.shape

    # State-conditioned drift re-centering: replace each block's conditional
    # historical log-state drift with the assumed path's, so the path's
    # multiple expansion isn't double-counted on top of what the sampled
    # regimes already did.
    eff_adj: dict = dict(cfg.return_adjustments or {})
    if W is not None and cfg.state_adjust_assets:
        drift_hist = np.zeros(len(window))
        d = (state_log[1:] - state_log[:-1]) * 12.0
        drift_hist[:-1] = np.nan_to_num(d)
        cond = W @ drift_hist  # (n_blocks,) conditional drift per slot
        path_drift = np.zeros(n_months)
        path_drift[:-1] = (path_log[1:] - path_log[:-1]) * 12.0
        adj_state = path_drift - cond[np.arange(n_months) // block]
        for a in cfg.state_adjust_assets:
            if a in assets:
                eff_adj[a] = eff_adj.get(a, 0.0) + adj_state

    if not 0 <= cfg.fee_annual < 0.1:
        raise ValueError(f"fee_annual must be in [0, 0.1), got {cfg.fee_annual}")
    path_returns = returns_hist[months]  # (n_paths, n_months, n_assets)
    if eff_adj or cfg.fee_annual:
        vals = [eff_adj.get(a, 0.0) for a in assets]
        if any(np.ndim(v) > 0 for v in vals):
            # Per-month adjustment paths (each scalar or length-n_months).
            adj = np.zeros((n_months, len(assets)))
            for j, v in enumerate(vals):
                v = np.asarray(v, dtype=float)
                if v.ndim > 0 and len(v) != n_months:
                    raise ValueError(
                        f"per-month return adjustment for {assets[j]!r} has "
                        f"length {len(v)}, expected {n_months}"
                    )
                adj[:, j] = v
            path_returns = path_returns + (adj - cfg.fee_annual)[None, :, :] / 12.0
        else:
            adj = np.array(vals)
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
    if w.decline and w.kind != "fixed_real":
        raise ValueError("spending decline only applies to fixed_real withdrawals")
    if not 0 <= w.decline < 1:
        raise ValueError(f"decline must be in [0, 1), got {w.decline}")
    if w.kind == "fixed_real":
        # Per-month real spending target: constant, or a step schedule.
        if w.schedule:
            spend_real_m = _schedule_array(w.schedule, n_months)
        else:
            spend_real_m = np.full(n_months, (w.amount or w.rate * initial_total) / 12.0)
        if w.decline:
            past = np.maximum(np.arange(n_months) - w.decline_start_month, 0) / 12.0
            spend_real_m = spend_real_m * (1.0 - w.decline) ** past
    elif w.kind == "percent_of_balance":
        monthly_withdrawal = np.full(n_paths, initial_total * w.rate / 12.0)
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
    # Rung income offsets household withdrawals like any income stream
    # (smoothed monthly); the remaining principal is carried in reported
    # balances at par (held to maturity) but is not drawable.
    if acct_ladders:
        lm = min(lyears * 12, n_months)
        for lad in acct_ladders.values():
            income_real_m[:lm] += lad.annual / 12.0
        lad_val_real = np.zeros(n_months + 1)
        for i, lad in acct_ladders.items():
            prin = np.asarray(lad.remaining_principal_real(), dtype=float)
            v = np.zeros(n_months + 1)
            for k in range(n_months + 1):
                if k // 12 < lyears:
                    v[k] = prin[k // 12]
            acct_lad_val[i] = v
            lad_val_real += v
    has_flows = bool(income_real_m.any() or expense_real_m.any())
    if w.flex_floor is not None:
        if w.kind != "fixed_real":
            raise ValueError("flex_floor only applies to fixed_real withdrawals")
        if not 0 < w.flex_floor <= 1:
            raise ValueError(f"flex_floor must be in (0, 1], got {w.flex_floor}")

    if not (0 <= cfg.tax_rate < 1 and 0 <= ord_rate < 1):
        raise ValueError(f"tax rates must be in [0, 1), got {cfg.tax_rate}/{ord_rate}")
    accts: list[_Acct] = []
    for i, s in enumerate(specs):
        # Raw weights: the ladder share of the balance bought rungs at t=0,
        # leaving balance x asset-weight dollars in each liquid asset.
        h = np.tile(acct_w_raw[i] * s.balance, (n_paths, 1))  # (n_paths, n_assets)
        b = h * s.cost_basis  # average-cost basis per asset (taxable only)
        accts.append(_Acct(kinds[i], acct_w[i], h, b,
                           taxed and kinds[i] == "taxable"))
    inflow_i = tax_i if tax_i is not None else draw_order[0]
    balance = np.empty((n_paths, n_months + 1))
    balance[:, 0] = initial_total
    depleted_month = np.full(n_paths, -1, dtype=int)
    total_withdrawn = np.zeros(n_paths)
    total_withdrawn_real = np.zeros(n_paths)
    total_unmet_real = np.zeros(n_paths)  # spending the household couldn't deliver

    realized = np.zeros(n_paths)  # gains realized since last tax settlement
    loss_carry = np.zeros(n_paths)  # <= 0, carried-forward losses
    div_acc = np.zeros(n_paths)  # dividends received since last settlement
    int_acc = np.zeros(n_paths)  # interest received since last settlement
    total_tax_real = np.zeros(n_paths)
    # Traditional-account state: nominal distributions this tax year (spending
    # withdrawals, later the tax payment itself), and the balance at the start
    # of the year that RMDs are computed from.
    dist_acc = np.zeros(n_paths)
    other_ord_acc = np.zeros(n_paths)  # taxable outside income since last settlement
    year_start_bal = np.full(n_paths, initial_total)
    # 10% penalty on early (pre-59.5) retirement-account draws: traditional
    # draws, and roth draws beyond the contribution basis.
    pen_cut = 0
    if cfg.early_penalty and cfg.age is not None:
        pen_cut = min(max(int(round((59.5 - cfg.age) * 12)), 0), n_months)
    roth_idx = [i for i, k in enumerate(kinds) if k == "roth"]
    pen_active = pen_cut > 0 and (trad or roth_idx)
    penalty_acc = np.zeros(n_paths)
    roth_basis = {
        i: np.full(n_paths, specs[i].balance * specs[i].cost_basis) for i in roth_idx
    }
    # 529s in a household waterfall: any draw funds the household's spending,
    # not education, so it is NON-QUALIFIED - each distribution splits
    # pro-rata between contributions (cost_basis) and earnings at the ratio
    # on the draw date; the earnings are ordinary income (taxed wherever a
    # tax regime exists) plus the 10% penalty, at any age. Standalone 529
    # runs model tuition draws and stay qualified (tax-free).
    q529_idx = [i for i, k in enumerate(kinds) if k == "529"] if multi else []
    q529_basis = {
        i: np.full(n_paths, specs[i].balance * specs[i].cost_basis) for i in q529_idx
    }
    # Per-account draw schedules (multi-account mode): drawn from the owning
    # account before the household waterfall; shortfalls fall back to it.
    sched_idx = [i for i, s in enumerate(specs) if s.schedule] if multi else []
    has_acct_sched = bool(sched_idx)
    acct_sched_real = {
        i: _schedule_array(specs[i].schedule, n_months, f"account #{i + 1} schedule")
        for i in sched_idx
    }
    if taxed:
        from .data import INCOME_CLASS

        classes = [INCOME_CLASS.get(a, "dividend") for a in assets]
        interest_mask = np.array([1.0 if c == "interest" else 0.0 for c in classes])
        dividend_mask = np.array([1.0 if c == "dividend" else 0.0 for c in classes])
        # anything else (muni) is tax-exempt income: spendable, never taxed
        path_income = income_hist[months]  # (n_paths, n_months, n_assets)
        # Withdrawals spend the previous month's dividends/interest as cash
        # first (those dollars have basis = value, so spending them realizes
        # no gains); only the shortfall is a gain-realizing sale. Unspent
        # income stays reinvested (its basis step-up already applied).
        accts[tax_i].income_credit = np.zeros((n_paths, len(assets)))
        # A taxable TIPS ladder throws off federal ordinary income each year:
        # its coupons plus the inflation accrual on remaining principal
        # ("phantom income"), both state-exempt Treasury interest. The tax is
        # paid from the portfolio.
        ladder_coupons = ladder_principal = None
        # Phantom income applies to the external ladder or an allocation-based
        # ladder held in the taxable account.
        lad = cfg.ladder if cfg.ladder is not None else acct_ladders.get(tax_i)
        if lad is not None and getattr(lad, "taxable", False) and getattr(lad, "faces", ()):
            ladder_coupons = lad.coupon_income_real()
            ladder_principal = lad.remaining_principal_real()

    def prorata_flow(acct: _Acct, scale: np.ndarray) -> None:
        """Apply a pro-rata sale (scale<1) or buy (scale>1) to one account's
        holdings+basis, booking realized gains on the sale portion."""
        nonlocal realized
        if acct.taxed:
            selling = scale < 1
            realized += np.where(
                selling, (1 - scale) * (acct.holdings.sum(1) - acct.basis.sum(1)), 0.0
            )
            buy_add = np.where(scale > 1, scale - 1, 0.0)[:, None] * acct.holdings
            acct.basis[:] = np.where(
                selling[:, None], acct.basis * scale[:, None], acct.basis + buy_add
            )
        acct.holdings[:] *= scale[:, None]

    def rmd_deemed(m: int) -> "np.ndarray | float":
        """RMD dollars beyond this year's distributions. With a taxable
        account present the shortfall is actually transferred there (sold
        pro-rata from the IRA, bought pro-rata into the taxable account with
        basis = value - post-tax dollars whose tax is levied at settlement);
        otherwise it is deemed distributed and stays invested."""
        trad_acct = accts[trad_i]
        age_now = cfg.age + (m + 1) // 12 - 1  # age attained this sim year
        from .tax import RMD_START_AGE, rmd_period

        if age_now < RMD_START_AGE:
            return 0.0
        rmd = year_start_bal / rmd_period(age_now)
        trad_tot = trad_acct.holdings.sum(axis=1)
        # Recognition is capped at the IRA's full value (rungs included -
        # they can be distributed in kind); only the liquid portion can
        # actually be transferred to the taxable account.
        ira_val = trad_tot
        if trad_i in acct_ladders:
            ira_val = ira_val + acct_lad_val[trad_i][m + 1] * cum_inflation[:, m + 1]
        deemed = np.minimum(np.maximum(rmd - dist_acc, 0.0), ira_val)
        if taxed:
            move = np.minimum(deemed, trad_tot)
            with np.errstate(invalid="ignore", divide="ignore"):
                scale_r = np.where(trad_tot > 0, (trad_tot - move) / trad_tot, 1.0)
            trad_acct.holdings *= scale_r[:, None]
            tx = accts[tax_i]
            tx_tot = tx.holdings.sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                shares = np.where(
                    tx_tot[:, None] > 0,
                    tx.holdings / np.where(tx_tot > 0, tx_tot, 1.0)[:, None],
                    tx.weights[None, :],
                )
            buy = shares * move[:, None]
            tx.holdings += buy
            tx.basis += buy
        return deemed

    # With rung income, depletion means unmet spending, not zero liquid
    # (percent-of-balance excepted: its target floats with wealth).
    unmet_mode = bool(acct_ladders) and w.kind != "percent_of_balance"

    def collect(due, first):
        """Pay a tax bill from `first`, then the remaining accounts in draw
        order — the bill is owed by the household, not one account. Returns
        (paid, paid_from_traditional); the traditional portion is itself a
        distribution. Any remainder means the household is broke."""
        order = [first] + [i for i in draw_order if i != first]
        paid = np.zeros(n_paths)
        from_trad_pay = np.zeros(n_paths)
        for i in order:
            tot = accts[i].holdings.sum(axis=1)
            pay = np.minimum(due, tot)
            with np.errstate(invalid="ignore", divide="ignore"):
                scale = np.where(tot > 0, (tot - pay) / tot, 1.0)
            prorata_flow(accts[i], scale)
            due = due - pay
            paid = paid + pay
            if trad and i == trad_i:
                from_trad_pay = pay
        return paid, from_trad_pay

    for m in range(n_months):
        totals = [a.holdings.sum(axis=1) for a in accts]
        total = totals[0]
        for t in totals[1:]:
            total = total + t
        alive = total > 0
        # Wealth for policy purposes (flex, percent-of-balance) includes the
        # ladder's remaining principal; draw capacity does not (rungs are
        # held to maturity, not sellable).
        total_rep = (
            total + lad_val_real[m] * cum_inflation[:, m] if acct_ladders else total
        )
        if trad and m % 12 == 0:
            year_start_bal = totals[trad_i].copy()
            if trad_i in acct_ladders:
                # RMDs are owed on the IRA's full value, rungs included
                # (their payouts count toward satisfying it via dist_acc).
                year_start_bal = (
                    year_start_bal + acct_lad_val[trad_i][m] * cum_inflation[:, m]
                )

        # Withdrawal / contribution at the start of the month, pro-rata across
        # holdings so the flow itself doesn't rebalance the portfolio.
        if w.kind == "percent_of_balance" and m > 0 and m % 12 == 0:
            monthly_withdrawal = total_rep * w.rate / 12.0
        if w.kind == "fixed_real":
            spend = spend_real_m[m] * cum_inflation[:, m]
            if w.flex_floor is not None:
                real_balance = total_rep / cum_inflation[:, m]
                spend *= np.clip(real_balance / initial_total, w.flex_floor, 1.0)
        elif w.kind == "percent_of_balance":
            spend = monthly_withdrawal.copy()
        else:
            spend = np.zeros(n_paths)
        # Account draw schedules: taken from the owning account itself; any
        # shortfall joins the household spending need below.
        pre_total = 0.0
        pre_draws: dict = {}
        if has_acct_sched:
            short = np.zeros(n_paths)
            for i in sched_idx:
                need = acct_sched_real[i][m] * cum_inflation[:, m]
                take = np.minimum(need, totals[i])
                pre_draws[i] = take
                short += need - take
                pre_total = pre_total + take
            spend = spend + short
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
        cap = total if not has_acct_sched else total - pre_total
        wd = np.where(alive, np.minimum(spend, cap), 0.0)
        if acct_ladders:
            unmet = spend - wd
            total_unmet_real += unmet / cum_inflation[:, m]
        total_withdrawn += wd
        total_withdrawn_real += wd / cum_inflation[:, m]
        if has_acct_sched:
            total_withdrawn += pre_total
            total_withdrawn_real += pre_total / cum_inflation[:, m]
        if taxed or trad or pension_only:
            other_ord_acc += income_taxed_real_m[m] * cum_inflation[:, m]

        # Waterfall: draw from accounts in withdraw_order until the need is met.
        draws: list = [None] * len(accts)
        remaining = wd
        for i in draw_order:
            avail = totals[i] if not has_acct_sched else totals[i] - pre_draws.get(i, 0.0)
            take = np.minimum(remaining, avail)
            draws[i] = take
            remaining = remaining - take
        if has_acct_sched:
            wf_529 = {qi: draws[qi] for qi in q529_idx}  # non-qualified portion
            for i in sched_idx:
                draws[i] = draws[i] + pre_draws[i]
        if trad:
            dist_acc += draws[trad_i]
        if pen_active and m < pen_cut:
            if trad:
                penalty_acc += 0.10 * draws[trad_i]
            for ri in roth_idx:
                from_basis = np.minimum(draws[ri], roth_basis[ri])
                roth_basis[ri] = roth_basis[ri] - from_basis
                penalty_acc += 0.10 * (draws[ri] - from_basis)
        for qi in q529_idx:
            with np.errstate(invalid="ignore", divide="ignore"):
                basis_frac = np.where(
                    totals[qi] > 0,
                    np.minimum(q529_basis[qi] / np.where(totals[qi] > 0, totals[qi], 1.0), 1.0),
                    1.0,
                )
            from_basis = draws[qi] * basis_frac
            q529_basis[qi] = np.maximum(q529_basis[qi] - from_basis, 0.0)
            # Scheduled (tuition) draws are qualified; only the waterfall
            # portion is non-qualified and taxed/penalized on its earnings.
            nq = wf_529[qi] if has_acct_sched else draws[qi]
            earnings = nq - nq * basis_frac
            penalty_acc += 0.10 * earnings
            if taxed or trad:
                other_ord_acc += earnings
        # Rung payouts from retirement-held ladders: an IRA's payouts are
        # distributions (ordinary income, RMD-countable, early-penalized);
        # a roth's consume contribution basis first for the early penalty.
        if acct_ladders and m < lyears * 12:
            if trad and trad_i in acct_ladders:
                pay_t = acct_ladders[trad_i].annual / 12.0 * cum_inflation[:, m]
                dist_acc += pay_t
                if pen_active and m < pen_cut:
                    penalty_acc += 0.10 * pay_t
            if pen_active and m < pen_cut:
                for ri in roth_idx:
                    if ri in acct_ladders:
                        pay_r = acct_ladders[ri].annual / 12.0 * cum_inflation[:, m]
                        from_basis = np.minimum(pay_r, roth_basis[ri])
                        roth_basis[ri] = roth_basis[ri] - from_basis
                        penalty_acc += 0.10 * (pay_r - from_basis)

        wd_to_sell = draws
        if taxed:
            # Fund the taxable draw from last month's income first: no sale,
            # no realized gain; remove exactly the basis those dollars carry.
            acct = accts[tax_i]
            spendable = np.minimum(
                acct.income_credit, np.minimum(acct.holdings, acct.basis)
            )
            credit_total = spendable.sum(axis=1)
            use = np.minimum(draws[tax_i], credit_total)
            with np.errstate(invalid="ignore", divide="ignore"):
                frac = np.where(credit_total > 0, use / credit_total, 0.0)
            sold = spendable * frac[:, None]
            acct.holdings -= sold
            acct.basis -= sold
            acct.income_credit[:] = 0.0  # unspent income reverts to reinvested
            wd_to_sell = list(draws)
            wd_to_sell[tax_i] = draws[tax_i] - use
            totals[tax_i] = acct.holdings.sum(axis=1)

        # Contributions and surplus income land in the taxable account.
        inflow = cfg.contribution_monthly * cum_inflation[:, m] + income_invested
        for i, acct in enumerate(accts):
            flow = inflow - wd_to_sell[i] if i == inflow_i else -wd_to_sell[i]
            with np.errstate(invalid="ignore", divide="ignore"):
                scale = np.where(totals[i] > 0, (totals[i] + flow) / totals[i], 0.0)
            prorata_flow(acct, scale)
            # A pro-rata scale can't add money to an empty account: seed
            # inflows into a drained (or depleted) account at target weights
            # instead of silently dropping them.
            if i == inflow_i and (cfg.contribution_monthly > 0 or has_flows):
                empty_in = (totals[i] <= 0) & (np.asarray(flow) > 0)
                if empty_in.any():
                    # An all-ladder account has no liquid weights; fall back
                    # to household weights, then equal weights.
                    wvec = acct.weights
                    if wvec.sum() <= 0:
                        wvec = weights if weights.sum() > 0 else np.full(
                            len(assets), 1.0 / len(assets)
                        )
                    add = np.where(empty_in, flow, 0.0)[:, None] * wvec[None, :]
                    acct.holdings += add
                    if acct.taxed:
                        acct.basis += add
        # A path is depleted the first time the household hits zero - whether
        # a withdrawal drained it here or a later-in-the-month settlement
        # (taxes, penalties) zeroed it before this check. With rung income
        # still arriving, a zero liquid balance isn't failure; depletion is
        # the first month spending goes unmet (percent-of-balance excepted:
        # its target floats, so liquid exhaustion still marks depletion).
        unmet_flags = unmet_mode
        if multi:
            for acct in accts:
                emptied = acct.holdings.sum(axis=1) <= 1e-9
                acct.holdings[emptied] = 0.0
                acct.basis[emptied] = 0.0
            if unmet_flags:
                newly_dead = (depleted_month < 0) & (unmet > 1e-9)
            else:
                house = accts[0].holdings.sum(axis=1)
                for acct in accts[1:]:
                    house = house + acct.holdings.sum(axis=1)
                newly_dead = (depleted_month < 0) & (house <= 1e-9)
            depleted_month[newly_dead] = m
        else:
            acct = accts[0]
            dead_liq = acct.holdings.sum(axis=1) <= 1e-9
            if unmet_flags:
                newly_dead = (depleted_month < 0) & (unmet > 1e-9)
            else:
                newly_dead = (depleted_month < 0) & dead_liq
            depleted_month[newly_dead] = m
            acct.holdings[dead_liq] = 0.0
            acct.basis[dead_liq] = 0.0

        # Market returns for the month.
        for acct in accts:
            acct.holdings *= 1 + path_returns[:, m, :]
            np.maximum(acct.holdings, 0.0, out=acct.holdings)

        if taxed:
            # Dividends/interest arrive inside the total return and are
            # reinvested; tax them at each asset's income rate and step the
            # basis up by the reinvested amount (it was already taxed).
            acct = accts[tax_i]
            income_amt = acct.holdings * path_income[:, m, :]
            int_acc += income_amt @ interest_mask
            div_acc += income_amt @ dividend_mask
            acct.basis += income_amt
            acct.income_credit = income_amt  # spendable by next month's withdrawal
            if ladder_coupons is not None and m // 12 < len(ladder_coupons):
                t = m // 12
                int_acc += (
                    ladder_coupons[t] / 12 * cum_inflation[:, m + 1]
                    + ladder_principal[t]
                    * (cum_inflation[:, m + 1] - cum_inflation[:, m])
                )

        # Rebalance each account back to its target weights (gliding or
        # rule-driven in single-account mode; static per account otherwise).
        if (m + 1) % cfg.rebalance_months == 0:
            for i, acct in enumerate(accts):
                if acct.weights.sum() <= 0:
                    continue  # all-ladder account: no liquid targets
                tot = acct.holdings.sum(axis=1)
                if not multi and cfg.allocation_rule is not None:
                    state = RuleState(
                        month=m,
                        n_months=n_months,
                        balance=tot,
                        price_level=cum_inflation[:, m + 1],
                        initial=initial_total,
                        monthly_withdrawal_real=(
                            float(spend_real_m[m]) if w.kind == "fixed_real" else 0.0
                        ),
                        hist_indices=months[:, m],
                    )
                    targets = np.asarray(cfg.allocation_rule(state), dtype=float)
                    if not np.allclose(targets.sum(axis=-1), 1.0, atol=1e-6):
                        raise ValueError(
                            "allocation_rule returned weights that do not sum to 1"
                        )
                elif not multi:
                    targets = target_w[m][None, :]
                else:
                    targets = acct.weights[None, :]
                new_holdings = tot[:, None] * targets
                if acct.taxed:
                    sold = np.maximum(acct.holdings - new_holdings, 0.0)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        gain_frac = np.where(
                            acct.holdings > 0, 1 - acct.basis / acct.holdings, 0.0
                        )
                    realized += (sold * gain_frac).sum(axis=1)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        shrink = np.where(
                            acct.holdings > 0, new_holdings / acct.holdings, 0.0
                        )
                    acct.basis[:] = np.where(
                        new_holdings < acct.holdings,
                        acct.basis * shrink,
                        acct.basis + (new_holdings - acct.holdings),
                    )
                acct.holdings = new_holdings

        # Tax settlement is independent of the rebalance schedule: annually
        # for brackets (a tax year), quarterly for flat rates.
        annual = (m + 1) % 12 == 0

        # Flat-rate taxable settlement, quarterly.
        if taxed and cfg.tax_brackets is None and (m + 1) % 3 == 0:
            net = realized + loss_carry
            gains = np.maximum(net, 0.0)
            loss_carry = np.minimum(net, 0.0)
            realized = np.zeros(n_paths)
            tax = (
                cfg.tax_rate * (gains + div_acc)
                + ord_rate * int_acc
                + (ord_rate + cfg.state_tax) * other_ord_acc
                + cfg.state_tax * (gains + div_acc)
            )
            div_acc = np.zeros(n_paths)
            int_acc = np.zeros(n_paths)
            other_ord_acc = np.zeros(n_paths)
            paid, from_trad_pay = collect(tax, tax_i)
            total_tax_real += paid / cum_inflation[:, m + 1]
            if trad:
                dist_acc += from_trad_pay

        # Brackets: one joint annual settlement - taxable investment income
        # and traditional distributions stack through the same brackets and
        # share one standard deduction, as on a real return. Paid from the
        # taxable account first when there is one, else the traditional
        # account, then the remaining accounts; the portion paid from the
        # IRA is itself a distribution.
        if cfg.tax_brackets is not None and (taxed or trad) and annual:
            from .tax import annual_tax

            gains = 0.0
            if taxed:
                net = realized + loss_carry
                gains = np.maximum(net, 0.0)
                loss_carry = np.minimum(net, 0.0)
                realized = np.zeros(n_paths)
            if trad:
                deemed = rmd_deemed(m)
                ordinary = dist_acc + deemed + other_ord_acc
            else:
                ordinary = other_ord_acc
            tax = annual_tax(
                int_acc if taxed else 0.0,
                (div_acc + gains) if taxed else 0.0,
                cfg.tax_brackets, cum_inflation[:, m + 1],
                state_rate=cfg.state_tax, other_ordinary=ordinary,
            )
            if taxed:
                div_acc = np.zeros(n_paths)
                int_acc = np.zeros(n_paths)
            other_ord_acc = np.zeros(n_paths)
            paid, from_trad_pay = collect(tax, tax_i if taxed else trad_i)
            total_tax_real += paid / cum_inflation[:, m + 1]
            if trad:
                dist_acc = from_trad_pay

        # Flat-rate traditional settlement, annual. Withholding semantics:
        # paid from the IRA first (that portion is a further distribution),
        # falling back to the other accounts.
        if trad and cfg.tax_brackets is None and annual:
            deemed = rmd_deemed(m)
            ordinary = dist_acc + deemed + (other_ord_acc if not taxed else 0.0)
            tax = (ord_rate + cfg.state_tax) * ordinary
            if not taxed:
                other_ord_acc = np.zeros(n_paths)
            paid, from_trad_pay = collect(tax, trad_i)
            total_tax_real += paid / cum_inflation[:, m + 1]
            dist_acc = from_trad_pay  # the IRA-paid portion is a distribution

        # Pension-only settlement: tax-free accounts, but taxable outside
        # income still runs through the active regime, annually.
        if pension_only and annual:
            if cfg.tax_brackets is not None:
                from .tax import annual_tax

                tax = annual_tax(
                    0.0, 0.0, cfg.tax_brackets, cum_inflation[:, m + 1],
                    state_rate=cfg.state_tax, other_ordinary=other_ord_acc,
                )
            else:
                tax = (ord_rate + cfg.state_tax) * other_ord_acc
            other_ord_acc = np.zeros(n_paths)
            paid, _ = collect(tax, draw_order[0])
            total_tax_real += paid / cum_inflation[:, m + 1]

        # Penalties (early withdrawals, non-qualified 529 earnings) settle
        # annually, paid from the taxable account first, then the others;
        # the payment is not a distribution.
        if annual and penalty_acc.any():
            due = penalty_acc
            penalty_acc = np.zeros(n_paths)
            pay_order = ([tax_i] if tax_i is not None else []) + [
                i for i in draw_order if i != tax_i
            ]
            paid = np.zeros(n_paths)
            for i in pay_order:
                tot = accts[i].holdings.sum(axis=1)
                pay = np.minimum(due, tot)
                with np.errstate(invalid="ignore", divide="ignore"):
                    scale_p = np.where(tot > 0, (tot - pay) / tot, 1.0)
                prorata_flow(accts[i], scale_p)
                due = due - pay
                paid = paid + pay
            total_tax_real += paid / cum_inflation[:, m + 1]

        tot_end = accts[0].holdings.sum(axis=1)
        for acct in accts[1:]:
            tot_end = tot_end + acct.holdings.sum(axis=1)
        if acct_ladders:
            tot_end = tot_end + lad_val_real[m + 1] * cum_inflation[:, m + 1]
        balance[:, m + 1] = tot_end

    # A settlement in the final month can zero a path after its last
    # depletion check ran. (Not in unmet mode: a ladder-annuitized path can
    # legitimately end at exactly zero with every dollar delivered.)
    if not unmet_mode:
        final_dead = (depleted_month < 0) & (balance[:, -1] <= 1e-9)
        depleted_month[final_dead] = n_months - 1

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
        account_kinds=tuple(kinds) if multi else None,
        account_terminal=(
            np.stack(
                [
                    a.holdings.sum(axis=1)
                    + (
                        acct_lad_val[i][n_months] * cum_inflation[:, -1]
                        if i in acct_ladders
                        else 0.0
                    )
                    for i, a in enumerate(accts)
                ],
                axis=1,
            )
            if multi
            else None
        ),
        ladder_annual=ladder_annual_total or None,
        total_unmet_real=total_unmet_real if acct_ladders else None,
    )


def summarize(result: SimResult, real: bool = True) -> dict:
    """Headline numbers for one simulation run."""
    bal = result.real_balance if real else result.balance
    terminal = bal[:, -1]
    pct = np.percentile(terminal, [5, 25, 50, 75, 95])
    years = result.config.years
    initial = total_initial(result.config)
    # Depleted paths count as -100%: excluding them would report survivor-only
    # growth next to all-path terminal percentiles.
    with np.errstate(divide="ignore"):
        cagr = np.where(terminal > 0, (terminal / initial) ** (1 / years) - 1, -1.0)
    return {
        "n_paths": result.n_paths,
        "success_rate": result.success_rate,
        "terminal_p5": pct[0],
        "terminal_p25": pct[1],
        "terminal_median": pct[2],
        "terminal_p75": pct[3],
        "terminal_p95": pct[4],
        "median_cagr": float(np.median(cagr)),
        "prob_loss": float((terminal < initial).mean()),
        "sample_window": f"{result.window.min()}..{result.window.max()}",
    }
