"""State-dependent allocation rules for use as SimConfig.allocation_rule.

Every rule maps an equity fraction to full portfolio weights via `mix_weights`,
holding the internal composition fixed (US:intl equity 5:2, bonds:cash 5:1 by
default) so experiments vary only the equity/defensive split.

These rules react to the *portfolio's own path* (balance, funded status), which
block-bootstrap sampling treats fairly. Market-signal rules (trend, volatility)
are also here but should only be used with mode='historical', where the sampled
months follow real chronology - resampling destroys the serial structure such
signals rely on.
"""

from __future__ import annotations

import numpy as np

from .simulate import RuleState

EQUITY_SPLIT = np.array([5 / 7, 2 / 7])  # us_equities, intl_equities
DEFENSIVE_SPLIT = np.array([5 / 6, 1 / 6])  # us_bonds_10yr, cash

# Asset order every rule below assumes:
ASSETS = ["us_equities", "intl_equities", "us_bonds_10yr", "cash"]


def mix_weights(equity_frac: np.ndarray) -> np.ndarray:
    """(n_paths,) equity fraction -> (n_paths, 4) weights in ASSETS order."""
    e = np.asarray(equity_frac, dtype=float)[:, None]
    return np.hstack([e * EQUITY_SPLIT[None, :], (1 - e) * DEFENSIVE_SPLIT[None, :]])


def funded_ratio_rule(
    real_rate: float = 0.015, equity_min: float = 0.30, equity_max: float = 0.80
):
    """LDI-style: bonds cover the liability (PV of remaining real withdrawals at
    a TIPS-like real rate), surplus goes to equities: e = 1 - 1/FR, clamped."""

    def rule(state: RuleState) -> np.ndarray:
        remaining = state.n_months - (state.month + 1)
        i = real_rate / 12
        af = (1 - (1 + i) ** -remaining) / i if remaining > 0 else 0.0
        liability = state.monthly_withdrawal_real * af
        with np.errstate(divide="ignore", invalid="ignore"):
            fr = np.where(
                liability > 0,
                np.divide(state.real_balance, liability, where=liability > 0,
                          out=np.full_like(state.real_balance, np.inf)),
                np.inf,
            )
        e = np.clip(1 - 1 / np.maximum(fr, 1e-9), equity_min, equity_max)
        return mix_weights(e)

    return rule


def ratchet_rule(
    threshold: float = 1.5, equity_before: float = 0.70, equity_after: float = 0.30
):
    """Bernstein's 'stop playing when you've won': permanently de-risk once the
    real balance reaches `threshold` x initial. One-way."""
    won = None
    last_month = -1

    def rule(state: RuleState) -> np.ndarray:
        nonlocal won, last_month
        # A run calls the rule at increasing months; a month that does not
        # advance means a new simulation started with the same rule object,
        # so forget the previous run's ratchet flags.
        if won is None or len(won) != len(state.balance) or state.month <= last_month:
            won = np.zeros(len(state.balance), dtype=bool)
        last_month = state.month
        won |= state.real_balance >= threshold * state.initial
        return mix_weights(np.where(won, equity_after, equity_before))

    return rule


def resurrection_rule(
    trigger: float = 0.80, equity_normal: float = 0.50, equity_underwater: float = 0.90
):
    """The anti-ratchet: crank up equity whenever the real balance is under
    `trigger` x initial. Included to quantify why 'gambling for resurrection'
    is a trap, not to recommend it."""

    def rule(state: RuleState) -> np.ndarray:
        under = state.real_balance < trigger * state.initial
        return mix_weights(np.where(under, equity_underwater, equity_normal))

    return rule


def trend_rule(signal_risk_on: np.ndarray, equity_on: float = 0.70, equity_off: float = 0.0):
    """10-month-SMA style timing. `signal_risk_on` is a boolean per historical
    month (True = equity index above its trailing average). Only meaningful with
    mode='historical'."""
    signal_risk_on = np.asarray(signal_risk_on, dtype=bool)

    def rule(state: RuleState) -> np.ndarray:
        on = signal_risk_on[state.hist_indices]
        return mix_weights(np.where(on, equity_on, equity_off))

    return rule


def vol_target_rule(
    realized_vol: np.ndarray, target_vol: float = 0.15, equity_base: float = 0.70,
    equity_cap: float = 0.90,
):
    """Volatility targeting: scale equity by target/realized vol. `realized_vol`
    is annualized trailing volatility per historical month. Only meaningful with
    mode='historical'."""
    realized_vol = np.asarray(realized_vol, dtype=float)

    def rule(state: RuleState) -> np.ndarray:
        vol = realized_vol[state.hist_indices]
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(vol > 0, target_vol / vol, 1.0)
        e = np.clip(equity_base * scale, 0.0, equity_cap)
        return mix_weights(e)

    return rule
