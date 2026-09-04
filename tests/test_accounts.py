"""Tests for income streams, expenses, spending schedules, and account types."""

import numpy as np
import pandas as pd
import pytest

from poorcast.cli import parse_at_age, parse_schedule
from poorcast.simulate import IncomeStream, SimConfig, Withdrawal, simulate
from poorcast.tax import annual_tax


def make_panel(n_months=480, ret=0.0, inflation=0.0, income=None):
    idx = pd.period_range("1960-01", periods=n_months, freq="M")
    data = {"a": np.full(n_months, ret), "inflation": np.full(n_months, inflation)}
    if income is not None:
        data["income_a"] = np.full(n_months, income)
    return pd.DataFrame(data, index=idx)


def cfg(**kw):
    base = dict(
        allocation={"a": 1.0}, initial=1000.0, years=1, n_sims=4, seed=0
    )
    base.update(kw)
    return SimConfig(**base)


# --- income streams and expenses -------------------------------------------


def test_income_offsets_withdrawal_exactly():
    panel = make_panel()
    c = cfg(
        withdrawal=Withdrawal("fixed_real", amount=120.0),
        income=(IncomeStream(60.0),),
    )
    r = simulate(panel, c)
    assert np.allclose(r.balance[:, -1], 1000.0 - 60.0)
    assert np.allclose(r.total_withdrawn, 60.0)


def test_income_surplus_is_invested():
    panel = make_panel()
    r = simulate(panel, cfg(income=(IncomeStream(120.0),)))
    assert np.allclose(r.balance[:, -1], 1000.0 + 120.0)


def test_income_start_month_respected():
    panel = make_panel()
    c = cfg(years=2, income=(IncomeStream(120.0, start_month=12),))
    r = simulate(panel, c)
    assert np.allclose(r.balance[:, 12], 1000.0)  # nothing in year 1
    assert np.allclose(r.balance[:, -1], 1000.0 + 120.0)


def test_income_is_inflation_adjusted():
    panel = make_panel(inflation=0.01)
    r = simulate(panel, cfg(income=(IncomeStream(120.0),)))
    infl = np.cumprod(np.full(12, 1.01)) / 1.01  # price level at each month start
    assert np.allclose(r.balance[:, -1], 1000.0 + (10.0 * infl).sum())


def test_one_time_expense():
    panel = make_panel()
    r = simulate(panel, cfg(expenses=((6, 250.0),)))
    assert np.allclose(r.balance[:, 6], 1000.0)
    assert np.allclose(r.balance[:, 7], 750.0)


def test_expense_can_deplete():
    panel = make_panel()
    r = simulate(panel, cfg(expenses=((3, 5000.0),)))
    assert (r.depleted_month == 3).all()


# --- spending schedules ------------------------------------------------------


def test_schedule_steps_apply_by_month():
    panel = make_panel()
    c = cfg(
        years=2,
        withdrawal=Withdrawal("fixed_real", schedule=((0, 120.0), (12, 60.0))),
    )
    r = simulate(panel, c)
    assert np.allclose(r.balance[:, 12], 1000.0 - 120.0)
    assert np.allclose(r.balance[:, -1], 1000.0 - 180.0)


def test_schedule_requires_fixed_real():
    panel = make_panel()
    with pytest.raises(ValueError, match="fixed_real"):
        simulate(panel, cfg(withdrawal=Withdrawal("none", schedule=((0, 1.0),))))


def test_parse_schedule_basic_and_gap():
    # 80k for ages 65-75, nothing 75-80, 60k from 80 on (age 65 now).
    s = parse_schedule("80000:65-75,60000:80+", age=65, initial=1e6)
    assert s == ((0, 80000.0), (120, 0.0), (180, 60000.0))


def test_parse_schedule_percent_and_past_segment():
    # A segment entirely before the current age is dropped; % of initial works.
    s = parse_schedule("4%:55-60,2%:60+", age=60, initial=1e6)
    assert s == ((0, 20000.0),)


def test_parse_schedule_bounded_end_stops_spending():
    s = parse_schedule("50000:65-70", age=65, initial=1e6)
    assert s == ((0, 50000.0), (60, 0.0))


def test_parse_schedule_rejects_overlap():
    with pytest.raises(ValueError, match="overlap"):
        parse_schedule("50000:65-75,60000:70+", age=65, initial=1e6)


def test_parse_at_age():
    assert parse_at_age("30000@67") == (30000.0, 67)
    assert parse_at_age("30,000") == (30000.0, None)


# --- account types -----------------------------------------------------------


def test_traditional_flat_rate_taxes_distributions():
    panel = make_panel()
    c = cfg(
        years=2,
        age=60,
        account="traditional",
        tax_ordinary=0.25,
        withdrawal=Withdrawal("fixed_real", amount=120.0),
    )
    r = simulate(panel, c)
    # Year 1: 120 withdrawn, tax 30 at year end. Year 2: 120 withdrawn plus
    # last year's 30 tax payment is itself a distribution -> tax 37.5.
    assert np.allclose(r.balance[:, 12], 1000.0 - 120.0 - 30.0)
    assert np.allclose(r.balance[:, -1], 850.0 - 120.0 - 37.5)
    assert np.allclose(r.total_tax_real, 67.5)


def test_traditional_brackets_standard_deduction_shields_small_distributions():
    panel = make_panel()
    c = cfg(
        initial=100_000.0,
        age=60,
        account="traditional",
        tax_brackets="single",
        withdrawal=Withdrawal("fixed_real", amount=12_000.0),
    )
    r = simulate(panel, c)  # 12k distribution < 16.1k standard deduction
    assert np.allclose(r.balance[:, -1], 100_000.0 - 12_000.0)
    assert np.allclose(r.total_tax_real, 0.0)


def test_traditional_rmd_deemed_distribution_taxed():
    panel = make_panel()
    c = cfg(age=75, account="traditional", tax_ordinary=0.20)
    r = simulate(panel, c)
    # No withdrawals; RMD at 75 = 1000/24.6, deemed distributed and taxed at
    # 20% but the dollars stay invested.
    expected_tax = 0.20 * 1000.0 / 24.6
    assert np.allclose(r.total_tax_real, expected_tax)
    assert np.allclose(r.balance[:, -1], 1000.0 - expected_tax)


def test_traditional_no_rmd_before_73():
    panel = make_panel()
    r = simulate(panel, cfg(age=60, account="traditional", tax_ordinary=0.20))
    assert np.allclose(r.total_tax_real, 0.0)


def test_traditional_requires_age():
    panel = make_panel()
    with pytest.raises(ValueError, match="age"):
        simulate(panel, cfg(account="traditional", tax_ordinary=0.25))


def test_roth_and_529_are_tax_free():
    panel = make_panel(ret=0.005)
    base = dict(withdrawal=Withdrawal("fixed_real", amount=50.0), years=5)
    r0 = simulate(panel, cfg(**base))
    for acct in ("roth", "529"):
        r = simulate(panel, cfg(**base, account=acct))
        assert np.array_equal(r.balance, r0.balance)


def test_tax_free_account_rejects_tax_settings():
    panel = make_panel()
    with pytest.raises(ValueError, match="tax-free"):
        simulate(panel, cfg(account="roth", tax_rate=0.2))


def test_pension_taxed_in_taxable_account_flat():
    panel = make_panel(income=0.0)
    c = cfg(tax_ordinary=0.25, income=(IncomeStream(120.0, taxable=True),))
    r = simulate(panel, c)
    # 10/mo pension invested; each quarterly settlement taxes 30 at 25%.
    assert np.allclose(r.balance[:, -1], 1000.0 + 120.0 - 30.0)


# --- annual_tax other_ordinary ----------------------------------------------


def test_other_ordinary_uses_deduction_and_brackets():
    tax = annual_tax(0.0, 0.0, "single", other_ordinary=np.array([16_100.0]))
    assert np.allclose(tax, 0.0)
    tax = annual_tax(0.0, 0.0, "single", other_ordinary=np.array([16_100.0 + 10_000.0]))
    assert np.allclose(tax, 1_000.0)  # all in the 10% bracket


def test_other_ordinary_raises_magi_for_niit_but_is_not_nii():
    # Single filer, 50k interest + 200k other ordinary income, hand-computed:
    # taxable ordinary = 250,000 - 16,100 = 233,900 -> bracket tax 51,304.
    # NIIT: MAGI 250k is 50k over the threshold, so all 50k of investment
    # income is NIIT-taxed (1,900) - the 200k itself is not investment income.
    tax = annual_tax(
        np.array([50_000.0]), 0.0, "single", other_ordinary=np.array([200_000.0])
    )
    ordinary = (
        0.10 * 12_400
        + 0.12 * (50_400 - 12_400)
        + 0.22 * (105_700 - 50_400)
        + 0.24 * (201_775 - 105_700)
        + 0.32 * (233_900 - 201_775)
    )
    assert np.allclose(tax, ordinary + 0.038 * 50_000.0)


# --- spending decline --------------------------------------------------------


def test_spend_decline_halves_each_year():
    panel = make_panel()
    c = cfg(
        years=2, withdrawal=Withdrawal("fixed_real", amount=120.0, decline=0.5)
    )
    r = simulate(panel, c)
    # Month 0 spends 10; a year later the target has halved to 5.
    assert np.allclose(r.balance[:, 0] - r.balance[:, 1], 10.0)
    assert np.allclose(r.balance[:, 12] - r.balance[:, 13], 5.0)


def test_spend_decline_start_month_delays_decline():
    panel = make_panel()
    c = cfg(
        years=2,
        withdrawal=Withdrawal(
            "fixed_real", amount=120.0, decline=0.5, decline_start_month=12
        ),
    )
    r = simulate(panel, c)
    assert np.allclose(r.balance[:, 12], 1000.0 - 120.0)  # year 1 at full target
    assert np.allclose(r.balance[:, 12] - r.balance[:, 13], 10.0)  # decline starts
    assert np.allclose(r.balance[:, -2] - r.balance[:, -1], 10.0 * 0.5 ** (11 / 12))


def test_spend_decline_composes_with_schedule():
    panel = make_panel()
    c = cfg(
        years=2,
        withdrawal=Withdrawal(
            "fixed_real", schedule=((0, 120.0), (12, 240.0)), decline=0.5
        ),
    )
    r = simulate(panel, c)
    assert np.allclose(r.balance[:, 0] - r.balance[:, 1], 10.0)
    assert np.allclose(r.balance[:, 12] - r.balance[:, 13], 10.0)  # 20 halved


def test_spend_decline_requires_fixed_real():
    panel = make_panel()
    with pytest.raises(ValueError, match="fixed_real"):
        simulate(panel, cfg(withdrawal=Withdrawal("none", decline=0.01)))


def test_parse_schedule_years_from_start_with_age_zero():
    # Without --age the CLI passes age=0: numbers are simulation years.
    s = parse_schedule("20000:2-6,100000:6-18", age=0, initial=1e6)
    assert s == ((24, 20000.0), (72, 100000.0), (216, 0.0))
