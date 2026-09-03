"""Hand-computed tests for federal bracket taxation (2026, Rev. Proc. 2025-32)."""

import numpy as np
import pandas as pd
import pytest

from poorcast.simulate import SimConfig, Withdrawal, simulate
from poorcast.tax import annual_tax
from tests.test_taxes import panel_with_income


def t(interest, pref, filing="single", pl=1.0):
    return float(annual_tax(np.array([interest]), np.array([pref]), filing, pl)[0])


def test_ordinary_brackets_single():
    # $66,500 interest - $16,100 std = $50,400 taxable:
    # 12,400*10% + 38,000*12% = 5,800
    assert np.isclose(t(66_500, 0), 5_800.0)


def test_ltcg_zero_bracket():
    # $40,000 gains - std -> $23,900 taxable, all under the $49,450 0% edge
    assert t(0, 40_000) == 0.0


def test_ltcg_stacks_on_ordinary():
    # taxable ordinary 40,000 (tax 1,240 + 3,312 = 4,552); 30,000 pref spans
    # 40,000..70,000: 9,450 at 0%, 20,550 at 15% = 3,082.50
    assert np.isclose(t(56_100, 30_000), 4_552.0 + 3_082.50)


def test_niit_above_threshold():
    # 300k dividends: pref taxable 283,900 -> 15% on (283,900-49,450) = 35,167.50
    # NIIT: 3.8% on (300,000-200,000) = 3,800
    assert np.isclose(t(0, 300_000), 35_167.50 + 3_800.0)


def test_inflation_indexing_doubles_thresholds_but_not_niit():
    # at price level 2, double nominal ordinary income doubles the tax exactly
    assert np.isclose(t(133_000, 0, pl=2.0), 2 * 5_800.0)
    # ...but the NIIT threshold stays nominal: 600k dividends at pl=2 pays NIIT
    # on 400k, not on 2x the pl=1 amount (which would be 2*100k=200k -> 7,600)
    tax_pl2 = t(0, 600_000, pl=2.0)
    ltcg_part = 2 * 35_167.50
    assert np.isclose(tax_pl2 - ltcg_part, 0.038 * 400_000)


def test_married_thresholds():
    # MFJ: $133,000 interest - $32,200 std = $100,800 taxable:
    # 24,800*10% + 76,000*12% = 11,600
    assert np.isclose(t(133_000, 0, filing="married"), 11_600.0)


def test_engine_income_under_deduction_pays_no_tax():
    # tiny portfolio: annual income far below the standard deduction
    panel = panel_with_income(np.zeros(480), income_a=0.001)
    cfg = SimConfig(allocation={"a": 1.0}, initial=10_000.0, years=10, n_sims=2,
                    seed=0, tax_brackets="single")
    r = simulate(panel, cfg)
    assert np.allclose(r.total_tax_real, 0.0)
    assert np.allclose(r.balance[:, -1], 10_000.0)


def test_engine_brackets_settle_annually():
    # large income portfolio: interest above deduction, taxed once per year.
    # $10M at 1%/mo interest: year income 1.2M nominal, flat prices
    panel = panel_with_income(np.zeros(480), income_a=0.01, name_a="us_bonds_10yr")
    cfg = SimConfig(allocation={"us_bonds_10yr": 1.0}, initial=10_000_000.0, years=1,
                    n_sims=2, seed=0, tax_brackets="single")
    r = simulate(panel, cfg)
    income = 10_000_000 * 0.01 * 12
    expected = float(annual_tax(np.array([income]), np.array([0.0]), "single")[0])
    assert np.isclose(r.total_tax_real[0], expected, rtol=1e-9)


def test_brackets_and_flat_rates_mutually_exclusive():
    panel = panel_with_income(np.zeros(480), income_a=0.001)
    with pytest.raises(ValueError, match="mutually exclusive"):
        simulate(panel, SimConfig(allocation={"a": 1.0}, initial=1000.0, years=10,
                                  n_sims=2, seed=0, tax_brackets="single", tax_rate=0.15))


def test_state_tax_on_preferential_only():
    # state adds a flat 5% on dividends+gains...
    assert np.isclose(t2(0, 100_000, state=0.05) - t2(0, 100_000), 5_000.0)
    # ...but NOT on (Treasury) interest
    assert np.isclose(t2(100_000, 0, state=0.05), t2(100_000, 0))


def t2(interest, pref, filing="single", state=0.0):
    return float(annual_tax(np.array([interest]), np.array([pref]), filing,
                            1.0, state_rate=state)[0])


def test_muni_income_fully_exempt():
    # muni-classed asset: income taxed by neither level, even with state tax
    panel = panel_with_income(np.zeros(480), income_a=0.005, name_a="muni_bonds")
    cfg = SimConfig(allocation={"muni_bonds": 1.0}, initial=1000.0, years=10,
                    n_sims=2, seed=0, tax_brackets="single", state_tax=0.05)
    r = simulate(panel, cfg)
    assert np.allclose(r.total_tax_real, 0.0)
    assert np.allclose(r.balance[:, -1], 1000.0 * 1.0 ** 120 + 0.005 * 0)  # flat


def test_treasury_interest_state_exempt_in_engine():
    # interest-heavy portfolio: adding state tax changes nothing
    panel = panel_with_income(np.zeros(480), income_a=0.01, name_a="us_bonds_10yr")
    base = dict(allocation={"us_bonds_10yr": 1.0}, initial=10_000_000.0, years=1,
                n_sims=2, seed=0, tax_brackets="single")
    r0 = simulate(panel, SimConfig(**base))
    r5 = simulate(panel, SimConfig(**base, state_tax=0.05))
    assert np.allclose(r0.total_tax_real, r5.total_tax_real)
