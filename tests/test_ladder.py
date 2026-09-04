"""Tests for TIPS ladder construction."""

import numpy as np
import pytest

from poorcast.ladder import build_ladder, rung_faces


def test_zero_yield_ladder_costs_face_value():
    lad = build_ladder(40_000, 30, 0.0)
    assert np.isclose(lad.cost, 40_000 * 30)
    assert np.allclose(rung_faces(40_000, 30, 0.0), 40_000)


def test_cash_flows_match_floor_exactly():
    # each year's income = maturing face + coupons from rungs still alive
    annual, years, ry = 40_000, 30, 0.02
    faces = rung_faces(annual, years, ry)
    for y in range(1, years + 1):
        income = faces[y - 1] + ry * faces[y - 1 :].sum()
        assert np.isclose(income, annual), f"year {y}"


def test_positive_yield_is_cheaper():
    c0 = build_ladder(40_000, 30, 0.0).cost
    c2 = build_ladder(40_000, 30, 0.02).cost
    assert c2 < c0
    assert np.isclose(c2, 895_858, rtol=1e-3)  # known value from design analysis


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError, match="positive"):
        build_ladder(0, 30, 0.02)
    with pytest.raises(ValueError, match="unsupported"):
        build_ladder(40_000, 30, 0.5)


def test_curve_pricing_flat_curve_matches_scalar():
    from poorcast.ladder import build_ladder_curve

    flat = build_ladder(40_000, 30, 0.02)
    curve = build_ladder_curve(40_000, 30, {5: 0.02, 30: 0.02})
    assert np.isclose(flat.cost, curve.cost)


def test_curve_cash_flows_match_floor_with_varying_coupons():
    from poorcast.ladder import build_ladder_curve

    lad = build_ladder_curve(40_000, 30, {5: 0.02, 10: 0.023, 20: 0.027, 30: 0.029})
    f, c = np.array(lad.faces), np.array(lad.coupons)
    for y in range(1, 31):
        income = f[y - 1] + (c[y - 1:] * f[y - 1:]).sum()
        assert np.isclose(income, 40_000), f"year {y}"
    # upward curve: long rungs discounted harder -> cheaper than flat-at-short
    assert lad.cost < build_ladder(40_000, 30, 0.02).cost


def test_taxable_ladder_income_taxed_in_engine():
    import pandas as pd

    from poorcast.simulate import SimConfig, simulate

    idx = pd.period_range("1960-01", periods=480, freq="M")
    panel = pd.DataFrame({"a": np.zeros(480), "inflation": np.zeros(480),
                          "income_a": np.zeros(480)}, index=idx)
    lad_tax = build_ladder(50_000, 10, 0.03, taxable=True)
    lad_def = build_ladder(50_000, 10, 0.03, taxable=False)
    base = dict(allocation={"a": 1.0}, initial=1e6, years=10, n_sims=2, seed=0,
                tax_rate=0.0, tax_ordinary=0.40)
    r_tax = simulate(panel, SimConfig(**base, ladder=lad_tax))
    r_def = simulate(panel, SimConfig(**base, ladder=lad_def))
    assert np.allclose(r_def.total_tax_real, 0.0)
    # zero inflation -> no accrual; tax = 40% of total coupon income exactly
    expected = 0.40 * lad_tax.coupon_income_real().sum()
    assert np.allclose(r_tax.total_tax_real, expected, rtol=1e-9)
    assert (r_tax.balance[:, -1] < r_def.balance[:, -1]).all()


def test_tail_yield_prices_beyond_curve_rungs():
    from poorcast.ladder import build_ladder_curve

    import numpy as np

    curve = {5: 0.02, 30: 0.03}
    base = build_ladder_curve(100.0, 40, curve)
    capped = build_ladder_curve(100.0, 40, curve, tail_yield=0.01)
    # The whole ladder re-solves (later coupons offset earlier faces), but
    # every year must still deliver exactly `annual`, and the low-yield tail
    # must make the ladder dearer.
    for spec in (base, capped):
        f, c = np.array(spec.faces), np.array(spec.coupons)
        for yr in range(40):
            assert (c[yr:] * f[yr:]).sum() + f[yr] == pytest.approx(100.0)
    assert capped.cost > base.cost
    assert sum(capped.faces[30:]) > sum(base.faces[30:])


def test_ladder_rows_deliver_constant_income():
    from poorcast.ladder import build_ladder, ladder_rows

    spec = build_ladder(40_000.0, 30, 0.02)
    rows = ladder_rows(spec)
    assert len(rows) == 30
    for r in rows:  # every year's total real cash flow equals the target
        assert r["income"] == pytest.approx(40_000.0)
    assert sum(r["face"] for r in rows) == pytest.approx(spec.cost)


def test_ladder_cli_from_cost_and_config(tmp_path, capsys):
    from poorcast.cli import main

    assert main(["ladder", "--cost", "1000000", "--years", "20", "--yield", "2"]) == 0
    assert "TIPS ladder" in capsys.readouterr().out

    cfg = tmp_path / "p.toml"
    cfg.write_text(
        "horizons = [30]\n"
        "[[account]]\ntype='taxable'\nbalance=1_000_000\n"
        "allocation = { us_equities = 80, tips_ladder = 20 }\n"
        "[tips_ladder]\nyield = 2.0\n"
    )
    assert main(["ladder", "--config", str(cfg)]) == 0
    out = capsys.readouterr().out
    assert "taxable account, $200,000" in out
    assert main(["ladder"]) == 2  # neither annual/cost/config
