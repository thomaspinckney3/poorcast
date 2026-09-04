"""Tests for TOML config loading (--config)."""

import pytest

from poorcast.cli import build_parser, main
from poorcast.config import ConfigError, load_config

FULL = """
age = 62
initial = 1_500_000
horizons = [25, 30]
contribute = 500
fees = 0.25

[allocation]
us_equities = 55
intl_equities = 15
us_bonds_10yr = 25
cash = 5

[withdrawal]
amount = "4%"
flex = true

[[income]]
annual = 30_000
at = 67

[[pension]]
annual = 12_000

[[expense]]
amount = 50_000
at = 70

[taxes]
account = "traditional"
filing = "married"
state = 5
cost_basis = 0.6

[tips_ladder]
annual = 30_000
yield = 1.5

[simulation]
sims = 2000
seed = 42
mode = "historical"

[output]
charts = false
"""


def write(tmp_path, text, name="plan.toml"):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def test_full_config_maps_to_arg_dests(tmp_path):
    out = load_config(write(tmp_path, FULL))
    assert out["age"] == 62
    assert out["initial"] == 1_500_000
    assert out["horizons"] == "25,30"
    assert out["contribute"] == 500
    assert out["fees"] == 0.25
    assert out["allocation"] == pytest.approx(
        {"us_equities": 0.55, "intl_equities": 0.15, "us_bonds_10yr": 0.25, "cash": 0.05}
    )
    assert out["withdraw"] == "4%"
    assert out["flex"] == 75.0
    assert out["income"] == ["30000@67"]
    assert out["pension"] == ["12000"]
    assert out["expense"] == ["50000@70"]
    assert out["account"] == "traditional"
    assert out["filing"] == "married"
    assert out["state_tax"] == 5
    assert out["cost_basis"] == 0.6
    assert out["tips_ladder"] == 30_000
    assert out["tips_ladder_yield"] == 1.5
    assert out["sims"] == 2000
    assert out["seed"] == 42
    assert out["mode"] == "historical"
    assert out["no_charts"] is True


def test_schedule_flattens_to_cli_string(tmp_path):
    text = """
    [withdrawal]
    schedule = [
      { amount = 90_000, from = 65, to = 75 },
      { amount = "5%", from = 75 },
    ]
    """
    assert load_config(write(tmp_path, text))["withdraw"] == "90000:65-75,5%:75+"


def test_cli_string_forms_accepted_for_streams(tmp_path):
    text = 'income = ["30000@67", "22000@69"]\n'
    assert load_config(write(tmp_path, text))["income"] == ["30000@67", "22000@69"]


def test_unknown_key_suggests_correction(tmp_path):
    with pytest.raises(ConfigError, match=r"withdrawl.*did you mean 'withdrawal'"):
        load_config(write(tmp_path, "[withdrawl]\namount = 4\n"))
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(write(tmp_path, "[taxes]\nfilling = 'married'\n"))


def test_allocation_must_sum_to_100(tmp_path):
    with pytest.raises(ConfigError, match="sum to 90"):
        load_config(write(tmp_path, "[allocation]\nus_equities = 90\n"))


def test_amount_and_schedule_are_exclusive(tmp_path):
    text = """
    [withdrawal]
    amount = "4%"
    schedule = [{ amount = 1, from = 65 }]
    """
    with pytest.raises(ConfigError, match="not both"):
        load_config(write(tmp_path, text))


def test_expense_requires_at(tmp_path):
    with pytest.raises(ConfigError, match="needs `at`"):
        load_config(write(tmp_path, "[[expense]]\namount = 1000\n"))


def test_missing_file_and_bad_toml(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(str(tmp_path / "nope.toml"))
    with pytest.raises(ConfigError, match="plan.toml"):
        load_config(write(tmp_path, "= not toml"))


def test_choice_values_validated(tmp_path):
    with pytest.raises(ConfigError, match="taxes.account must be one of"):
        load_config(write(tmp_path, "[taxes]\naccount = '401k'\n"))


def test_file_values_become_defaults_and_flags_win(tmp_path):
    path = write(tmp_path, FULL)
    defaults = load_config(path)
    args = build_parser(defaults).parse_args(["run", "--config", path])
    assert args.age == 62 and args.withdraw == "4%" and args.account == "traditional"
    args = build_parser(defaults).parse_args(
        ["run", "--config", path, "--withdraw", "3.5%", "--account", "taxable"]
    )
    assert args.withdraw == "3.5%" and args.account == "taxable"
    assert args.age == 62  # untouched file value still applies


def test_main_reports_config_errors(tmp_path, capsys):
    bad = write(tmp_path, "[withdrawl]\namount = 4\n")
    assert main(["run", "--config", bad]) == 2
    assert "did you mean" in capsys.readouterr().out


def test_decline_number_and_table_forms(tmp_path):
    text = "[withdrawal]\namount = '4%'\ndecline = 1.5\n"
    assert load_config(write(tmp_path, text))["spend_decline"] == "1.5"
    text = "[withdrawal]\namount = '4%'\ndecline = { rate = 1, from = 75 }\n"
    assert load_config(write(tmp_path, text))["spend_decline"] == "1@75"


def test_account_sections_parse(tmp_path):
    text = """
    withdraw_order = ["taxable", "traditional", "roth"]
    [[account]]
    type = "taxable"
    balance = 12_000_000
    cost_basis = 0.6
    allocation = { us_equities = 60, muni_bonds = 35, cash = 5 }
    [[account]]
    type = "roth"
    balance = 500_000
    """
    out = load_config(write(tmp_path, text))
    assert out["withdraw_order"] == ("taxable", "traditional", "roth")
    assert out["accounts"][0]["kind"] == "taxable"
    assert out["accounts"][0]["cost_basis"] == 0.6
    assert out["accounts"][0]["allocation"]["us_equities"] == pytest.approx(0.6)
    assert out["accounts"][1] == {"kind": "roth", "balance": 500_000}
    with pytest.raises(ConfigError, match="balance"):
        load_config(write(tmp_path, "[[account]]\ntype = 'roth'\n"))


def test_adjustments_table_parses(tmp_path):
    text = "[adjustments]\nus_bonds_10yr = -1.1\nmuni_bonds = -1.8\n"
    out = load_config(write(tmp_path, text))
    assert out["adjust"] == {"us_bonds_10yr": -1.1, "muni_bonds": -1.8}


def test_pe_path_parses_and_builds_rates(tmp_path):
    import numpy as np
    from poorcast.cli import parse_pe_path, pe_path_rates

    text = "pe_path = [{year = 0, pe = 30}, {year = 10, pe = 20}, {year = 40, pe = 30}]\n"
    out = load_config(write(tmp_path, text))
    assert out["pe_path"] == "30@0,20@10,30@40"
    pts = parse_pe_path(out["pe_path"])
    rates = pe_path_rates(pts, 480)
    assert np.allclose(rates[:120], np.log(20 / 30) / 10)
    assert np.allclose(rates[120:], np.log(30 / 20) / 30)
    # cumulative log-PE change is exactly zero over the round trip
    assert abs(rates.sum() / 12) < 1e-12


def test_optimize_section_parses(tmp_path):
    text = "[optimize]\nequity = [40, 80, 10]\nladder = [0, 4_000_000, 1_000_000]\n"
    out = load_config(write(tmp_path, text))
    assert out["optimize"] is True
    assert out["optimize_grid"] == {"equity": [40, 80, 10],
                                    "ladder": [0, 4_000_000, 1_000_000]}
    with pytest.raises(ConfigError, match="min, max, step"):
        load_config(write(tmp_path, "[optimize]\nequity = [40, 80]\n"))
