"""End-to-end `poorcast run` tests on a synthetic panel (no network, no data/)."""

import numpy as np
import pandas as pd

from poorcast import cli


def synthetic_panel(start="1960-01", n=480):
    idx = pd.period_range(start, periods=n, freq="M")
    rng = np.random.default_rng(0)
    cols = {
        "us_equities": rng.normal(0.006, 0.04, n),
        "us_bonds_10yr": rng.normal(0.003, 0.01, n),
        "inflation": np.full(n, 0.002),
        "income_us_equities": np.full(n, 0.0015),
        "income_us_bonds_10yr": np.full(n, 0.003),
    }
    return pd.DataFrame(cols, index=idx)


def test_sampling_note_survives_household_without_top_level_allocation(
    tmp_path, capsys, monkeypatch
):
    monkeypatch.setattr(cli.data_mod, "load_panel", lambda: synthetic_panel())
    plan = tmp_path / "plan.toml"
    plan.write_text(
        "age = 65\nhorizons = [2]\n"
        "[[account]]\ntype = 'taxable'\nbalance = 500_000\n"
        "allocation = { us_equities = 60, us_bonds_10yr = 40 }\n"
        "[[account]]\ntype = 'roth'\nbalance = 200_000\n"
        "allocation = { us_equities = 60, us_bonds_10yr = 40 }\n"
        "[withdrawal]\namount = 30_000\n"
        "[simulation]\nsims = 20\nseed = 1\nstart = '1950-01'\n"
    )
    assert cli.main(["run", "--config", str(plan), "--no-charts"]) == 0
    out = capsys.readouterr().out
    assert "sampling starts 1960-01 (not 1950-01)" in out


def test_sampling_note_survives_tips_ladder_allocation(capsys, monkeypatch):
    monkeypatch.setattr(cli.data_mod, "load_panel", lambda: synthetic_panel())
    rc = cli.main([
        "run", "--allocation", "us_equities=60,us_bonds_10yr=20,tips_ladder=20",
        "--withdraw", "40000", "--horizons", "2", "--sims", "20", "--seed", "1",
        "--start", "1950-01", "--no-charts",
    ])
    assert rc == 0
    assert "sampling starts 1960-01" in capsys.readouterr().out


def test_glide_equity_builds_per_account_targets(tmp_path, capsys, monkeypatch):
    from poorcast import cli as climod

    captured = {}
    real = climod.simulate

    def spy(panel, cfg):
        captured["cfg"] = cfg
        return real(panel, cfg)

    monkeypatch.setattr(climod, "simulate", spy)
    monkeypatch.setattr(cli.data_mod, "load_panel", lambda: synthetic_panel())
    plan = tmp_path / "plan.toml"
    plan.write_text(
        "age = 55\nhorizons = [2]\n"
        "[[account]]\ntype = 'taxable'\nbalance = 1_000_000\n"
        "allocation = { us_equities = 60, us_bonds_10yr = 40 }\n"
        "[[account]]\ntype = 'roth'\nbalance = 100_000\n"
        "allocation = { us_equities = 50, us_bonds_10yr = 50 }\n"
        "glide_to = { us_equities = 30, us_bonds_10yr = 70 }\n"
        "[[account]]\ntype = '529'\nbalance = 100_000\n"
        "allocation = { us_equities = 50, us_bonds_10yr = 50 }\n"
        "[withdrawal]\namount = 30_000\n"
        "[simulation]\nsims = 20\nseed = 1\n"
    )
    rc = cli.main(["run", "--config", str(plan), "--glide-equity", "80",
                   "--glide-years", "1", "--no-charts"])
    assert rc == 0
    accts = captured["cfg"].accounts
    import pytest

    assert accts[0].allocation_end == pytest.approx({"us_equities": 0.8, "us_bonds_10yr": 0.2})
    assert accts[1].allocation_end == pytest.approx({"us_equities": 0.3, "us_bonds_10yr": 0.7})  # own glide_to kept
    assert accts[2].allocation_end is None  # 529 left alone
    assert captured["cfg"].glide_years == 1
    assert "gliding to 80% us_equities" in capsys.readouterr().out


def test_glide_equity_single_account(capsys, monkeypatch):
    monkeypatch.setattr(cli.data_mod, "load_panel", lambda: synthetic_panel())
    rc = cli.main(["run", "--allocation", "us_equities=40,us_bonds_10yr=60",
                   "--glide-equity", "70", "--glide-years", "1", "--withdraw", "4%",
                   "--horizons", "2", "--sims", "20", "--seed", "1", "--no-charts"])
    assert rc == 0
    assert "gliding to 70% us_equities, 30% us_bonds_10yr" in capsys.readouterr().out
    assert cli.main(["run", "--allocation", "us_equities=100", "--glide-equity", "70",
                     "--horizons", "1", "--sims", "5", "--no-charts"]) == 2  # no defensive bucket
