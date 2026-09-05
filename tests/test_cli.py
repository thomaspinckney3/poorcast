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
