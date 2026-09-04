"""Read `poorcast run` settings from a TOML file (--config).

Keys mirror the CLI flags (dashes -> underscores, grouped into sections);
structured forms are provided where the flag syntax is a compressed
mini-language (withdrawal schedules, income streams), and the CLI string
forms remain valid as values. load_config() returns a dict of argparse
dest -> value that the CLI applies via set_defaults(), so explicit
command-line flags always win over the file.

Unknown keys are hard errors (with a did-you-mean hint): a silently ignored
typo would produce a confidently wrong forecast.
"""

from __future__ import annotations

import difflib
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]


class ConfigError(Exception):
    pass


def _reject_unknown(given: dict, allowed: set[str], where: str) -> None:
    unknown = [k for k in given if k not in allowed]
    if unknown:
        hints = []
        for k in unknown:
            m = difflib.get_close_matches(k, sorted(allowed), n=1)
            hints.append(repr(k) + (f" (did you mean {m[0]!r}?)" if m else ""))
        raise ConfigError(f"unknown key(s) in {where}: {', '.join(hints)}")


def _num(x, where: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ConfigError(f"{where} must be a number, got {x!r}")
    return float(x)


def _int(x, where: str) -> int:
    if isinstance(x, bool) or not isinstance(x, int):
        raise ConfigError(f"{where} must be an integer, got {x!r}")
    return x


def _bool(x, where: str) -> bool:
    if not isinstance(x, bool):
        raise ConfigError(f"{where} must be true or false, got {x!r}")
    return x


def _str(x, where: str, choices: tuple[str, ...] | None = None) -> str:
    if not isinstance(x, str):
        raise ConfigError(f"{where} must be a string, got {x!r}")
    if choices and x not in choices:
        raise ConfigError(f"{where} must be one of {', '.join(choices)}, got {x!r}")
    return x


def _allocation(table, where: str) -> dict[str, float]:
    if not isinstance(table, dict) or not table:
        raise ConfigError(f"[{where}] must be a table of asset = percent entries")
    alloc = {k: _num(v, f"{where}.{k}") / 100.0 for k, v in table.items()}
    total = sum(alloc.values())
    if abs(total - 1.0) > 1e-6:
        raise ConfigError(
            f"[{where}] percentages sum to {total * 100:g}, expected 100"
        )
    return alloc


def _amount_text(x, where: str) -> str:
    """A dollar number or a '4%'-style string, as CLI text."""
    if isinstance(x, str):
        return x
    return f"{_num(x, where):g}"


def _schedule_text(segs, where: str) -> str:
    """[{amount, from, to?}, ...] -> the CLI schedule string."""
    if not isinstance(segs, list) or not segs:
        raise ConfigError(f"{where} must be an array of {{amount, from, to}} tables")
    parts = []
    for i, seg in enumerate(segs):
        w = f"{where}[{i}]"
        if not isinstance(seg, dict):
            raise ConfigError(f"{w} must be a table like {{amount = 80_000, from = 65}}")
        _reject_unknown(seg, {"amount", "from", "to"}, w)
        for key in ("amount", "from"):
            if key not in seg:
                raise ConfigError(f"{w} needs `{key}`")
        amt = _amount_text(seg["amount"], f"{w}.amount")
        frm = _int(seg["from"], f"{w}.from")
        if "to" in seg:
            parts.append(f"{amt}:{frm}-{_int(seg['to'], f'{w}.to')}")
        else:
            parts.append(f"{amt}:{frm}+")
    return ",".join(parts)


def _stream_text(entry, where: str, amount_key: str, require_at: bool) -> str:
    """A table {annual/amount, at?} or a ready CLI string like '30000@67'."""
    if isinstance(entry, str):
        return entry
    if not isinstance(entry, dict):
        raise ConfigError(
            f"{where} entries must be tables like {{{amount_key} = 30_000, at = 67}}"
        )
    _reject_unknown(entry, {amount_key, "at"}, where)
    if amount_key not in entry:
        raise ConfigError(f"{where} needs `{amount_key}`")
    amt = _num(entry[amount_key], f"{where}.{amount_key}")
    if "at" in entry:
        return f"{amt:g}@{_int(entry['at'], f'{where}.at')}"
    if require_at:
        raise ConfigError(f"{where} needs `at` (the age it happens)")
    return f"{amt:g}"


def _streams(value, where: str, amount_key: str = "annual", require_at: bool = False):
    if not isinstance(value, list):
        raise ConfigError(f"{where} must be an array (use [[{where}]] tables)")
    return [
        _stream_text(e, f"[[{where}]] #{i + 1}", amount_key, require_at)
        for i, e in enumerate(value)
    ]


TOP_KEYS = {
    "age", "initial", "horizons", "contribute", "optimize", "fees",
    "allocation", "glide", "withdrawal", "income", "pension", "expense",
    "taxes", "tips_ladder", "simulation", "output",
    "account", "withdraw_order", "adjustments", "pe_path",
}
ACCOUNT_KINDS = ("taxable", "traditional", "roth", "529")


def load_config(path: str) -> dict:
    """Parse a TOML config into {argparse dest: value} overrides for `run`."""
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config file not found: {path}")
    try:
        raw = tomllib.loads(p.read_text())
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path}: {e}")
    _reject_unknown(raw, TOP_KEYS, path)
    out: dict = {}

    if "age" in raw:
        out["age"] = _int(raw["age"], "age")
    if "initial" in raw:
        out["initial"] = _num(raw["initial"], "initial")
    if "contribute" in raw:
        out["contribute"] = _num(raw["contribute"], "contribute")
    if "fees" in raw:
        out["fees"] = _num(raw["fees"], "fees")
    if "optimize" in raw:
        out["optimize"] = _bool(raw["optimize"], "optimize")
    if "horizons" in raw:
        h = raw["horizons"]
        if isinstance(h, list):
            out["horizons"] = ",".join(str(_int(y, "horizons")) for y in h)
        else:
            out["horizons"] = str(_int(h, "horizons"))

    if "allocation" in raw:
        out["allocation"] = _allocation(raw["allocation"], "allocation")

    # Assumed P/E valuation path for US equities, e.g.
    # pe_path = [{year = 0, pe = 30}, {year = 10, pe = 20}, {year = 40, pe = 30}]
    if "pe_path" in raw:
        pts = raw["pe_path"]
        if not isinstance(pts, list) or len(pts) < 2:
            raise ConfigError(
                "pe_path must be an array of at least two {year, pe} points"
            )
        parts = []
        for i, p in enumerate(pts):
            where = f"pe_path[{i}]"
            if not isinstance(p, dict):
                raise ConfigError(f"{where} must be a table like {{year = 10, pe = 20}}")
            _reject_unknown(p, {"year", "pe"}, where)
            for key in ("year", "pe"):
                if key not in p:
                    raise ConfigError(f"{where} needs `{key}`")
            parts.append(
                f"{_num(p['pe'], where + '.pe'):g}@{_num(p['year'], where + '.year'):g}"
            )
        out["pe_path"] = ",".join(parts)

    # Per-asset annual return adjustments in %/yr (e.g. anchoring bond
    # returns at today's yields): [adjustments] us_bonds_10yr = -1.1
    if "adjustments" in raw:
        adj = raw["adjustments"]
        if not isinstance(adj, dict) or not adj:
            raise ConfigError("[adjustments] must be a table of asset = percent/yr")
        out["adjust"] = {k: _num(v, f"adjustments.{k}") for k, v in adj.items()}

    # Multi-account household: repeated [[account]] sections.
    if "account" in raw:
        entries = raw["account"]
        if not isinstance(entries, list) or not entries:
            raise ConfigError(
                "account must be given as [[account]] sections (type, balance, "
                "optional allocation/cost_basis)"
            )
        accounts = []
        for i, e in enumerate(entries):
            where = f"[[account]] #{i + 1}"
            if not isinstance(e, dict):
                raise ConfigError(f"{where} must be a table")
            _reject_unknown(
                e, {"type", "balance", "allocation", "cost_basis", "schedule"}, where
            )
            for key in ("type", "balance"):
                if key not in e:
                    raise ConfigError(f"{where} needs `{key}`")
            acct = {
                "kind": _str(e["type"], f"{where}.type", ACCOUNT_KINDS),
                "balance": _num(e["balance"], f"{where}.balance"),
            }
            if "allocation" in e:
                acct["allocation"] = _allocation(e["allocation"], f"{where}.allocation")
            if "cost_basis" in e:
                acct["cost_basis"] = _num(e["cost_basis"], f"{where}.cost_basis")
            if "schedule" in e:
                # Kept as CLI schedule text; the CLI resolves ages via --age.
                acct["schedule"] = _schedule_text(e["schedule"], f"{where}.schedule")
            accounts.append(acct)
        out["accounts"] = accounts
    if "withdraw_order" in raw:
        order = raw["withdraw_order"]
        if not isinstance(order, list):
            raise ConfigError("withdraw_order must be an array of account types")
        out["withdraw_order"] = tuple(
            _str(k, "withdraw_order", ACCOUNT_KINDS) for k in order
        )

    if "glide" in raw:
        g = raw["glide"]
        _reject_unknown(g, {"to", "years"}, "[glide]")
        if "to" not in g:
            raise ConfigError("[glide] needs `to` (the ending allocation)")
        out["glide_to"] = _allocation(g["to"], "glide.to")
        if "years" in g:
            out["glide_years"] = _int(g["years"], "glide.years")

    if "withdrawal" in raw:
        w = raw["withdrawal"]
        _reject_unknown(
            w, {"amount", "schedule", "strategy", "flex", "decline"}, "[withdrawal]"
        )
        if "amount" in w and "schedule" in w:
            raise ConfigError("[withdrawal] takes `amount` or `schedule`, not both")
        if "amount" in w:
            out["withdraw"] = _amount_text(w["amount"], "withdrawal.amount")
        if "schedule" in w:
            out["withdraw"] = _schedule_text(w["schedule"], "withdrawal.schedule")
        if "strategy" in w:
            out["withdraw_strategy"] = _str(
                w["strategy"], "withdrawal.strategy",
                ("fixed-real", "percent-of-balance"),
            )
        if "flex" in w:
            out["flex"] = 75.0 if w["flex"] is True else _num(w["flex"], "withdrawal.flex")
        if "decline" in w:
            d = w["decline"]
            if isinstance(d, dict):
                _reject_unknown(d, {"rate", "from"}, "withdrawal.decline")
                if "rate" not in d:
                    raise ConfigError("withdrawal.decline needs `rate` (%/yr)")
                rate = _num(d["rate"], "withdrawal.decline.rate")
                if "from" in d:
                    frm = _int(d["from"], "withdrawal.decline.from")
                    out["spend_decline"] = f"{rate:g}@{frm}"
                else:
                    out["spend_decline"] = f"{rate:g}"
            else:
                out["spend_decline"] = f"{_num(d, 'withdrawal.decline'):g}"

    if "income" in raw:
        out["income"] = _streams(raw["income"], "income")
    if "pension" in raw:
        out["pension"] = _streams(raw["pension"], "pension")
    if "expense" in raw:
        out["expense"] = _streams(
            raw["expense"], "expense", amount_key="amount", require_at=True
        )

    if "taxes" in raw:
        t = raw["taxes"]
        _reject_unknown(
            t,
            {"account", "filing", "state", "cost_basis", "rate", "ordinary",
             "early_penalty"},
            "[taxes]",
        )
        if "early_penalty" in t:
            out["no_early_penalty"] = not _bool(t["early_penalty"], "taxes.early_penalty")
        if "account" in t:
            out["account"] = _str(
                t["account"], "taxes.account",
                ("taxable", "traditional", "roth", "529"),
            )
        if "filing" in t:
            out["filing"] = _str(t["filing"], "taxes.filing", ("single", "married"))
        if "state" in t:
            out["state_tax"] = _num(t["state"], "taxes.state")
        if "cost_basis" in t:
            out["cost_basis"] = _num(t["cost_basis"], "taxes.cost_basis")
        if "rate" in t:
            out["tax_rate"] = _num(t["rate"], "taxes.rate")
        if "ordinary" in t:
            out["tax_ordinary"] = _num(t["ordinary"], "taxes.ordinary")

    if "tips_ladder" in raw:
        lad = raw["tips_ladder"]
        _reject_unknown(
            lad, {"annual", "yield", "curve", "deferred", "years"}, "[tips_ladder]"
        )
        # `annual` sizes the external (income-targeted) ladder; without it the
        # section just prices tips_ladder allocation entries.
        if "annual" in lad and "years" in lad:
            raise ConfigError(
                "[tips_ladder] `years` applies to tips_ladder allocations; "
                "the external `annual` ladder always spans the horizon"
            )
        if "annual" in lad:
            out["tips_ladder"] = _num(lad["annual"], "tips_ladder.annual")
        if "years" in lad:
            out["ladder_years"] = _int(lad["years"], "tips_ladder.years")
        if "yield" in lad:
            out["tips_ladder_yield"] = _num(lad["yield"], "tips_ladder.yield")
        if "curve" in lad:
            out["tips_ladder_curve"] = _bool(lad["curve"], "tips_ladder.curve")
        if "deferred" in lad:
            out["tips_ladder_deferred"] = _bool(lad["deferred"], "tips_ladder.deferred")

    if "simulation" in raw:
        s = raw["simulation"]
        _reject_unknown(
            s,
            {"sims", "mode", "block", "rebalance", "start", "end", "seed",
             "nominal", "multiple_expansion"},
            "[simulation]",
        )
        for key, dest, conv in [
            ("sims", "sims", _int), ("block", "block", _int),
            ("rebalance", "rebalance", _int), ("seed", "seed", _int),
            ("start", "start", _str), ("end", "end", _str),
            ("nominal", "nominal", _bool),
            ("multiple_expansion", "multiple_expansion", _num),
        ]:
            if key in s:
                out[dest] = conv(s[key], f"simulation.{key}")
        if "mode" in s:
            out["mode"] = _str(s["mode"], "simulation.mode", ("bootstrap", "historical"))

    if "output" in raw:
        o = raw["output"]
        _reject_unknown(o, {"dir", "charts"}, "[output]")
        if "dir" in o:
            out["out"] = _str(o["dir"], "output.dir")
        if "charts" in o:
            out["no_charts"] = not _bool(o["charts"], "output.charts")

    return out
