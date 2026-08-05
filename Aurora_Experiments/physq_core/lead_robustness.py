# lead_robustness.py
import numpy as np

import paired_stats as ps
import run_composite_stats as rcs
import run_paired_stats as rps

AXIS_FAMILIES = {"balance": "wind_balance", "conservation": "dry_air_mass"}


def build_tables(pt, lead, tags, axis_families=None, dates=None, spread_method="iqr_raw"):
    """{axis: ({tag: {label: deltas}}, {label: denom})} for one lead."""
    axis_families = axis_families or AXIS_FAMILIES
    tables = {}
    for axis, fam in axis_families.items():
        deltas, denoms = rps._prepared(pt, lead, fam, tags, dates, spread_method)
        if deltas:
            tables[axis] = (deltas, denoms)
    return tables


def _ratio_draws(tables, pair, block, n_boot, agg="mean"):
    """(point_ratio, ratios[n_boot]) for one pair on the composite endpoint."""
    a, b = pair
    any_table = next(iter(tables.values()))[0]
    n = len(next(iter(next(iter(any_table.values())).values())))
    full = np.arange(n)
    idx = ps.boot_indices(n, block=block, n_boot=n_boot)
    pa = rcs.composite(tables, a, full, agg)
    pb = rcs.composite(tables, b, full, agg)
    ratios = np.array([rcs.composite(tables, b, i, agg)
                       / max(rcs.composite(tables, a, i, agg), 1e-12) for i in idx])
    return (pb / pa if pa > 0 else float("nan")), ratios


def trend_slope(leads, ratio_draws):
    x = np.log(np.asarray(sorted(leads), dtype=np.float64))
    x = x - x.mean()
    denom = float((x ** 2).sum())
    if denom <= 0:
        raise ValueError("need >= 2 distinct leads to fit a trend")
    Y = np.log(np.column_stack([ratio_draws[l] for l in sorted(leads)]))
    slopes = (Y - Y.mean(axis=1, keepdims=True)) @ x / denom
    lo, hi = np.percentile(slopes, [2.5, 97.5])
    point = float(np.median(slopes))
    return {"slope": point, "ci_lo": float(lo), "ci_hi": float(hi),
            "n_boot": int(slopes.size),
            "significant": bool(lo > 0.0 or hi < 0.0)}


def run(pt, pair, leads, axis_families=None, block=2, n_boot=4000, agg="mean",
        dates=None, spread_method="iqr_raw"):
    leads = sorted(leads)
    tags = list(pair)
    rows, draws = [], {}
    for lead in leads:
        tables = build_tables(pt, lead, tags, axis_families, dates, spread_method)
        if len(tables) < len(axis_families or AXIS_FAMILIES):
            raise SystemExit(
                f"lead {lead}h resolved only {sorted(tables)}; expected every axis family")
        row = rcs.compare(tables, [pair], block=block, n_boot=n_boot, agg=agg)
        if not row:
            raise SystemExit(f"pair {pair} not present in the store at lead {lead}h")
        row[0]["lead"] = lead
        rows.append(row[0])
        if len(leads) > 1:
            _, draws[lead] = _ratio_draws(tables, pair, block, n_boot, agg)
    return rows, (trend_slope(leads, draws) if len(leads) > 1 else None)
