# lead_robustness.py
import numpy as np

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


def trend_slope(leads, ratios):
    leads = sorted(leads)
    x = np.log(np.asarray(leads, dtype=np.float64))
    x = x - x.mean()
    denom = float((x ** 2).sum())
    if denom <= 0:
        raise ValueError("need >= 2 distinct leads to fit a trend")
    y = np.log(np.asarray([ratios[l] for l in leads], dtype=np.float64))
    y = y - y.mean()
    return {"slope": float((y @ x) / denom)}


def run(pt, pair, leads, axis_families=None, dates=None, spread_method="iqr_raw"):
    leads = sorted(leads)
    tags = list(pair)
    rows = []
    for lead in leads:
        tables = build_tables(pt, lead, tags, axis_families, dates, spread_method)
        if len(tables) < len(axis_families or AXIS_FAMILIES):
            raise SystemExit(
                f"lead {lead}h resolved only {sorted(tables)}; expected every axis family")
        row = rcs.compare(tables, [pair])
        if not row:
            raise SystemExit(f"pair {pair} not present in the store at lead {lead}h")
        row[0]["lead"] = lead
        rows.append(row[0])
    trend = (trend_slope(leads, {r["lead"]: r["ratio"] for r in rows})
             if len(leads) > 1 else None)
    return rows, trend
