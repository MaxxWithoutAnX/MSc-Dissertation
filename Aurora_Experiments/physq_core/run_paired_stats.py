"""Apply the paired physics-vs-RMSE comparison to every measured pair and write CSVs."""
# run_paired_stats.py
import physq_path

import argparse
import csv
import os

import numpy as np
import torch

import paired_stats as ps
from plot_common import MetricsRun

PAIRS = [("W8A8_knee", "W8A8_rmse_knee"),
         ("W8A8_span1", "W8A8_rmse_span1"),
         ("W8A8_span2", "W8A8_rmse_span2"),
         ("W8A8_span3", "W8A8_rmse_span3"),
         ("W4W8_span1", "W4W8_rmse_span1"),
         ("W4W8_span2", "W4W8_rmse_span2"),
         ("W4W8_span3", "W4W8_rmse_span3")]

FIELDS = ["pair", "family", "lead", "physics", "rmse", "ratio"]
REMOVED_FIELDS = ["tag", "family", "lead", "value", "floor_value", "removed_pct"]

DEFAULT_REMOVED_TAGS = ["W8A8_knee", "W8A8_rmse_knee", "W8A8_probe_divergent",
                        "W8A8_probe_rmse_favoured", "W8A8_span1", "W8A8_rmse_span1"]


def _runs(pt, tags):
    return {t: MetricsRun(t, pt[t]) for t in tags if t in pt}


def _prepared(pt, lead, family, tags, dates=None, spread_method="iqr_raw"):
    fp32 = MetricsRun("FP32", pt["FP32"])
    runs = _runs(pt, tags)
    if not runs:
        return {}, {}
    probe = next(iter(runs.values()))
    specs = ps.specs_for_family(fp32, probe, family, lead)
    out, denoms = {}, {}
    for t, run in runs.items():
        d, denoms = ps.family_deltas(fp32, run, specs, lead, dates, spread_method)
        out[t] = d
    return out, denoms


def compare_pairs(pt=None, lead=120, family="wind_balance", pairs=None,
                  dates=None, spread_method="iqr_raw", prepared=None):
    pairs = pairs if pairs is not None else PAIRS
    if prepared is not None:
        deltas, denoms = prepared
    else:
        tags = [t for pair in pairs for t in pair]
        deltas, denoms = _prepared(pt, lead, family, tags, dates, spread_method)
    if not deltas or not any(deltas.values()):
        return []
    rows = []
    for a, b in pairs:
        if a not in deltas or b not in deltas:
            continue
        r = ps.paired_ratio(deltas[a], deltas[b], denoms)
        rows.append({"pair": f"{a}|{b}", "family": family, "lead": lead,
                     "physics": r["point_a"], "rmse": r["point_b"], "ratio": r["ratio"]})
    return rows


def damage_removed(pt=None, lead=120, family="wind_balance", tags=None,
                   floor_tag="W8A8_floor", dates=None, spread_method="iqr_raw",
                   prepared=None):
    """Percent of the floor's distortion that each config removes.
    This is the form the knee headline takes: 42.7% vs 2.2%. `prepared` as in compare_pairs."""
    tags = tags or DEFAULT_REMOVED_TAGS
    if prepared is not None:
        deltas, denoms = prepared
    else:
        deltas, denoms = _prepared(pt, lead, family, list(tags) + [floor_tag],
                                   dates, spread_method)
    if floor_tag not in deltas:
        return []
    n = len(next(iter(deltas[floor_tag].values())))
    full = np.arange(n)
    fl = deltas[floor_tag]
    base = ps.axis_value(fl, denoms, full)
    rows = []
    for t in tags:
        if t not in deltas:
            continue
        val = ps.axis_value(deltas[t], denoms, full)
        rows.append({"tag": t, "family": family, "lead": lead, "value": val,
                     "floor_value": base, "removed_pct": 100.0 * (1.0 - val / max(base, 1e-12))})
    return rows


def _write(path, fields, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {path} ({len(rows)} rows)", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pt", default=os.path.join("harness_results", "harness_results.pt"))
    ap.add_argument("--lead", type=int, default=120)
    ap.add_argument("--family", default="wind_balance")
    ap.add_argument("--outdir", default="harness_results")
    ap.add_argument("--removed-tags", default=None, dest="removed_tags",
                    help="comma-separated tags for the damage-removed table. Defaults to "
                         "DEFAULT_REMOVED_TAGS; override to add configs that did not exist "
                         "when that list was frozen (e.g. W8A8_knee__superseded).")
    ap.add_argument("--pairs", default=None,
                    help="comma-separated 'physics|rmse' pairs, overriding PAIRS.")
    a = ap.parse_args(argv)

    pt = torch.load(a.pt, map_location="cpu", weights_only=False)
    os.makedirs(a.outdir, exist_ok=True)

    pairs = ([tuple(p.split("|")) for p in a.pairs.split(",")] if a.pairs else None)
    tags = a.removed_tags.split(",") if a.removed_tags else None
    pair_rows = compare_pairs(pt, lead=a.lead, family=a.family, pairs=pairs)
    sfx = "" if a.family == "wind_balance" else f"_{a.family}"
    _write(os.path.join(a.outdir, f"harness_results_paired{sfx}.csv"), FIELDS, pair_rows)
    rem_rows = damage_removed(pt, lead=a.lead, family=a.family, tags=tags)
    _write(os.path.join(a.outdir, f"harness_damage_removed{sfx}.csv"),
           REMOVED_FIELDS, rem_rows)

    print(f"\n=== paired physics-vs-RMSE, {a.family} @ {a.lead}h ===")
    print(f"{'pair':40}{'physics':>10}{'rmse':>10}{'ratio':>8}")
    for r in pair_rows:
        print(f"  {r['pair']:38}{r['physics']:10.4f}{r['rmse']:10.4f}{r['ratio']:8.2f}")
    print(f"\n=== damage removed vs the uniform floor ===")
    print(f"{'config':30}{'removed %':>11}")
    for r in rem_rows:
        print(f"  {r['tag']:28}{r['removed_pct']:11.1f}")


if __name__ == "__main__":
    main()
