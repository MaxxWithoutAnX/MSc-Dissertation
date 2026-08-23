"""Paired test on the minimax endpoint max(balance, conservation)."""
# run_composite_stats.py
import physq_path

import argparse
import csv
import os

import numpy as np
import torch

import paired_stats as ps
from run_paired_stats import PAIRS, _prepared

AXIS_FAMILIES = {"balance": "wind_balance", "conservation": "dry_air_mass"}
FIELDS = ["pair", "endpoint", "lead", "physics", "rmse", "ratio",
          "physics_argmax", "rmse_argmax"]
LABEL_FIELDS = ["pair", "endpoint", "lead", "binding_axis", "family", "label",
                "distortion", "physics_axis", "contribution"]


def composite(tables, tag, idx):
    """max over axes of the axis value on resample `idx`. NaN axes are skipped.

    There must be exactly one composite in the codebase."""
    vals = [ps.axis_value(t[tag], d, idx) for t, d in tables.values() if tag in t]
    vals = [v for v in vals if np.isfinite(v)]
    return max(vals) if vals else float("nan")


def argmax_axis(tables, tag, idx):
    best, name = -np.inf, ""
    for axis, (t, d) in tables.items():
        if tag not in t:
            continue
        v = ps.axis_value(t[tag], d, idx)
        if np.isfinite(v) and v > best:
            best, name = v, axis
    return name


def label_rows(tables, pairs, lead):
    any_table = next(iter(tables.values()))[0]
    n = len(next(iter(next(iter(any_table.values())).values())))
    full = np.arange(n)
    rows = []
    for a, b in pairs:
        if not all(a in t and b in t for t, _ in tables.values()):
            continue
        pa = composite(tables, a, full)
        axis_b = argmax_axis(tables, b, full)
        if not axis_b:
            continue
        deltas_b, denoms_b = tables[axis_b]
        contrib = ps.label_contributions(deltas_b[b], denoms_b, pa)
        vals = ps.label_values(deltas_b[b], denoms_b)
        for label, c in contrib.items():
            rows.append({"pair": f"{a}|{b}", "endpoint": "max(balance,conservation)",
                         "lead": lead, "binding_axis": axis_b,
                         "family": AXIS_FAMILIES[axis_b], "label": label,
                         "distortion": vals[label], "physics_axis": pa,
                         "contribution": c})
    return rows


def check_contributions(rows, lrows):
    by_pair = {}
    for r in lrows:
        by_pair.setdefault(r["pair"], []).append(r["contribution"])
    for r in rows:
        got = by_pair.get(r["pair"])
        if got and np.isfinite(r["ratio"]):
            m = float(np.mean(got))
            if abs(m - r["ratio"]) > 1e-9 * max(1.0, abs(r["ratio"])):
                raise SystemExit(f"{r['pair']}: mean(contribution)={m!r} != ratio="
                                 f"{r['ratio']!r} -- the strip sample is not the point")


def write_label_rows(path, lrows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LABEL_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(lrows)
    print(f"wrote {path} ({len(lrows)} rows)")


def compare(tables, pairs):
    any_table = next(iter(tables.values()))[0]
    n = len(next(iter(next(iter(any_table.values())).values())))
    full = np.arange(n)
    rows = []
    for a, b in pairs:
        if not all(a in t and b in t for t, _ in tables.values()):
            continue
        pa, pb = composite(tables, a, full), composite(tables, b, full)
        rows.append({"pair": f"{a}|{b}", "endpoint": "max(balance,conservation)",
                     "physics": pa, "rmse": pb,
                     "ratio": (pb / pa if pa > 0 else float("nan")),
                     "physics_argmax": argmax_axis(tables, a, full),
                     "rmse_argmax": argmax_axis(tables, b, full)})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pt", default=os.path.join("harness_results_merged",
                                                 "harness_results.pt"))
    ap.add_argument("--outdir", default="harness_results_merged")
    ap.add_argument("--lead", type=int, default=120)
    ap.add_argument("--pairs", default=None,
                    help="comma-separated 'physics|rmse' pairs, overriding PAIRS -- use to "
                         "score a DIFFERENT set of pairs (e.g. the over-funded anticontrols) "
                         "on the same composite endpoint")
    ap.add_argument("--out", default=None)
    ap.add_argument("--labels-out", default=None,
                    help="path for the per-label sample behind each composite ratio (see "
                         "label_rows). Default <outdir>/harness_results_composite_labels."
                         "csv; pass a path when --pairs scores a different set, so the two "
                         "runs cannot clobber each other's file.")
    a = ap.parse_args(argv)

    pairs = ([tuple(p.split("|")) for p in a.pairs.split(",")] if a.pairs else PAIRS)
    pt = torch.load(a.pt, map_location="cpu", weights_only=False)
    tags = sorted({t for pair in pairs for t in pair})
    tables = {}
    for axis, fam in AXIS_FAMILIES.items():
        deltas, denoms = _prepared(pt, a.lead, fam, tags)
        if deltas:
            tables[axis] = (deltas, denoms)
    if not tables:
        raise SystemExit("no axis families resolved -- nothing to test")

    rows = compare(tables, pairs)
    for r in rows:
        r["lead"] = a.lead

    os.makedirs(a.outdir, exist_ok=True)
    out = a.out or os.path.join(a.outdir, "harness_results_composite.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    lrows = label_rows(tables, pairs, a.lead)
    check_contributions(rows, lrows)
    write_label_rows(a.labels_out or os.path.join(
        a.outdir, "harness_results_composite_labels.csv"), lrows)

    print(f"wrote {out}\n\ntested {len(rows)}")
    print(f"\n{'pair':36}{'phys':>8}{'rmse':>8}{'ratio':>8}  binding axis")
    for r in rows:
        print(f"  {r['pair']:34}{r['physics']:8.3f}{r['rmse']:8.3f}{r['ratio']:8.2f}"
              f"  {r['physics_argmax']}/{r['rmse_argmax']}")


if __name__ == "__main__":
    main()
