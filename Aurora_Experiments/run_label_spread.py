"""Per-label distortion samples for the figure strips."""
# run_label_spread.py
import physq_path

import argparse
import csv
import os

import paired_stats as ps
from run_paired_stats import DEFAULT_REMOVED_TAGS, PAIRS, _prepared

PAIR_FIELDS = ["pair", "family", "lead", "label", "physics", "rmse", "ratio",
               "contribution"]
REMOVED_FIELDS = ["tag", "family", "lead", "label", "value", "floor_value", "removed_pct"]


def pair_rows(pt=None, lead=120, family="wind_balance", pairs=None, prepared=None):
    """One row per (pair, label). `prepared` injects ({tag: deltas}, denoms) directly,
    matching run_paired_stats.compare_pairs' testing seam."""
    pairs = pairs if pairs is not None else PAIRS
    if prepared is not None:
        deltas, denoms = prepared
    else:
        tags = [t for pair in pairs for t in pair]
        deltas, denoms = _prepared(pt, lead, family, tags)
    rows = []
    for a, b in pairs:
        if a not in deltas or b not in deltas:
            continue
        va = ps.label_values(deltas[a], denoms)
        vb = ps.label_values(deltas[b], denoms)
        contrib = ps.label_contributions(deltas[b], denoms,
                                         ps.axis_value(deltas[a], denoms,
                                                       slice(None)))
        for label, ratio in ps.label_ratios(deltas[a], deltas[b], denoms).items():
            if va[label] <= 1e-12:
                ratio = float("nan")
            rows.append({"pair": f"{a}|{b}", "family": family, "lead": lead,
                         "label": label, "physics": va[label], "rmse": vb[label],
                         "ratio": ratio,
                         "contribution": contrib.get(label, float("nan"))})
    return rows


def removed_rows(pt=None, lead=120, family="wind_balance", tags=None,
                 floor_tag="W8A8_floor", prepared=None):
    """One row per (tag, label): percent of that label's floor distortion removed."""
    tags = tags or DEFAULT_REMOVED_TAGS
    if prepared is not None:
        deltas, denoms = prepared
    else:
        deltas, denoms = _prepared(pt, lead, family, list(tags) + [floor_tag])
    if floor_tag not in deltas:
        return []
    floor = ps.label_values(deltas[floor_tag], denoms)
    rows = []
    for t in tags:
        if t not in deltas:
            continue
        for label, val in ps.label_values(deltas[t], denoms).items():
            base = floor.get(label)
            if base is None:
                continue
            removed_pct = float("nan") if base <= 1e-12 else 100.0 * (1.0 - val / base)
            rows.append({"tag": t, "family": family, "lead": lead, "label": label,
                         "value": val, "floor_value": base,
                         "removed_pct": removed_pct})
    return rows


def write_csv(path, fields, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path} ({len(rows)} rows)")


def parse_pairs(text):
    if not text:
        return None
    return [tuple(p.split("|")) for p in text.split(",") if p]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pt", default="harness_results_n48/harness_results.pt")
    ap.add_argument("--outdir", default="harness_results_n48")
    ap.add_argument("--lead", type=int, default=120)
    ap.add_argument("--family", default="wind_balance")
    ap.add_argument("--pairs", default=None,
                    help="comma-separated 'physics|rmse' pairs, overriding PAIRS -- use to "
                         "emit the sample behind a DIFFERENT pair set (e.g. the over-funded "
                         "anticontrols) on the same family")
    ap.add_argument("--out", default=None,
                    help="path for the per-pair CSV (default "
                         "<outdir>/harness_results_labels.csv)")
    a = ap.parse_args(argv)

    pairs = parse_pairs(a.pairs)

    import torch
    pt = torch.load(a.pt, map_location="cpu", weights_only=False)
    write_csv(a.out or os.path.join(a.outdir, "harness_results_labels.csv"), PAIR_FIELDS,
              pair_rows(pt, lead=a.lead, family=a.family, pairs=pairs))
    if pairs is None:
        write_csv(os.path.join(a.outdir, "harness_damage_removed_labels.csv"),
                  REMOVED_FIELDS, removed_rows(pt, lead=a.lead, family=a.family))


if __name__ == "__main__":
    main()
