"""Paired test of each physics config against its over-funded RMSE partner."""
# run_anticontrols.py
import physq_path

import argparse
import csv
import os

import torch

from run_paired_stats import FIELDS, compare_pairs

PAIRS = [("W8A8_knee", "W8A8_rmseup_knee"),
         ("W8A8_span1", "W8A8_rmseup_span1"),
         ("W4W8_span1", "W4W8_rmseup_span1")]

OUT_FIELDS = FIELDS + ["cost_excess_pct"]


def cost_excess(csv_path):
    if not os.path.exists(csv_path):
        return {}
    cost = {}
    for r in csv.DictReader(open(csv_path)):
        for k in ("predicted_cost", "cost"):
            if r.get(k):
                cost[r["tag"]] = float(r[k])
                break
    out = {}
    for a, b in PAIRS:
        if a in cost and b in cost and cost[a] > 0:
            out[a] = 100.0 * (cost[b] - cost[a]) / cost[a]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pt", default=os.path.join("harness_results_merged",
                                                 "harness_results.pt"))
    ap.add_argument("--outdir", default="harness_results_merged")
    ap.add_argument("--lead", type=int, default=120)
    ap.add_argument("--family", default="wind_balance")
    a = ap.parse_args(argv)

    pt = torch.load(a.pt, map_location="cpu", weights_only=False)
    rows = compare_pairs(pt, lead=a.lead, family=a.family, pairs=PAIRS)
    if not rows:
        raise SystemExit("no over-funded pairs present in the .pt -- nothing to test")

    excess = cost_excess(os.path.join(a.outdir, "harness_results.csv"))
    for r in rows:
        r["cost_excess_pct"] = excess.get(r["pair"].split("|")[0], float("nan"))

    out = os.path.join(a.outdir, "harness_results_anticontrol.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {out}\n\n{len(rows)} tested\n")
    print(f"{'pair':46}{'ratio':>7}{'+cost%':>9}")
    for r in rows:
        print(f"  {r['pair']:44}{r['ratio']:7.2f}{r['cost_excess_pct']:9.1f}")


if __name__ == "__main__":
    main()
