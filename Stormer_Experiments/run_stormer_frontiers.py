""" Generates Stormer's frontiers.
"""
import physq_path

import os

import torch

from allocator import load_distortion_table
from frontiers import CONSISTENCY, flop_cost
from run_frontiers import run_frontier, write_random_csv

HERE = os.path.dirname(os.path.abspath(__file__))

SCHEME_CSV = {
    "W4":      "ablation_analysis/ablations_W4/sensitivity.csv",
    "W8":      "ablation_analysis/ablations_W8/sensitivity.csv",
    "W8A8":    "ablation_analysis/ablations_W8A8/sensitivity.csv",
    "W8A8_sq": "ablation_analysis/ablations_W8A8_sq/sensitivity.csv",
}


DEFAULT_OUTDIR = os.path.join(HERE, "frontiers", "corrected")


def _csv(scheme):
    return os.path.join(HERE, SCHEME_CSV[scheme])


def run_all(outdir=DEFAULT_OUTDIR, lead=120):
    """Build Frontier A, Frontier B, the SmoothQuant overlay and the random band.
    Returns {tag: run_frontier result dict}."""
    os.makedirs(outdir, exist_ok=True)
    out = {}

    out["A_W4W8"] = run_frontier(
        {"W4": _csv("W4"), "W8": _csv("W8")},
        ["W4", "W8", "bf16"], "W4", "weight", "A_W4W8", lead, outdir)

    out["B_W8A8"] = run_frontier(
        {"W8A8": _csv("W8A8")},
        ["W8A8", "bf16"], "W8A8", "flop", "B_W8A8", lead, outdir)

    out["B_W8A8_sq"] = run_frontier(
        {"W8A8_sq": _csv("W8A8_sq")},
        ["W8A8_sq", "bf16"], "W8A8_sq", "flop", "B_W8A8_sq", lead, outdir)

    # Random protection-order band for the Frontier B anticontrol, on B's own cost axis.
    ct = torch.load(os.path.join(HERE, "cost_tables.pt"), weights_only=False)
    d_eval = load_distortion_table({"W8A8": _csv("W8A8")}, lead=lead, axes=CONSISTENCY)
    cost = flop_cost(list(d_eval), ct, floor="W8A8")
    write_random_csv({"W8A8": _csv("W8A8")}, cost, "W8A8",
                     os.path.join(outdir, "random_W8A8.csv"), lead=lead)
    return out


def main():
    res = run_all()
    for tag, r in res.items():
        print(f"{tag:12} physics={len(r['physics']):3d} pts  rmse={len(r['rmse']):3d} pts  "
              f"b2 variants={len(r['b2'])}", flush=True)
        order = r["b2"].get("representative|wind", [])
        print(f"             protected order (representative|wind): {order[:4]}", flush=True)


if __name__ == "__main__":
    main()
