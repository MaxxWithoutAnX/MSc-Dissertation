""" Build all frontiers
"""
import physq_path

import os

import torch

from frontiers import flop_cost, CONSISTENCY
from allocator import load_distortion_table
from axis_noise_floor import DEFAULT_FRAC
from run_frontiers import run_frontier_B, run_frontier_A, write_random_csv

OUT = "frontiers/corrected"
LEAD = 120
MIN_EFFECT_FRAC = DEFAULT_FRAC  # 0.01, relative to the per-(axis, precision) max


def main(outdir=OUT, min_effect_frac=MIN_EFFECT_FRAC, lead=LEAD):
    OUT_ = outdir
    os.makedirs(OUT_, exist_ok=True)
    ct = torch.load("cost_tables.pt", weights_only=False)
    fl = dict(min_effect_frac=min_effect_frac)

    # Frontier B: W8A8 (headline) + W8A8_sq (SmoothQuant overlay), protected-FLOP-fraction axis.
    resB = run_frontier_B("W8A8", lead=lead, outdir=OUT_, **fl)
    run_frontier_B("W8A8_sq", lead=lead, outdir=OUT_, **fl)

    # Frontier A: W4+W8 -> bf16, model-weight-bytes axis.
    run_frontier_A(w4_csv="ablation_analysis/ablations_W4/sensitivity.csv",
                   w8_csv="ablation_analysis/ablations_W8/sensitivity.csv",
                   lead=lead, outdir=OUT_, **fl)

    w8a8_csv = {"W8A8": "ablation_analysis/ablations_W8A8/sensitivity.csv"}
    d_eval = load_distortion_table(w8a8_csv, lead=lead, axes=CONSISTENCY)
    cost = flop_cost(list(d_eval), ct, floor="W8A8")
    write_random_csv(w8a8_csv, cost, "W8A8", os.path.join(OUT_, "random_W8A8.csv"),
                     seeds=range(20), lead=lead)

    print(f"\nFrontier B W8A8: {len(resB['physics'])} physics points "
          f"(min_effect_frac={min_effect_frac}) -> {OUT_}/")
    return {"outdir": OUT_, "physics_points": len(resB["physics"])}


if __name__ == "__main__":
    main()
