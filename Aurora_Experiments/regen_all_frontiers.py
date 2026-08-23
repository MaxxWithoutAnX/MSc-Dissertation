import physq_path

import csv
import os

import torch

from frontiers import flop_cost, b2_sweep, CONSISTENCY
from allocator import load_distortion_table
from axis_noise_floor import DEFAULT_FRAC
from run_frontiers import run_frontier_B, run_frontier_A, write_random_csv

OUT = "frontier_corrected"
LEAD = 120
MIN_EFFECT_FRAC = DEFAULT_FRAC  # 0.01, relative to the per-(axis, precision) max
AXIS_FLOOR = 0.0                # absolute rule: implemented, measured, NOT used (see
                                # axis_noise_floor.py -- the frontiers differ ~60x in scale)


def main(outdir=OUT, min_effect_frac=MIN_EFFECT_FRAC, lead=LEAD, axis_floor=AXIS_FLOOR):
    assert outdir != "frontier_flops_real", \
        "refusing to overwrite the frontiers the measured manifest was frozen from"
    OUT_ = outdir
    os.makedirs(OUT_, exist_ok=True)
    ct = torch.load("cost_tables.pt", weights_only=False)
    fl = dict(min_effect_frac=min_effect_frac, axis_floor=axis_floor)

    # Frontier B: W8A8 (headline) + W8A8_sq (SmoothQuant overlay), protected-FLOP-fraction axis.
    resB = run_frontier_B("W8A8", lead=lead, outdir=OUT_, **fl)
    run_frontier_B("W8A8_sq", lead=lead, outdir=OUT_, **fl)

    # Frontier A: W4+W8 -> bf16, model-weight-bytes axis.
    run_frontier_A(w4_csv="ablation_analysis_ablations_W4/sensitivity.csv",
                   w8_csv="ablation_analysis_ablations_W8/sensitivity.csv",
                   lead=lead, outdir=OUT_, **fl)

    w8a8_csv = {"W8A8": "ablation_analysis_ablations_W8A8/sensitivity.csv"}
    d_eval = load_distortion_table(w8a8_csv, lead=lead, axes=CONSISTENCY)
    cost = flop_cost(list(d_eval), ct, floor="W8A8")
    write_random_csv(w8a8_csv, cost, "W8A8", os.path.join(OUT_, "random_W8A8.csv"),
                     seeds=range(20), lead=lead)

    # B2 lapse/HypRel decision sweep on the W8A8 headline (FLOP cost) -- the decision table.
    sweep = b2_sweep(w8a8_csv, cost, ["W8A8", "bf16"], lead=lead, **fl)
    with open(os.path.join(OUT_, "b2_lapse_decision_W8A8.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "top8_protected_order"])
        for name, order in sweep.items():
            w.writerow([name, "|".join(order[:8])])

    print(f"\nFrontier B W8A8: {len(resB['physics'])} physics points "
          f"(axis_floor={axis_floor}, min_effect_frac={min_effect_frac}) -> {OUT_}/")
    print(f"\nB2 protected-order sweep (W8A8 @{lead}h, FLOP cost):")
    for name, order in sweep.items():
        print(f"  {name:22s} -> {order[:6]}")
    return {"outdir": OUT_, "physics_points": len(resB["physics"]), "b2": sweep}


if __name__ == "__main__":
    main()
