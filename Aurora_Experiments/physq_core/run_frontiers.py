""" Creates the frontiers.
"""
import physq_path

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from allocator import load_distortion_table
from frontiers import (build_frontier, guide_budget_frontier, load_tables, random_band,
                       b2_sweep, weight_memory_cost, flop_cost, CONSISTENCY, _axis_sum)
from select_harness_configs import encode_config

WG = [(1.0, 1.0), (1.0, 0.0), (0.0, 1.0)]

PANELS = CONSISTENCY + ("standard",)
PANEL_TITLE = {"balance": "balance  (guide axis)",
               "conservation": "conservation  (guide axis)",
               "standard": "RMSE  (NOT optimised by the physics guide)"}


def score_on_axis(points, d_axis, axis="standard"):
    """Rescores the config on another axis
    """
    for p in points:
        if "config" in p:
            p[axis] = _axis_sum(d_axis, p["config"], (axis,))[axis]
    return points


def _write_frontier_csv(path, points, floor):
    """ Writes a frontier's points as cost/balance/consevation/protectd/config to a csv
    """
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        has_std = bool(points) and "standard" in points[0]
        w.writerow(["cost", "balance", "conservation", "protected", "config"]
                   + (["standard"] if has_std else []))
        for p in points:
            prot = "|".join(g for g, pr in p["config"].items() if pr == "bf16")
            w.writerow([f"{p['cost']:.6g}", f"{p['balance']:.6g}",
                        f"{p['conservation']:.6g}", prot,
                        encode_config(p["config"], floor)]
                       + ([f"{p['standard']:.6g}"] if has_std else []))


def write_guide_order_csv(path, points, floor):
    """ Used for control matchihng budgets. Used by select_harness_configs.match_at_budget
    """
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "cost", "balance", "conservation", "protected", "config"])
        for p in points:
            prot = "|".join(g for g, pr in p["config"].items() if pr == "bf16")
            w.writerow([p["k"], f"{p['cost']:.6g}", f"{p['balance']:.6g}",
                        f"{p['conservation']:.6g}", prot,
                        encode_config(p["config"], floor)])


def write_random_csv(scheme_csvs, cost, floor, path, seeds=range(20), lead=120):
    """One row per (seed, k-protected) random protection order: seed,k,cost,config. The
    selector filters these to a target cost for the random anticontrol band.
    """
    d_eval = load_distortion_table(scheme_csvs, lead=lead, axes=CONSISTENCY)
    groups = list(d_eval)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "k", "cost", "config"])
        for s in seeds:
            rng = np.random.default_rng(s)
            order = list(rng.permutation(np.array(groups, dtype=object)))
            protected = set()
            for k in range(len(groups) + 1):
                cfg = {g: ("bf16" if g in protected else floor) for g in groups}
                c = sum(cost[g][cfg[g]] for g in cfg)
                w.writerow([s, k, f"{c:.6g}", encode_config(cfg, floor)])
                if k < len(groups):
                    protected.add(order[k])


def _cost_label(cost_kind, floor):
    """X-axis label. The two frontiers share this plotting code but NOT their cost units:
    'flop' is a dimensionless share of quantisable Linear FLOPs (0..1), 'weight' is absolute
    weight bytes. """
    if cost_kind == "flop":
        return f"share of quantisable Linear FLOPs at bf16  ({floor} floor)"
    return f"model weight bytes  ({floor} floor)"


def _panels(phys, rmse, band, floor, title, png_path, cost_label, keys=PANELS):
    """One panel per axis in `keys`. An axis missing from the points is skipped, so this
    still renders a two-panel figure for callers that never scored the RMSE axis."""
    keys = [k for k in keys if phys and k in phys[0]]
    fig, axes = plt.subplots(1, len(keys), figsize=(5.5 * len(keys), 4.2), squeeze=False)
    for ax, key in zip(axes[0], keys):
        for curve in band:                        # random envelope (thin grey)
            if curve and key in curve[0]:
                ax.plot([pt["cost"] for pt in curve], [pt[key] for pt in curve],
                        color="0.8", lw=0.6, zorder=1)
        ax.plot([p["cost"] for p in rmse], [p[key] for p in rmse],
                "s--", color="tab:orange", label="RMSE-guided", zorder=3)
        ax.plot([p["cost"] for p in phys], [p[key] for p in phys],
                "o-", color="tab:blue", label="physics-guided", zorder=4)
        ax.set_xlabel(cost_label)
        ax.set_ylabel(f"{key} distortion (Σ|SVR|)")
        ax.set_title(PANEL_TITLE.get(key, key), fontsize=10)
    axes[0][0].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    plt.close(fig)


# Back-compat alias: this was _two_panel until the RMSE panel was added.
_two_panel = _panels


def run_dev_frontier_W8(lead=120, outdir="."):
    """W8 -> bf16 protection frontier (dev/validation vehicle). floor = W8."""
    os.makedirs(outdir, exist_ok=True)
    scheme_csvs = {"W8": "ablation_analysis/ablations_W8/sensitivity.csv"}
    precisions = ["W8", "bf16"]
    ct = torch.load("cost_tables.pt", weights_only=False)

    d_guide_p, d_eval, ax_p = load_tables(scheme_csvs, "physics", lead=lead)
    d_guide_r, _, ax_r = load_tables(scheme_csvs, "rmse", lead=lead)
    cost = weight_memory_cost(list(d_eval), ct, precisions)   # model-size axis (bytes)

    phys = build_frontier(d_guide_p, d_eval, cost, precisions, ax_p, WG)
    rmse = build_frontier(d_guide_r, d_eval, cost, precisions, ax_r, [(1.0,)])
    band = random_band(d_eval, cost, floor="W8", seeds=range(20))

    score_on_axis(phys, d_guide_r)
    score_on_axis(rmse, d_guide_r)
    for curve in band:
        score_on_axis(curve, d_guide_r)

    _write_frontier_csv(os.path.join(outdir, "frontier_W8_physics.csv"), phys, "W8")
    _write_frontier_csv(os.path.join(outdir, "frontier_W8_rmse.csv"), rmse, "W8")
    _two_panel(phys, rmse, band, "W8",
               f"W8->bf16 dev frontier @ {lead}h (weight-memory axis)",
               os.path.join(outdir, "frontier_W8_dev.png"),
               _cost_label("weight", "W8"))

    sweep = b2_sweep(scheme_csvs, cost, precisions, lead=lead)
    with open(os.path.join(outdir, "protected_set_W8.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "protected_order"])
        for name, order in sweep.items():
            w.writerow([name, "|".join(order)])

    return {"physics": phys, "rmse": rmse, "band": band, "b2": sweep}


def run_frontier(scheme_csvs, precisions, floor, cost_kind, tag, lead=120, outdir=".",
                 min_effect_frac=0.0):
    """ Logic to run and create csv/png for the frontiers.
    Args:
        scheme_csvs [dict]          : {precision: sensitivity.csv path}.
        precisions [list[str]]      : candidates. precisions[0] is the fully quantised run
        floor [str]                 : unprotected precision, used for the cost table and CSV encoding
        cost_kind [str]             : "flop" (Frontier B, protected-FLOP-fraction) or "weight"
                                    (Frontier A, model-weight-bytes)
        tag [str]                   : filename stem, e.g. "B_W8A8"
        lead [int]                  : lead time in hours
        outdir [str]                : created if absent
        min_effect_frac [float]     : removes points if damage of group below this*max SVR
    Returns:
        dict: {"physics", "rmse", "band", "b2"}. Writes frontier_<tag>_{physics,rmse}.csv,
            guide_order_<tag>_rmse.csv, protected_set_<tag>.csv and frontier_<tag>.png.
    """
    os.makedirs(outdir, exist_ok=True)
    ct = torch.load("cost_tables.pt", weights_only=False)
    d_guide_p, d_eval, ax_p = load_tables(scheme_csvs, "physics", lead=lead,
                                          min_effect_frac=min_effect_frac)
    d_guide_r, _, ax_r = load_tables(scheme_csvs, "rmse", lead=lead,
                                     min_effect_frac=min_effect_frac)
    groups = list(d_eval)
    if cost_kind == "flop":
        cost = flop_cost(groups, ct, floor=floor)
    else:
        cost = weight_memory_cost(groups, ct, precisions)

    phys = build_frontier(d_guide_p, d_eval, cost, precisions, ax_p, WG)
    rmse = build_frontier(d_guide_r, d_eval, cost, precisions, ax_r, [(1.0,)])
    band = random_band(d_eval, cost, floor=floor, seeds=range(20))

    score_on_axis(phys, d_guide_r)
    score_on_axis(rmse, d_guide_r)
    for curve in band:
        score_on_axis(curve, d_guide_r)

    _write_frontier_csv(os.path.join(outdir, f"frontier_{tag}_physics.csv"), phys, floor)
    _write_frontier_csv(os.path.join(outdir, f"frontier_{tag}_rmse.csv"), rmse, floor)

    # unfiltered RMSE budget sequence, for exact-cost matching
    rmse_order = guide_budget_frontier(d_guide_r, d_eval, cost, precisions, ax_r,
                                       [(1.0,)], floor=floor)
    write_guide_order_csv(os.path.join(outdir, f"guide_order_{tag}_rmse.csv"),
                          rmse_order, floor)

    _two_panel(phys, rmse, band, floor, f"{tag} @ {lead}h",
               os.path.join(outdir, f"frontier_{tag}.png"),
               _cost_label(cost_kind, floor))

    sweep = b2_sweep(scheme_csvs, cost, precisions, lead=lead)
    with open(os.path.join(outdir, f"protected_set_{tag}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "protected_order"])
        for name, order in sweep.items():
            w.writerow([name, "|".join(order)])
    return {"physics": phys, "rmse": rmse, "band": band, "b2": sweep}


def run_frontier_B(scheme="W8A8", lead=120, outdir=".", min_effect_frac=0.0):
    """Frontier B: {scheme, bf16}, floor=scheme, protected-FLOP-fraction axis (the
    deployment/headline frontier). scheme='W8A8_sq' gives the SmoothQuant overlay."""
    csv_path = f"ablation_analysis/ablations_{scheme}/sensitivity.csv"
    return run_frontier({scheme: csv_path}, [scheme, "bf16"], scheme, "flop",
                        f"B_{scheme}", lead, outdir, min_effect_frac)


def run_frontier_A(w4_csv="ablation_analysis/ablations_W4/sensitivity.csv",
                   w8_csv="ablation_analysis/ablations_W8/sensitivity.csv",
                   lead=120, outdir=".", min_effect_frac=0.0):
    """Frontier A: {W4, W8, bf16}, floor=W4, model-weight-bytes axis (the compression /
    model-size scientific frontier)."""
    return run_frontier({"W4": w4_csv, "W8": w8_csv}, ["W4", "W8", "bf16"], "W4",
                        "weight", "A_W4W8", lead, outdir, min_effect_frac)


def main():
    res = run_dev_frontier_W8(lead=120, outdir=".")
    print(f"physics frontier: {len(res['physics'])} points; "
          f"B2 variants: {len(res['b2'])}")
    for name, order in res["b2"].items():
        print(f"  {name:24s} -> {order[:4]}")


if __name__ == "__main__":
    main()
