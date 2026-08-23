# allocator.py
from collections import OrderedDict

import numpy as np

CONSISTENCY_AXES = ("balance", "conservation")


def combined_distortion(d_axis, weights):
    """Weighted sum over axes of a per-config or per-(group,precision) distortion dict."""
    return float(sum(weights[a] * d_axis.get(a, 0.0) for a in weights))


def config_distortion(d, config):
    """Vector distortion of a full config: {axis: Σ_g d[g][p_g][axis]}."""
    out = {a: 0.0 for a in CONSISTENCY_AXES}
    for g, p in config.items():
        for a in CONSISTENCY_AXES:
            out[a] += d[g][p][a]
    return out


def worst_axis(d_config):
    return float(max(d_config[a] for a in CONSISTENCY_AXES))


def allocate_lagrangian(d, cost, weights, lam, precisions):
    """Per-group independent argmin of combined_distortion + lam*cost. Exact on the
    convex hull; groups are independent under additivity."""
    config = OrderedDict()
    for g in d:
        best_p, best_obj = None, float("inf")
        for p in precisions:
            obj = combined_distortion(d[g][p], weights) + lam * cost[g][p]
            if obj < best_obj:
                best_obj, best_p = obj, p
        config[g] = best_p
    return dict(config)


def _pareto(points):
    """Keep points not dominated on (cost up-good? no: lower cost + lower balance +
    lower conservation all better). A point is dominated if another is <= on all
    three and < on at least one."""
    keep = []
    for p in points:
        dominated = False
        for q in points:
            if q is p:
                continue
            if (q["cost"] <= p["cost"] and q["balance"] <= p["balance"]
                    and q["conservation"] <= p["conservation"]
                    and (q["cost"] < p["cost"] or q["balance"] < p["balance"]
                         or q["conservation"] < p["conservation"])):
                dominated = True
                break
        if not dominated:
            keep.append(p)
    return keep


def propose_frontier(d, cost, precisions, weight_grid, lam_grid):
    """Sweep (axis-weights x lambda), emit unique Pareto configs with vector distortion."""
    seen, points = set(), []
    for wb, wc in weight_grid:
        weights = {"balance": wb, "conservation": wc}
        for lam in lam_grid:
            cfg = allocate_lagrangian(d, cost, weights, lam, precisions)
            key = tuple(sorted(cfg.items()))
            if key in seen:
                continue
            seen.add(key)
            dc = config_distortion(d, cfg)
            points.append({"config": cfg,
                           "cost": float(sum(cost[g][p] for g, p in cfg.items())),
                           "balance": dc["balance"], "conservation": dc["conservation"]})
    return _pareto(points)


import csv as _csv
from collections import defaultdict


def load_distortion_table(scheme_csvs, lead=120, axes=CONSISTENCY_AXES,
                          reducer=None, families=None, min_effect_frac=0.0,
                          axis_floor=0.0):
    import ablation_comp as ac
    from plot_common import AGG_CLASS
    d = defaultdict(lambda: defaultdict(dict))
    groups = set()
    for precision, path in scheme_csvs.items():
        acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        with open(path, newline="") as fh:
            for row in _csv.DictReader(fh):
                if int(float(row["lead"])) != lead:
                    continue
                fam = _family_of_metric(row["metric"])
                axis = AGG_CLASS.get(fam)
                if axis not in axes:
                    continue
                acc[row["group"]][axis][fam].append(float(row["distortion"]))
        for g, axesd in acc.items():
            groups.add(g)
            for a in axes:
                fam_vals = {}
                for f, v in axesd.get(a, {}).items():
                    if families and a in families and f not in families[a]:
                        continue
                    finite = [x for x in v if np.isfinite(x)]
                    fam_vals[f] = sum(finite) / len(finite) if finite else float("nan")
                d[g][precision][a] = ac._reduce_axis(fam_vals, a, reducer) if fam_vals else 0.0
    if min_effect_frac > 0.0 and axis_floor > 0.0:
        raise ValueError(
            "min_effect_frac and axis_floor are mutually exclusive -- applying both makes "
            "the effective threshold impossible to state. Use axis_floor for selection; "
            "min_effect_frac only to reproduce the pre-2026-07-31 frontier.")
    if axis_floor > 0.0:
        from axis_noise_floor import apply_absolute_floor
        apply_absolute_floor(d, axes=list(axes), floor=axis_floor)
    elif min_effect_frac > 0.0:
        from axis_noise_floor import apply_floor
        apply_floor(d, axes=list(axes), frac=min_effect_frac)
    for g in groups:                     # bf16 = protected reference
        d[g]["bf16"] = {a: 0.0 for a in axes}
    return {g: dict(p) for g, p in d.items()}


def _family_of_metric(label):
    """Map a sensitivity.csv metric label back to its registry family key."""
    if label.startswith("RMSE"):
        return "RMSE"
    if label.startswith("LapseW1"):
        return "lapse_rate"
    if label.startswith(("Vag/Vg", "|Vag|")):
        return "wind_balance"
    if label.startswith("div/vort"):
        return "div_vort"
    if label.startswith("HypsRel"):
        return "hypsometric"
    if label.startswith("|DryAir"):
        return "dry_air_mass"
    if label.startswith(("neg-q", "|neg-q")):
        return "neg_humidity"
    return ""
