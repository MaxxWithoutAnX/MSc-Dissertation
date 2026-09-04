""" Allocation of precision to layer groups. Given a (group, precision) distortion table and cost, creates and sweeps 
    lambda and returns configurations that are optimal given the ablation analysis results.
"""
from collections import OrderedDict, defaultdict
import numpy as np
import csv as _csv

CONSISTENCY_AXES = ("balance", "conservation")


def combined_distortion(d_axis, weights):
    """Scalarise a distortion vector: sum_a weights[a] * d_axis[a]
    Args:
        d_axis [dict]   : {axis[str]: distortion[float]}. 
        weights [dict] : {axis [str]: weight[float]}
    Returns:
        float: Weighted sum
        """
    return float(sum(weights[a] * d_axis.get(a, 0.0) for a in weights))


def config_distortion(d, config):
    """Vector distortion of a full config: {axis: Σ_g d[g][p_g][axis]}.
    Args:
        d [dict]        : distortion table d[group][precision][axis]
        config [dict]   : {group: precision}, one precision per group
    Returns:
        dict : {axis: distortion}. 
    """
    out = {a: 0.0 for a in CONSISTENCY_AXES}
    for g, p in config.items():
        for a in CONSISTENCY_AXES:
            out[a] += d[g][p][a]
    return out


def worst_axis(d_config):
    """ Maximum value of distortion.
    Args:
        d_config [dict] : {axis:distortion} for one config
    Returns:
        float : maximum value of the axes.
    """
    return float(max(d_config[a] for a in CONSISTENCY_AXES))


def allocate_lagrangian(d, cost, weights, lam, precisions):
    """Chooses one precision per group by argmin of  combined_distortion + lam*cost.
    Args:
        d [dict]                : distortion table d[group][precision][axis].
        cost [dict]             : cost[group][precision], any scalar cost in units matched to lam.
        weights [dict]          : {axis: weight} passed to combined_distortion.
        lam [float]             : cost multiplier. 0 = ignore cost (pick the least damaging precision everywhere). large = ignore distortion (pick the cheapest).
        precisions [iterable]   : precisions to allocate per group.
    Returns:
        dict: {group: precision}. Every group in d and the precision it should be allocated
    """
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
    """ Pareto filter over the three minimised objectives: cost, balance, conservation.
    Args:
        points [list[dict]] : each with "cost", "balance", "conservation" keys.
    Returns:
        list[dict]          : the input dicts (same objects, input order) that survive.
    """
    keep = []
    for p in points:
        dominated = False
        for q in points:
            if q is p:
                continue
            # AI helped me with this
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
    """Sweep (axis-weights x lambda), emit unique Pareto configs with vector distortion.
    Args:
        d [dict]: distortion table d[group][precision][axis]
        cost [dict]: cost[group][precision]
        precisions [iterable]: candidate precisions per group
        weight_grid [iterable[tuple]]: (balance_weight, conservation_weight) pairs
        lam_grid [iterable[float]]: cost multipliers to sweep
    Returns:
        list[dict]: Pareto-optimal points, each {"config", "cost", "balance", "conservation"}. Configs recurring across the sweep are emitted once.
    """
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



def load_distortion_table(scheme_csvs, lead=120, axes=CONSISTENCY_AXES,
                          reducer=None, families=None, min_effect_frac=0.0, value_col="distortion"):
    """ Build the distortion table d[group][precision][axis] from OAT sensitivity CSVs.
    Args:
        scheme_csvs [dict]      : {precision: path to that scheme's sensitivity.csv}
        lead [int]              : lead time in hours
        axes [tuple]            : metric axes
        reducer [str | None]    : how to collapse families within an axis to a single value
        families [dict | None]  : optional {axis: set(family)} restricting the reduction 
                                  to a family subset. Used if wanting to exclude certain metrics from the axis.
        min_effect_frac [float] : zero any axis value below this fraction of that axis's maximum over groups WITHIN its own
                                 precision. Only used on Aurora.
        value_col [str]         : which per-metric column to reduce. "distortion" (default) is
                                sigma-gated, max(0,|delta|-sigma)/IQR; "svr" is the same quantity with
                                sigma=0. 
    Returns:
        dict: {group: {precision: {axis: value}}} where value is the measured damage at that group, precision, axis
    """
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
                acc[row["group"]][axis][fam].append(
                    abs(float(row["svr"])) if value_col == "svr"
                    else float(row[value_col]))
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
    if min_effect_frac > 0.0:
        from axis_noise_floor import apply_floor
        apply_floor(d, axes=list(axes), frac=min_effect_frac)
    for g in groups:                     # bf16 = protected reference. FP32 for Stormer, don't want to change it as may risk breaking things.
        d[g]["bf16"] = {a: 0.0 for a in axes}
    return {g: dict(p) for g, p in d.items()}


def _family_of_metric(label):
    """Map sensitivity.csv metric label back to its registry family key.
    Args:
        label [str]: label associated with metrics
    Returns:
        str    
    """
    if label.startswith("RMSE"):
        return "RMSE"
    if label.startswith(("Vag/Vg", "|Vag|")):
        return "wind_balance"
    if label.startswith("HypsRel"):
        return "hypsometric"
    if label.startswith("|DryAir"):
        return "dry_air_mass"
    if label.startswith(("neg-q", "|neg-q")):
        return "neg_humidity"
    return ""
