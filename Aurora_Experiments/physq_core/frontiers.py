""" Predicted mixed precision frontier functions made over the ablation analysis. Note that bf16 is used here to represent full precision
This is wrong and should be FP32 and was done as an error due to misunderstanding of how autocast worked.
Strings have not been changed to avoid breaking anythin but results are as if FP32. Any orderings/ranking would not be affected by this
as everything would just scale by a factor of 2.
"""
import numpy as np

from cost_tables import STORAGE_BITS

CONSISTENCY = ("balance", "conservation")


def flop_cost(groups, ct, floor="W8A8"):
    """Frontier B proxy calculating how much it costs to quantise a group
    Args:
        groups [iterable]   : layer-group names
        ct [dict]           : cost_tables.pt, ct[group]["params"]
    Returns:
        dict: cost[group][precision] -> FLOPs
    """
    total = sum(ct[g]["flops"] for g in groups) or 1.0   # guard all-zero/empty 
    return {g: {floor: 0.0, "bf16": ct[g]["flops"] / total} for g in groups}


def weight_memory_cost(groups, ct, precisions):
    """Frontier A: weight bytes per group at each precision.
    Args:
        groups [iterable]       : layer-group names
        ct [dict]               : cost_tables.pt, ct[group]["params"]
        precisions [iterable]   : precisions to price
    Returns:
        dict: cost[group][precision] -> bytes
    """
    return {g: {p: ct[g]["params"] * STORAGE_BITS[p] / 8.0 for p in precisions}
            for g in groups}


def measured_cost(profile):
    """Pluggable measured cost: profile[g][p] -> measured peak VRAM / latency. Copied
    so callers can mutate the result without touching the caller's dict."""
    return {g: dict(pv) for g, pv in profile.items()}


from allocator import allocate_lagrangian, load_distortion_table


def _sweep_configs(d_guide, cost, precisions, guide_axes, weight_grid, lam_grid):
    """Unique configs from the (axis-weights x lambda) Lagrangian sweep on the guide.
    Args:
        d_guide [dict]                  : distortion table d[group][precision][axis]
        cost [dict]                     : cost[group][precision]
        precisions [iterable]           : candidate precisions per group
        guide_axes [tuple]              : axes being optimised
        weight_grid [iterable[tuple]]   : axis-weight vectors
        lam_grid [iterable[float]]      : cost multipliers from auto_lambda_grid
    Returns:
        list[dict]: {group: precision} configs in first-seen order 
    """
    seen, configs = set(), []
    for w in weight_grid:
        assert len(w) == len(guide_axes), f"weight tuple {w} arity != guide_axes {guide_axes}"
        weights = dict(zip(guide_axes, w))
        for lam in lam_grid:
            cfg = allocate_lagrangian(d_guide, cost, weights, lam, precisions)
            key = tuple(sorted(cfg.items()))
            if key not in seen:
                seen.add(key)
                configs.append(cfg)
    return configs


def _axis_sum(d, cfg, axes):
    """ Additive surrogate total per axis
    Args:
        d [dict]        : distortion table d[group][precision][axis]
        cfg [dict]      : {group: precision}
        axes [iterable] : axes to total
    Returns:
        dict: {axis: float} in the table's SVR units.
    """
    return {a: sum(d[g][cfg[g]][a] for g in cfg) for a in axes}


def _cost_sum(cost, cfg):
    """ Total cost of a config
    Args:
        cost [dict] : cost[group][precision], units set by the cost function used (FLOP fraction or weight bytes)
        cfg [dict]  : {group: precision}
    Returns:
        float : total in same units as cost
    """
    return sum(cost[g][cfg[g]] for g in cfg)


def _pareto(points, key):
    """Filters out points lower on (cost, disotrtion) than all other points.
    Args:
        points [list[dict]] : candidate points
        key [callable]      : point -> tuple of minimised objectives.
    Returns:
        list[dict]: the surviving input objects which are all predictions at their cost.
    """
    keep = []
    for p in points:
        kp = key(p)
        dominated = any(
            q is not p
            and all(a <= b for a, b in zip(key(q), kp))
            and any(a < b for a, b in zip(key(q), kp))
            for q in points)
        if not dominated:
            keep.append(p)
    return keep


def auto_lambda_grid(d_guide, cost, precisions, guide_axes, n=80, margin=10.0):
    """ Log spaced grid spanning the distrtion/delta cost crossovers. Used so that it works on both FLOPs and weight bytes
    which are different orders of magnitude.
    Args:
        d_guide [dict]          : distortion table d[group][precision][axis]
        cost [dict]             : cost[group][precision]
        precisions [iterable]   : candidate precisions
        guide_axes [tuple]      : axes the guide optimises
        n [int]                 : number of grid points
        margin [float]          : Adds margin to both ends for any configs near extremes.
    Returns:
        list[float]: lambdas, ascending
    """
    ratios = []
    for g in d_guide:
        for p1 in precisions:
            for p2 in precisions:
                dc = cost[g][p2] - cost[g][p1]
                if dc <= 0:
                    continue
                deltas = [d_guide[g][p1][a] - d_guide[g][p2][a] for a in guide_axes]
                for dd in (sum(deltas), *deltas):
                    if dd > 0:
                        ratios.append(dd / dc)
    if not ratios:
        return [1.0]
    return list(np.geomspace(min(ratios) / margin, max(ratios) * margin, n))


def build_frontier(d_guide, d_eval, cost, precisions, guide_axes,
                   weight_grid, lam_grid=None):
    """ Builds the frontiers
    Args:
        d_guide [dict]                  : table the allocation optimises
        d_eval [dict]                   : table the points are scored on
        cost [dict]                     : cost[group][precision]
        precisions [iterable]           : candidate precision. Precisions[0] is the uniform endpoint.
        guide_axes [tuple]              : axes of d_guide
        weight_grid [iterable[tuple]]   : axis-weight vectors to sweep
        lam_grid [iterable | None]      : None auto-derives via auto_lambda_grid.
    Returns:
        list[dict]: cost-sorted points, each dict with keys {"config", "cost", "balance",
            "conservation", "guide"}.
    """
    if lam_grid is None:
        lam_grid = auto_lambda_grid(d_guide, cost, precisions, guide_axes)

    def _point(cfg):
        e = _axis_sum(d_eval, cfg, CONSISTENCY)
        return {"config": cfg, "cost": _cost_sum(cost, cfg),
                "balance": e["balance"], "conservation": e["conservation"],
                "guide": _axis_sum(d_guide, cfg, guide_axes)}

    pts = [_point(cfg) for cfg in _sweep_configs(d_guide, cost, precisions, guide_axes,
                                                 weight_grid, lam_grid)]
    kept = _pareto(pts, lambda p: (p["cost"],) + tuple(p["guide"][a] for a in guide_axes))

    seen = {tuple(sorted(p["config"].items())) for p in kept}
    for uniform in ({g: precisions[0] for g in d_guide}, {g: "bf16" for g in d_guide}):
        key = tuple(sorted(uniform.items()))
        if key not in seen:
            seen.add(key)
            kept.append(_point(uniform))
    return sorted(kept, key=lambda p: p["cost"])


def load_tables(scheme_csvs, guided_by, lead=120, reducer=None, families=None,
                min_effect_frac=0.0):
    """ Builds guide and eval tables for a single frontier
    Args:
        scheme_csvs [dict]  : {precision: sensitivity.csv path}
        guided_by [str]     : "physics" or "rmse"
    Returns:
        tuple: (d_guide, d_eval, guide_axes)
    """
    d_eval = load_distortion_table(scheme_csvs, lead=lead, axes=CONSISTENCY,
                                   reducer=reducer, families=families,
                                   min_effect_frac=min_effect_frac)
    if guided_by == "physics":
        return d_eval, d_eval, CONSISTENCY
    if guided_by == "rmse":
        d_guide = load_distortion_table(scheme_csvs, lead=lead, axes=("standard",),
                                        min_effect_frac=min_effect_frac)
        return d_guide, d_eval, ("standard",)
    raise ValueError(f"unknown guided_by {guided_by!r}")


def random_frontier(d_eval, cost, floor, seed):
    """Protect groups in a random order as a control
    Args:
        d_eval [dict], cost [dict], floor [str] : the unprotected precision.
        seed [int]                              : numpy default_rng seed.
    Returns:
        list[dict]: len(groups)+1 points, k ascending, each {"k","cost","config",
            "balance","conservation"}.
    """
    rng = np.random.default_rng(seed)
    groups = list(d_eval)
    order = list(rng.permutation(np.array(groups, dtype=object)))
    pts, protected = [], set()
    for k in range(len(groups) + 1):
        cfg = {g: ("bf16" if g in protected else floor) for g in groups}
        e = _axis_sum(d_eval, cfg, CONSISTENCY)
        pts.append({"k": k, "cost": _cost_sum(cost, cfg), "config": dict(cfg),
                    "balance": e["balance"], "conservation": e["conservation"]})
        if k < len(groups):
            protected.add(order[k])
    return pts


def random_band(d_eval, cost, floor, seeds=range(20)):
    """One random_frontier per seed -> list of curves; the driver renders the envelope."""
    return [random_frontier(d_eval, cost, floor, s) for s in seeds]


def protected_order(d_eval, cost, precisions, weight_grid, lam_grid):
    """Order in which groups first flip to bf16 along the cost-sorted physics frontier.
    Returns:
        list[str]: groups in first protected order
    """
    fr = build_frontier(d_eval, d_eval, cost, precisions, CONSISTENCY, weight_grid, lam_grid)
    seen, order = set(), []
    for p in fr:
        for g, pr in p["config"].items():
            if pr == "bf16" and g not in seen:
                seen.add(g)
                order.append(g)
    return order


def guide_budget_frontier(d_guide, d_eval, cost, precisions, guide_axes,
                          weight_grid, lam_grid=None, floor=None):
    """ Every unique config the sweep produces, scord on d_eval.
    Args:
        d_guide [dict]: table the allocation optimises
        d_eval [dict]: table the points are scored
        cost [dict]: cost[group][precision].
        precisions [iterable]: candidate precisions
        guide_axes [tuple]: axes of d_guide
        weight_grid [iterable[tuple]]: axis-weight vectors to sweep
        lam_grid [iterable | None]: None auto-derives via auto_lambda_grid.
        floor [str | None]: unprotected precision for the uniform endpoint
    Returns:
        list[dict]: cost-sorted, each {"k", "cost", "config", "balance", "conservation"}
    """
    if lam_grid is None:
        lam_grid = auto_lambda_grid(d_guide, cost, precisions, guide_axes)
    floor = floor or precisions[0]
    cfgs = _sweep_configs(d_guide, cost, precisions, guide_axes, weight_grid, lam_grid)

    seen = {tuple(sorted(c.items())) for c in cfgs}
    for uniform in ({g: floor for g in d_guide}, {g: "bf16" for g in d_guide}):
        key = tuple(sorted(uniform.items()))
        if key not in seen:                     
            seen.add(key)                       
            cfgs.append(uniform)

    pts = []
    for cfg in cfgs:
        e = _axis_sum(d_eval, cfg, CONSISTENCY)
        pts.append({"k": sum(1 for pr in cfg.values() if pr == "bf16"),
                    "cost": _cost_sum(cost, cfg), "config": cfg,
                    "balance": e["balance"], "conservation": e["conservation"]})
    return sorted(pts, key=lambda p: p["cost"])


B2_VARIANTS = {
    "representative|wind":     dict(reducer="representative", families=None),
    "mean|all":                dict(reducer="mean", families=None),
    "mean|wind":               dict(reducer="mean", families={"balance": {"wind_balance"}}),
    "mean|wind+hyps":          dict(reducer="mean", families={"balance": {"wind_balance", "hypsometric"}}),
}


def b2_sweep(scheme_csvs, cost, precisions, lead=120,
             weight_grid=None, lam_grid=None, min_effect_frac=0.0):
    """ Sweeps all reducder variants to check robustness.
    Returns:
        dict : {variant_name: [group]}. Lists ranking order of all groups under the different reducer variants
    """
    weight_grid = weight_grid or [(1.0, 1.0), (1.0, 0.0), (0.0, 1.0)]
    out = {}
    for name, kw in B2_VARIANTS.items():
        d_eval = load_distortion_table(scheme_csvs, lead=lead, axes=CONSISTENCY,
                                       min_effect_frac=min_effect_frac, **kw)
        out[name] = protected_order(d_eval, cost, precisions, weight_grid, lam_grid)
    return out
