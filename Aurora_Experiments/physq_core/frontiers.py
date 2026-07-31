# frontiers.py
import numpy as np

from cost_tables import PRECISION_BITS

CONSISTENCY = ("balance", "conservation")


def flop_cost(groups, ct, floor="W8A8"):
    """Frontier B proxy: protecting a group to bf16 costs its FLOP fraction; the
    int8/W8A8 floor runs free."""
    total = sum(ct[g]["flops"] for g in groups) or 1.0   # guard all-zero/empty (cf. int8_flop_fraction)
    return {g: {floor: 0.0, "bf16": ct[g]["flops"] / total} for g in groups}


def weight_memory_cost(groups, ct, precisions):
    """Frontier A: weight bytes per group at each precision (sum -> model size; /1e9 = GB)."""
    return {g: {p: ct[g]["params"] * PRECISION_BITS[p] / 8.0 for p in precisions}
            for g in groups}


def measured_cost(profile):
    """Pluggable measured cost: profile[g][p] -> measured peak VRAM / latency. Copied
    so callers can mutate the result without touching the caller's dict."""
    return {g: dict(pv) for g, pv in profile.items()}


from allocator import allocate_lagrangian, load_distortion_table


def _sweep_configs(d_guide, cost, precisions, guide_axes, weight_grid, lam_grid):
    """Unique configs from the (axis-weights x lambda) Lagrangian sweep on the guide."""
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
    return {a: sum(d[g][cfg[g]][a] for g in cfg) for a in axes}


def _cost_sum(cost, cfg):
    return sum(cost[g][cfg[g]] for g in cfg)


def _pareto(points, key):
    """Keep points not dominated under `key` (a tuple where lower is better on every
    component). p dominated if some q is <= on all and < on at least one."""
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
                min_effect_frac=0.0, axis_floor=0.0):
    d_eval = load_distortion_table(scheme_csvs, lead=lead, axes=CONSISTENCY,
                                   reducer=reducer, families=families,
                                   min_effect_frac=min_effect_frac, axis_floor=axis_floor)
    if guided_by == "physics":
        return d_eval, d_eval, CONSISTENCY
    if guided_by == "rmse":
        d_guide = load_distortion_table(scheme_csvs, lead=lead, axes=("standard",),
                                        min_effect_frac=min_effect_frac,
                                        axis_floor=axis_floor)
        return d_guide, d_eval, ("standard",)
    raise ValueError(f"unknown guided_by {guided_by!r}")


def random_frontier(d_eval, cost, floor, seed):
    """Protect groups in a random order (floor -> bf16); one point per #protected k."""
    rng = np.random.default_rng(seed)
    groups = list(d_eval)
    order = list(rng.permutation(np.array(groups, dtype=object)))
    pts, protected = [], set()
    for k in range(len(groups) + 1):
        cfg = {g: ("bf16" if g in protected else floor) for g in groups}
        e = _axis_sum(d_eval, cfg, CONSISTENCY)
        pts.append({"k": k, "cost": _cost_sum(cost, cfg),
                    "balance": e["balance"], "conservation": e["conservation"]})
        if k < len(groups):
            protected.add(order[k])
    return pts


def random_band(d_eval, cost, floor, seeds=range(20)):
    """One random_frontier per seed -> list of curves; the driver renders the envelope."""
    return [random_frontier(d_eval, cost, floor, s) for s in seeds]


def protected_order(d_eval, cost, precisions, weight_grid, lam_grid):
    """Order in which groups first flip to bf16 along the cost-sorted physics frontier."""
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
    if lam_grid is None:
        lam_grid = auto_lambda_grid(d_guide, cost, precisions, guide_axes)
    floor = floor or precisions[0]
    cfgs = _sweep_configs(d_guide, cost, precisions, guide_axes, weight_grid, lam_grid)

    seen = {tuple(sorted(c.items())) for c in cfgs}
    for uniform in ({g: floor for g in d_guide}, {g: "bf16" for g in d_guide}):
        key = tuple(sorted(uniform.items()))
        if key not in seen:                     # the equivalence floor can make all-bf16
            seen.add(key)                       # unreachable by the sweep (ties go to floor)
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
    "mean|wind+lapse":         dict(reducer="mean", families={"balance": {"wind_balance", "lapse_rate"}}),
    "mean|wind+both":          dict(reducer="mean", families={"balance": {"wind_balance", "hypsometric", "lapse_rate"}}),
    "first_pc|wind+both":      dict(reducer="first_pc", families={"balance": {"wind_balance", "hypsometric", "lapse_rate"}}),
}


def b2_sweep(scheme_csvs, cost, precisions, lead=120,
             weight_grid=None, lam_grid=None, min_effect_frac=0.0, axis_floor=0.0):
    weight_grid = weight_grid or [(1.0, 1.0), (1.0, 0.0), (0.0, 1.0)]
    out = {}
    for name, kw in B2_VARIANTS.items():
        d_eval = load_distortion_table(scheme_csvs, lead=lead, axes=CONSISTENCY,
                                       min_effect_frac=min_effect_frac,
                                       axis_floor=axis_floor, **kw)
        out[name] = protected_order(d_eval, cost, precisions, weight_grid, lam_grid)
    return out
