# axis_noise_floor.py
import os

import numpy as np

DEFAULT_FRAC = 0.01           # fraction of the per-(axis, precision) maximum

FILM_BALANCE_W8 = 0.03066         # G1: must be erased. Per-precision threshold is 0.05331.
DECODER_HEADS_STANDARD_W8 = 0.09295   # G2: must survive. It IS the W8 standard maximum.
COND_EMBED_BALANCE_W8 = 0.37931       # G3: real signal the old global rule erased.

MEASURED_AXIS_NULL_P95 = {"balance": 0.0036, "conservation": 0.0042, "standard": 0.0061}

FLOOR_PCTL = 95.0


def _iter_cells(d, axes):
    """(group, precision, axis) for every finite cell present."""
    for g in d:
        for p in d[g]:
            for a in axes:
                if a in d[g][p] and np.isfinite(d[g][p][a]):
                    yield g, p, a


def apply_floor(d, axes=None, frac=DEFAULT_FRAC):
    if axes is None:
        axes = sorted({a for g in d for p in d[g] for a in d[g][p]})
    if frac <= 0.0:
        return d
    return _relative_per_precision(d, list(axes), frac)


def apply_absolute_floor(d, axes=None, floor=0.0):
    if axes is None:
        axes = sorted({a for g in d for p in d[g] for a in d[g][p]})
    if floor <= 0.0:
        return d
    for g, p, a in list(_iter_cells(d, axes)):
        if d[g][p][a] < floor:
            d[g][p][a] = 0.0
    return d


def _relative_per_precision(d, axes, frac):
    precisions = {p for g in d for p in d[g]}
    for a in axes:
        for p in precisions:
            finite = [d[g][p][a] for g in d
                      if p in d[g] and a in d[g][p] and np.isfinite(d[g][p][a])]
            if not finite:
                continue
            thresh = frac * max(finite)
            for g in d:
                if p in d[g] and a in d[g][p] and d[g][p][a] < thresh:
                    d[g][p][a] = 0.0
    return d


def _relative_global_DEPRECATED(d, axes, frac):
    for a in axes:
        finite = [d[g][p][a] for g, p, _ in _iter_cells(d, [a])]
        if not finite:
            continue
        thresh = frac * max(finite)
        for g, p, aa in list(_iter_cells(d, [a])):
            if d[g][p][a] < thresh:
                d[g][p][a] = 0.0
    return d


def axis_floors(members_dir="noise_floor_ens", lead=120, axes=None):
    if not os.path.isdir(members_dir):
        return {}
    import torch

    from plot_common import MetricsRun

    paths = sorted(p for p in os.listdir(members_dir) if p.endswith(".pt"))
    if len(paths) < 3:
        return {}
    ref = MetricsRun("ref", torch.load(os.path.join(members_dir, paths[0]),
                                      map_location="cpu", weights_only=False))
    per_axis = {a: [] for a in (axes or [])}
    for p in paths[1:]:
        mem = MetricsRun(p, torch.load(os.path.join(members_dir, p),
                                       map_location="cpu", weights_only=False))
        for a in per_axis:
            per_axis[a].append(_axis_value_vs_reference(ref, mem, a, lead))
    return {a: float(np.nanpercentile(v, FLOOR_PCTL)) for a, v in per_axis.items() if v}


def _axis_value_vs_reference(ref, member, axis, lead):
    """Axis value of one ensemble member against the unperturbed reference, through the SAME
    reduction the allocator uses. Mirrors paired_stats.axis_value."""
    import paired_stats as ps

    specs = ps.specs_for_family(ref, member, axis, lead)
    if not specs:
        return float("nan")
    deltas, denoms = ps.family_deltas(ref, member, specs, lead, None, "iqr_raw")
    idx = np.arange(len(next(iter(deltas.values()))))
    return ps.axis_value(deltas, denoms, idx)
