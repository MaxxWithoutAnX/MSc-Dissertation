"""Determines whether the distortions calculated are large enough they are worth protecting.
"""
import numpy as np

DEFAULT_FRAC = 0.01           # fraction of the per-(axis, precision) maximum. Arbitrarily chosen gate where if distortion is less than 1% of maximum, group can be ignored


def _iter_cells(d, axes):
    """Creates tuple (group, precision, axis) for every finite cell present.
    Args:
        d [dict]       : distortion table d[group][precision][axis]
        axes [iterable]: axes to visit; cells for other axes are skipped
    Returns:
        tuple: (group, precision, axis)
    """
    for g in d:
        for p in d[g]:
            for a in axes:
                if a in d[g][p] and np.isfinite(d[g][p][a]):
                    return g, p, a


def apply_floor(d, axes=None, frac=DEFAULT_FRAC):
    """ Removes points thresholded at frac * maximum SVR over groups
    Args:
        d [dict]                : distortion table d[group][precision][axis]
        axes [iterable | None]  : axes to gate
        frac [float]            : fraction of the per-(axis, precision) maximum.
    Returns:
        dict: the same object `d`, mutated
    """
    if axes is None:
        axes = sorted({a for g in d for p in d[g] for a in d[g][p]})
    if frac <= 0.0:
        return d
    return _relative_per_precision(d, list(axes), frac)


def _relative_per_precision(d, axes, frac):
    """Implementation of apply_floor() per axis, precision
    """
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


