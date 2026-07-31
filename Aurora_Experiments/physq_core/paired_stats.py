# paired_stats.py
import numpy as np

import ablation_comp as ac
import plot_common as pc


def specs_for_family(fp32_run, config_run, family, lead):
    """The metric specs in one registry family that both runs can supply at `lead`."""
    specs = ac.available_specs(fp32_run, config_run, ac.metric_registry(fp32_run), lead)
    return [s for s in specs if s.group == family]


def family_deltas(fp32_run, config_run, specs, lead, dates=None,
                  spread_method="iqr_raw"):
    deltas, denoms = {}, {}
    for s in specs:
        f = np.asarray(pc.per_init_series(fp32_run, s, lead), dtype=np.float64)
        q = np.asarray(pc.per_init_series(config_run, s, lead), dtype=np.float64)
        deltas[s.label] = q - f
        if spread_method == "iqr_raw":
            q75, q25 = np.percentile(f, [75, 25])
            denoms[s.label] = float(q75 - q25)
        else:
            denoms[s.label] = float(pc.spread(f, dates, spread_method))
    return deltas, denoms


def axis_value(deltas, denoms, idx):
    vals = [abs(float(deltas[l][idx].mean())) / denoms[l]
            for l in deltas if denoms.get(l, 0.0) > 0]
    return float(np.mean(vals)) if vals else float("nan")


def boot_indices(n_dates, block=2, n_boot=4000, seed=0):
    block = max(1, min(block, n_dates))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n_dates / block))
    starts = rng.integers(0, n_dates - block + 1, size=(n_boot, n_blocks))
    return (starts[..., None] + np.arange(block)).reshape(n_boot, -1)[:, :n_dates]


def paired_ratio(deltas_a, deltas_b, denoms, idx_matrix):
    full = np.arange(next(iter(deltas_a.values())).shape[0])
    pa, pb = axis_value(deltas_a, denoms, full), axis_value(deltas_b, denoms, full)
    ratios = np.array([axis_value(deltas_b, denoms, i)
                       / max(axis_value(deltas_a, denoms, i), 1e-12)
                       for i in idx_matrix])
    lo, hi = np.percentile(ratios, [2.5, 97.5])
    n_boot = len(ratios)
    p = 2.0 * min(float((ratios <= 1.0).mean()), float((ratios >= 1.0).mean()))
    return {"point_a": pa, "point_b": pb,
            "ratio": (pb / pa if pa > 0 else float("nan")),
            "ci_lo": float(lo), "ci_hi": float(hi),
            "p_value": max(min(p, 1.0), 1.0 / n_boot)}
