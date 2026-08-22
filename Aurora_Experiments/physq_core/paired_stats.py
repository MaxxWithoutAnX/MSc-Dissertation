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


def label_values(deltas, denoms):
    return {l: abs(float(deltas[l].mean())) / denoms[l]
            for l in deltas if denoms.get(l, 0.0) > 0}


def label_ratios(deltas_a, deltas_b, denoms):
    va, vb = label_values(deltas_a, denoms), label_values(deltas_b, denoms)
    return {l: vb[l] / max(va[l], 1e-12) for l in va if l in vb}


def label_contributions(deltas_b, denoms_b, axis_a):
    if not np.isfinite(axis_a) or axis_a <= 0:
        return {}
    return {l: v / axis_a for l, v in label_values(deltas_b, denoms_b).items()}


def paired_ratio(deltas_a, deltas_b, denoms):
    full = np.arange(next(iter(deltas_a.values())).shape[0])
    pa, pb = axis_value(deltas_a, denoms, full), axis_value(deltas_b, denoms, full)
    return {"point_a": pa, "point_b": pb,
            "ratio": (pb / pa if pa > 0 else float("nan"))}
