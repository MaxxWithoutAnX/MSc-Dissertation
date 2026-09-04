"""Paired comparison of the configurations disotrtions. Analysis script no inference here. A label referes to the label given to the metric eg RMSE Q700, Vag/Vg 500hPa, ...
"""
import numpy as np

import ablation_comp as ac
import plot_common as pc


def specs_for_family(fp32_run, config_run, family, lead):
    """The metric specs in one registry family that both runs can supply at `lead`.
    Args:
        fp32_run [MetricsRun]   : the FP32 baseline
        config_run [MetricsRun] : the config being compared against it
        family [str]            : registry family key, e.g. "wind_balance", "dry_air_mass"
        lead [int]              : lead time in hours
    Returns:
        list[MetricSpec]
    """
    specs = ac.available_specs(fp32_run, config_run, ac.metric_registry(fp32_run), lead)
    return [s for s in specs if s.group == family]


def family_deltas(fp32_run, config_run, specs, lead, dates=None,
                  spread_method="iqr_raw"):
    """per initialisation paired deltas
    Args:
        fp32_run [MetricsRun]   : the FP32 baseline
        config_run [MetricsRun] : the config being compared against it
        specs [list[MetricSpec]]: metrics to extract
        lead [int]              : lead time in hours
        dates [list | None]     : init dates
        spread_method [str]     : "iqr_raw" (default) uses the FP32 raw IQR
    Returns:
        tuple: (deltas, denoms)
            deltas [dict]: {label: np.ndarray} of per-init differences, length n_inits
            denoms [dict]: {label: float} FP32 spread in that metric's own units
    """
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
    """|mean(delta over the selected inits)| / denom
    Args:
        deltas [dict]   : {label: per-init delta array}
        denoms [dict]   : {label: denominator}
        idx [np.ndarray]: integer indices selecting which inits to average. np.arange(n) for the full sample
    Returns:
        float: the axis value in SVR units
    """
    vals = [abs(float(deltas[l][idx].mean())) / denoms[l]
            for l in deltas if denoms.get(l, 0.0) > 0]
    return float(np.mean(vals)) if vals else float("nan")


def label_values(deltas, denoms):
    """ Per label distortion sample {label: |mean delta| / denom}
    Args:
        deltas [dict]: {label: per-init delta array}
        denoms [dict]: {label: denominator}
    Returns:
        dict: {label: float}. Per label distotion.
    """
    return {l: abs(float(deltas[l].mean())) / denoms[l]
            for l in deltas if denoms.get(l, 0.0) > 0}


def label_ratios(deltas_a, deltas_b, denoms):
    """ Per label ratio of b/a. {label: D_b(label) / D_a(label)}
    """
    va, vb = label_values(deltas_a, denoms), label_values(deltas_b, denoms)
    return {l: vb[l] / max(va[l], 1e-12) for l in va if l in vb}


def label_contributions(deltas_b, denoms_b, axis_a):
    if not np.isfinite(axis_a) or axis_a <= 0:
        return {}
    return {l: v / axis_a for l, v in label_values(deltas_b, denoms_b).items()}


def paired_ratio(deltas_a, deltas_b, denoms):
    """ {point_a, point_b, ratio} for the two configs' axis distortion. Used to compare physics and RMSE allocations and get ratios.
    """
    full = np.arange(next(iter(deltas_a.values())).shape[0])
    pa, pb = axis_value(deltas_a, denoms, full), axis_value(deltas_b, denoms, full)
    return {"point_a": pa, "point_b": pb,
            "ratio": (pb / pa if pa > 0 else float("nan"))}
