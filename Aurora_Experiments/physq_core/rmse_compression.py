# rmse_compression.py
import numpy as np


HEADLINE = {"Z500": "geopotential_500", "T850": "temperature_850",
            "T2M": "2m_temperature", "MSLP": "mean_sea_level_pressure",
            "Q700": "specific_humidity_700",
            "U500": "u_component_of_wind_500", "V500": "v_component_of_wind_500"}


def headline_keys(lead):
    return [f"w_rmse_{v}_{lead}" for v in HEADLINE.values()]


def tag_data(pt, tag):
    d = pt[tag]
    if isinstance(d, dict) and isinstance(d.get("metrics"), dict):
        return d["metrics"]
    return d


def _per_init_fracs(pt, tag, lead, variables):
    """(config - fp32)/fp32 per init, averaged over variables. Shape (n_dates,)."""
    fp = tag_data(pt, "FP32")
    cfg = tag_data(pt, tag)
    inits = sorted(fp.keys())
    out = []
    for i in inits:
        per_var = []
        for vk in variables:
            f = float(fp[i][lead]["RMSE"][vk])
            c = float(cfg[i][lead]["RMSE"][vk])
            per_var.append((c - f) / f)
        out.append(float(np.mean(per_var)))
    return np.asarray(out)


def rmse_degradation(pt, tag, lead, variables=None):
    """Mean signed RMSE degradation vs fp32, in percent. Negative = better than fp32."""
    v = variables if variables is not None else headline_keys(lead)
    return 100.0 * float(np.mean(_per_init_fracs(pt, tag, lead, v)))


def rmse_difference(pt, tag_a, tag_b, lead, variables=None):
    v = variables if variables is not None else headline_keys(lead)
    a = _per_init_fracs(pt, tag_a, lead, v)
    b = _per_init_fracs(pt, tag_b, lead, v)
    return {"diff": 100.0 * float(np.mean(a - b))}
