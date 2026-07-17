import datetime
import os
from collections import namedtuple

import numpy as np
import torch

# --- run identities / styling ---------------------------------------------------
RUNS = ["FP32", "W8", "W8A8"]
QUANT_RUNS = ["W8", "W8A8"]
RUN_COLORS = {"FP32": "black", "W8": "tab:blue", "W8A8": "tab:orange"}
RUN_LS = {"FP32": "--", "W8": "-", "W8A8": "-"}

KEY_VARS = {
    "T2M":   "2m_temperature",
    "U10":   "10m_u_component_of_wind",
    "V10":   "10m_v_component_of_wind",
    "MSLP":  "mean_sea_level_pressure",
    "Z500":  "geopotential_500",
    "T850":  "temperature_850",
    "Q700":  "specific_humidity_700",
    "U500":  "u_component_of_wind_500",
    "V500":  "v_component_of_wind_500",
}
SCORECARD_VARS = {
    "Z500": "geopotential_500",
    "T850": "temperature_850",
    "Q700": "specific_humidity_700",
    "U500": "u_component_of_wind_500",
    "MSLP": "mean_sea_level_pressure",
    "T2M":  "2m_temperature",
}
SPECTRAL_VARS = {
    "Z500": "geopotential_500",
    "T850": "temperature_850",
    "Q700": "specific_humidity_700",
}
# PSD y-axis units (square of the variable's physical units in ERA5).
PSD_UNITS = {
    "T2M": "K²", "T850": "K²",
    "U10": "m²s⁻²", "V10": "m²s⁻²",
    "U500": "m²s⁻²", "V500": "m²s⁻²",
    "MSLP": "Pa²",
    "Z500": "m⁴s⁻⁴",
    "Q700": "(kg/kg)²",
}

EARTH_CIRCUMFERENCE_KM = 40075.0   # equatorial - conventional choice for zonal spectra
EFF_RES_THRESHOLD = 0.5
BAND_EDGES_KM = [(150.0, 500.0), (500.0, 1000.0), (1000.0, np.inf)]
PROFILE_LEVELS_HEADLINE = [850, 500, 250]

PAIRED_BLOCK = 6
BAND_BLOCK = 6
N_BOOT = 2000

# levels/pairs lists are stored in-band under group-specific prefixes
_LEVELS_KEY = {
    "wind_balance": "wbal_levels",
    "div_vort": "divvort_levels",
    "dke": "dke_levels",
    "dke_pert": "dke_levels",
    "q_bias": "qbias_levels",
    "neg_humidity": "negq_levels",
    "dry_air_mass": "dryair_levels",
}


def _val(v):
    """Stored scalars are 0-d torch tensors; be tolerant of plain floats."""
    return float(v.item()) if torch.is_tensor(v) else float(v)


def _arr(v):
    return v.numpy() if torch.is_tensor(v) else np.asarray(v)


class MetricsRun:
    """One run's all_metrics dict with array-shaped accessors."""

    def __init__(self, key, data):
        self.key = key
        self.data = data
        self.dates = sorted(data.keys())         # chronological - bootstrap needs time order
        self.leads = sorted(data[self.dates[0]].keys())
        sample = data[self.dates[0]][self.leads[0]]
        self.has_pert = "dke_pert" in sample

    def scalar(self, group, name_fmt):
        """[D, L] array of a scalar metric; name_fmt has a {lt} placeholder."""
        return np.array([[_val(self.data[d][lt][group][name_fmt.format(lt=lt)])
                          for lt in self.leads] for d in self.dates])

    def scalar_at(self, group, name_fmt, lead):
        """[D] array of one scalar metric at a single lead."""
        return np.array([_val(self.data[d][lead][group][name_fmt.format(lt=lead)])
                         for d in self.dates])

    def spectrum_stack(self, group, name_fmt, lead):
        """[D, n_wave] stack of a stored 1-d tensor at one lead."""
        return np.stack([_arr(self.data[d][lead][group][name_fmt.format(lt=lead)])
                         for d in self.dates])

    def spectrum_mean(self, group, name_fmt, lead):
        return self.spectrum_stack(group, name_fmt, lead).mean(axis=0)

    def levels(self, group):
        lt0 = self.leads[0]
        key = f"{_LEVELS_KEY[group]}_{lt0}"
        return list(self.data[self.dates[0]][lt0][group][key])

    def hyps_pairs(self):
        lt0 = self.leads[0]
        return list(self.data[self.dates[0]][lt0]["hypsometric"][f"hyps_pairs_{lt0}"])

    def wavenumbers(self, group="power spectrum"):
        lt0 = self.leads[0]
        g = self.data[self.dates[0]][lt0][group]
        if group in ("dke", "dke_pert"):
            return _arr(g[f"dke_wavenumbers_{lt0}"]).astype(float)
        var = next(iter(KEY_VARS.values()))
        return _arr(g[f"wavenumbers_{var}_{lt0}"]).astype(float)

    def wavelength_km(self, group="power spectrum"):
        return EARTH_CIRCUMFERENCE_KM / self.wavenumbers(group)

    def sh_degrees(self):
        """Total spherical-harmonic wavenumber axis l = 1..l_max (from 'sh power spectrum')."""
        lt0 = self.leads[0]
        return _arr(self.data[self.dates[0]][lt0]["sh power spectrum"][f"sh_degrees_{lt0}"]).astype(float)

    def sh_wavelength_km(self):
        """Physical wavelength (km) for each SH degree: 2*pi*R/sqrt(l(l+1)), as stored."""
        lt0 = self.leads[0]
        return _arr(self.data[self.dates[0]][lt0]["sh power spectrum"][f"sh_wavelength_km_{lt0}"]).astype(float)


def load_run(key):
    path = f"all_metrics_{key}.pt"
    print(f"loading {path} ...", flush=True)
    data = torch.load(path, weights_only=False, map_location="cpu")
    return MetricsRun(key, data)


# --- bootstrap statistics --------------------
def block_bootstrap(values, block=BAND_BLOCK, n_boot=N_BOOT, ci=95, rng_seed=0):
    values = np.asarray(values, dtype=np.float64)
    D = values.shape[0]
    block = max(1, min(block, D))
    rng = np.random.default_rng(rng_seed)

    n_blocks = int(np.ceil(D / block))
    starts = rng.integers(0, D - block + 1, size=(n_boot, n_blocks))
    idx = (starts[..., None] + np.arange(block)).reshape(n_boot, -1)[:, :D]
    boot_means = values[idx].mean(axis=1)              # [n_boot] or [n_boot, K]

    alpha = (100 - ci) / 2
    lo, hi = np.percentile(boot_means, [alpha, 100 - alpha], axis=0)
    return values.mean(axis=0), lo, hi


def lag1_autocorr(x):
    """Lag-1 autocorrelation of a 1-D series (rows must be in time order)."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 3 or np.std(x[:-1]) == 0 or np.std(x[1:]) == 0:
        return 0.0
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def paired_diff_test(a, b, block=PAIRED_BLOCK, n_boot=N_BOOT, ci=95, rng_seed=0):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape and a.ndim == 1, "paired test needs matching [D] arrays"
    d = a - b
    mean_diff, lo, hi = block_bootstrap(d, block=block, n_boot=n_boot, ci=ci, rng_seed=rng_seed)
    r1 = lag1_autocorr(d)
    n = len(d)
    n_eff = n * (1 - r1) / (1 + r1) if r1 > -1 else float(n)
    return {
        "mean_a": a.mean(),
        "mean_b": b.mean(),
        "mean_diff": float(mean_diff),
        "rel_diff_pct": float(100.0 * mean_diff / b.mean()) if b.mean() != 0 else np.nan,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "significant": bool(lo > 0 or hi < 0),
        "frac_a_higher": float((d > 0).mean()),
        "lag1_autocorr": r1,
        "n": n,
        "n_eff": float(n_eff),
    }


# --- signal-to-variability normalisation -----------------------------------------
def _iqr(x):
    q25, q75 = np.percentile(np.asarray(x, dtype=np.float64), [25, 75])
    return float(q75 - q25)


def deseasonalize(values, dates, method="monthly_anom"):
    values = np.asarray(values, dtype=np.float64)
    if method == "monthly_anom":
        months = np.array([d.month for d in dates])
        anom = values.copy()
        for m in np.unique(months):
            sel = months == m
            anom[sel] -= values[sel].mean()
        return anom
    if method == "harmonic":
        doy = np.array([d.timetuple().tm_yday for d in dates], dtype=np.float64)
        w = 2.0 * np.pi * doy / 365.25
        X = np.column_stack([np.ones_like(w), np.sin(w), np.cos(w),
                             np.sin(2 * w), np.cos(2 * w)])
        coef, *_ = np.linalg.lstsq(X, values, rcond=None)
        return values - X @ coef
    raise ValueError(f"unknown deseasonalize method {method!r}")


def spread(values, dates, method="iqr_monthly_anom"):
    if method == "iqr_raw":
        return _iqr(values)
    if method == "iqr_monthly_anom":
        return _iqr(deseasonalize(values, dates, "monthly_anom"))
    if method == "iqr_harmonic":
        return _iqr(deseasonalize(values, dates, "harmonic"))
    if method == "std_monthly_anom":
        return float(np.std(deseasonalize(values, dates, "monthly_anom")))
    raise ValueError(f"unknown spread method {method!r}")


SVR_DEFAULT_METHOD = "iqr_monthly_anom"


def svr_test(q, f, dates, method=SVR_DEFAULT_METHOD, block=PAIRED_BLOCK):
    q = np.asarray(q, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    denom = spread(f, dates, method)
    mean_diff, lo, hi = block_bootstrap(q - f, block=block)
    if denom == 0:
        return {"svr": np.nan, "svr_ci_lo": np.nan, "svr_ci_hi": np.nan,
                "svr_denom": 0.0, "svr_method": method, "svr_significant": False}
    return {
        "svr": float(mean_diff / denom),
        "svr_ci_lo": float(lo / denom),
        "svr_ci_hi": float(hi / denom),
        "svr_denom": denom,
        "svr_method": method,
        "svr_significant": bool(lo > 0 or hi < 0),
    }


# --- Spearman (numpy-only) --------------------------------------------------------
def _rankdata_avg(x):
    """Average ranks (1-based); ties get the mean of their rank range."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    # average ranks over tie groups
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman_matrix(X):
    """[M, M] Spearman rank correlation of the columns of X [D, M]."""
    R = np.column_stack([_rankdata_avg(X[:, m]) for m in range(X.shape[1])])
    return np.corrcoef(R, rowvar=False)


MetricSpec = namedtuple("MetricSpec", "label group name_fmt kind transform derived_fn",
                        defaults=(None, None))


def _mid_trop_pair(pairs):
    """The hypsometric layer whose midpoint is closest to 550 hPa."""
    def mid(p):
        a, b = p.split("-")
        return 0.5 * (int(a) + int(b))
    return min(pairs, key=lambda p: abs(mid(p) - 550.0))


def _eff_res_series(var):
    def fn(run, lead):
        wl = run.sh_wavelength_km()
        ratios = run.spectrum_stack("sh power spectrum", f"sh_spectral_ratio_{var}_{{lt}}", lead)
        from eval_metrics import effective_resolution_wavelength
        return np.array([effective_resolution_wavelength(r, wl, EFF_RES_THRESHOLD)
                         for r in ratios])
    return fn


def _dke_band_frac_series(level, wl_max_km=1000.0):
    def fn(run, lead):
        wl = run.sh_wavelength_km()
        mask = wl < wl_max_km
        dke = run.spectrum_stack("dke", f"sh_dke_dke_{level}_{{lt}}", lead)
        bg = run.spectrum_stack("dke", f"sh_dke_bg_{level}_{{lt}}", lead)
        return dke[:, mask].sum(axis=1) / (2.0 * bg[:, mask].sum(axis=1))
    return fn


def metric_registry(run):
    sample = run.data[run.dates[0]][run.leads[0]]
    specs = []
    for label, var in SCORECARD_VARS.items():
        specs.append(MetricSpec(f"RMSE {label}", "RMSE", f"w_rmse_{var}_{{lt}}", "error_pos"))
    for label, var in SPECTRAL_VARS.items():
        if "spec_div" in sample:
            specs.append(MetricSpec(f"SpecDiv> {label}", "spec_div", f"sh_spec_div_{var}_{{lt}}", "error_pos"))
            specs.append(MetricSpec(f"SpecDiv< {label}", "spec_div", f"sh_spec_div_back_{var}_{{lt}}", "error_pos"))
        if "spec_res" in sample:
            specs.append(MetricSpec(f"SpecRes {label}", "spec_res", f"sh_spec_res_{var}_{{lt}}", "error_pos"))
        if "RQE" in sample:
            specs.append(MetricSpec(f"|RQE| {label}", "RQE", f"rqe_{var}_{{lt}}", "error_pos", np.abs))
    if "wind_balance" in sample:
        for L in [L for L in PROFILE_LEVELS_HEADLINE if L in run.levels("wind_balance")]:
            specs.append(MetricSpec(f"Vag/Vg {L}hPa", "wind_balance",
                                    f"wbal_ageo_geo_pred_{L}_{{lt}}", "error_pos"))
            specs.append(MetricSpec(f"|Vag| {L}hPa", "wind_balance",
                                    f"wbal_vag_pred_{L}_{{lt}}", "error_pos"))
    if "div_vort" in sample:
        for L in [L for L in PROFILE_LEVELS_HEADLINE if L in run.levels("div_vort")]:
            specs.append(MetricSpec(f"div/vort {L}hPa", "div_vort",
                                    f"divvort_ratio_pred_{L}_{{lt}}", "error_pos"))
    if "hypsometric" in sample:
        pair = _mid_trop_pair(run.hyps_pairs())
        specs.append(MetricSpec(f"Hyps {pair}", "hypsometric",
                                f"hyps_rms_pred_{pair}_{{lt}}", "error_pos"))
    if "dry_air_mass" in sample:
        specs.append(MetricSpec("|DryAir Md err|", "dry_air_mass",
                                "dryair_Md_err_{lt}", "error_pos", np.abs))
    if "neg_humidity" in sample:
        specs.append(MetricSpec("neg-q fraction", "neg_humidity",
                                "negq_frac_pred_{lt}", "error_pos"))
        specs.append(MetricSpec("|neg-q mass|", "neg_humidity",
                                "negq_mass_pred_{lt}", "error_pos", np.abs))
    if "sh power spectrum" in sample:
        specs.append(MetricSpec("EffRes Z500", None, None, "error_pos",
                                None, _eff_res_series("geopotential_500")))
    if "dke" in sample and 500 in run.levels("dke"):
        specs.append(MetricSpec("DKE<1000km 500hPa", None, None, "error_pos",
                                None, _dke_band_frac_series(500)))
    return specs


def per_init_series(run, spec, lead):
    """[D] per-init values for a registry metric at one lead."""
    if spec.derived_fn is not None:
        vals = spec.derived_fn(run, lead)
    else:
        vals = run.scalar_at(spec.group, spec.name_fmt, lead)
    if spec.transform is not None:
        vals = spec.transform(vals)
    return vals


# --- output helpers -----------------------------------------------------------------
def ensure_dir(name):
    os.makedirs(name, exist_ok=True)
    return name


def save_fig(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"wrote {path}", flush=True)
