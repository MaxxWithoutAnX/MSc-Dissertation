""" used to plot graphs for the initial fully quantised runs. Individual graphs and statistics to understand quantisation damage.
"""
import sys

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from eval_metrics import effective_resolution_wavelength, predictable_scale_wavelength
from plot_common import (
    BAND_BLOCK, EARTH_CIRCUMFERENCE_KM, EFF_RES_THRESHOLD, KEY_VARS, PSD_UNITS,
    SVR_DEFAULT_METHOD, block_bootstrap, deseasonalize, ensure_dir, load_run,
    metric_registry, per_init_series, save_fig, spearman_matrix,
)

key = sys.argv[1] if len(sys.argv) > 1 else "FP32"
run = load_run(key)
out_dir = ensure_dir(f"plots/{key}")

dates = run.dates
lead_times = run.leads
lt_colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(lead_times)))


def boot_band(per_date):
    """(mean, lo, hi) over the date axis (95% CI, monthly blocks)."""
    return block_bootstrap(np.asarray(per_date), block=BAND_BLOCK)


# --- RMSE vs lead time (shaded = 95% block-bootstrap CI across init dates) ---------
fig, axes = plt.subplots(3, 3, figsize=(9, 9), sharey=False)
for ax, (label, var) in zip(axes.flat, KEY_VARS.items()):
    m, lo, hi = boot_band(run.scalar("RMSE", f"w_rmse_{var}_{{lt}}"))
    ax.plot(lead_times, m, marker='o')
    ax.fill_between(lead_times, lo, hi, color='C0', alpha=0.25)
    ax.set_title(label)
    ax.set_xlabel("Lead time (h)")
    ax.set_ylabel("RMSE")
    ax.grid(True, alpha=0.3)
plt.suptitle("RMSE vs Lead Time")
plt.tight_layout()
save_fig(fig, out_dir, "rmse_vs_lead_time.png")

fig, axes = plt.subplots(3, 3, figsize=(9, 9), sharey=False)
for ax, (label, var) in zip(axes.flat, KEY_VARS.items()):
    m, lo, hi = boot_band(run.scalar("spec_div", f"sh_spec_div_{var}_{{lt}}"))
    ax.plot(lead_times, m, marker='o', color='C0', label="fwd KL (missing power)")
    ax.fill_between(lead_times, lo, hi, color='C0', alpha=0.25)
    mb, lob, hib = boot_band(run.scalar("spec_div", f"sh_spec_div_back_{var}_{{lt}}"))
    ax.plot(lead_times, mb, marker='s', ls='--', color='C1', label="bwd KL (spurious power)")
    ax.fill_between(lead_times, lob, hib, color='C1', alpha=0.25)
    ax.set_title(label)
    ax.set_xlabel("Lead time (h)")
    ax.set_ylabel("Spectral divergence")
    ax.grid(True, alpha=0.3)
    if ax is axes.flat[0]:
        ax.legend(fontsize=7)
plt.suptitle("Spherical-harmonic spectral divergence vs lead time (fwd = blurring, bwd = injection)")
plt.tight_layout()
save_fig(fig, out_dir, "spec_div_vs_lead_time.png")

# --- Spectral residuals -------------------------------------------------------------
fig, axes = plt.subplots(3, 3, figsize=(9, 9), sharey=False)
for ax, (label, var) in zip(axes.flat, KEY_VARS.items()):
    m, lo, hi = boot_band(run.scalar("spec_res", f"sh_spec_res_{var}_{{lt}}"))
    ax.plot(lead_times, m, marker='o')
    ax.fill_between(lead_times, lo, hi, color='C0', alpha=0.25)
    ax.set_title(label)
    ax.set_xlabel("Lead time (h)")
    ax.set_ylabel("Spectral residual")
    ax.grid(True, alpha=0.3)
plt.suptitle("Spherical-harmonic spectral residuals vs lead time")
plt.tight_layout()
save_fig(fig, out_dir, "spec_res_vs_lead_time.png")

# --- RMSE seasonality: per-init RMSE through the year, one line per lead -----------
fig, axes = plt.subplots(3, 3, figsize=(15, 11))
for ax, (label, var) in zip(axes.flat, KEY_VARS.items()):
    per_date = run.scalar("RMSE", f"w_rmse_{var}_{{lt}}")          # [D, L]
    for j, lt in enumerate(lead_times):
        ax.plot(dates, per_date[:, j], label=f"+{lt}h", linewidth=0.8)
    ax.set_xlabel("Init time")
    ax.set_ylabel("RMSE")
    ax.set_title(label)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
fig.suptitle(f"Lat-weighted RMSE seasonality ({len(dates)} inits)")
fig.autofmt_xdate()
fig.tight_layout()
save_fig(fig, out_dir, "rmse_seasonality.png")

# --- RQE vs forecast time + percentile curves ---------------------------------------
q_arr = np.asarray(run.data[dates[0]][lead_times[0]]["RQE"][f"quantiles_{lead_times[0]}"])
xticks_lt = np.arange(0, max(lead_times) + 1, 24)

fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharey=False)
for ax, (label, var) in zip(axes.flat, KEY_VARS.items()):
    means, lo, hi = boot_band(run.scalar("RQE", f"rqe_{var}_{{lt}}"))
    ax.plot(lead_times, means, marker='o', color='C3', lw=1.2, label='Stormer')
    ax.fill_between(lead_times, lo, hi, color='C3', alpha=0.25)
    ax.axhline(0.0, color='black', lw=0.8, ls='--')
    ax.set_title(label)
    ax.set_xlabel("Forecast time (hrs)")
    ax.set_ylabel(f"{label} relative quantile error")
    ax.set_xticks(xticks_lt)
    ax.grid(True, alpha=0.3)
    if ax is axes.flat[0]:
        ax.legend(fontsize=8, loc='upper right')
fig.suptitle("RQE vs forecast time")
fig.tight_layout()
save_fig(fig, out_dir, "rqe_vs_lead_time.png")

x_pos = -np.log10(1.0 - q_arr)   # 90%->1, 99%->2, 99.9%->3, 99.99%->4
fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharey=False)
for ax, (label, var) in zip(axes.flat, KEY_VARS.items()):
    truth_curves = np.concatenate([run.spectrum_stack("RQE", f"q_true_{var}_{{lt}}", lt)
                                   for lt in lead_times])
    ax.plot(x_pos, truth_curves.mean(axis=0), color='black', ls='--', lw=1.4, label='ERA5')
    for lt, c in zip(lead_times, lt_colors):
        pred_curves = run.spectrum_stack("RQE", f"q_pred_{var}_{{lt}}", lt)
        ax.plot(x_pos, pred_curves.mean(axis=0), color=c, lw=1.2, label=f"+{lt}h")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{q*100:g}%" for q in q_arr])
    ax.set_xlabel("Percentile")
    ax.set_ylabel(label)
    ax.set_title(label)
    ax.grid(True, alpha=0.3)
    if ax is axes.flat[0]:
        ax.legend(fontsize=8, loc='upper left')
fig.suptitle("Percentile curves")
fig.tight_layout()
save_fig(fig, out_dir, "rqe_percentile_curves.png")

# --- Geostrophic wind balance: vertical profiles ------------------------------------
levels = run.levels("wind_balance")


def wbal_profiles(metric):
    """(pred {lt: [D, n_levels]}, truth [D, n_levels]) so plots can bootstrap
    across init dates. Truth at the first lead only (ERA5 balance barely varies
    with valid time and one lead keeps rows strictly time-ordered)."""
    pred_prof = {lt: np.column_stack([run.scalar_at("wind_balance",
                                                    f"wbal_{metric}_pred_{L}_{{lt}}", lt)
                                      for L in levels]) for lt in lead_times}
    lt0 = lead_times[0]
    truth = np.column_stack([run.scalar_at("wind_balance",
                                           f"wbal_{metric}_truth_{L}_{{lt}}", lt0)
                             for L in levels])
    return pred_prof, truth


def plot_wbal(ax, metric, xlabel):
    """95% bootstrap bands on ERA5 and the longest lead only, for readability."""
    pred_prof, truth_prof = wbal_profiles(metric)
    tm, tlo, thi = block_bootstrap(truth_prof, block=BAND_BLOCK)
    ax.plot(tm, levels, color="black", ls="--", lw=1.6, marker="o", label="ERA5 truth")
    ax.fill_betweenx(levels, tlo, thi, color="black", alpha=0.15)
    for lt, c in zip(lead_times, lt_colors):
        pm, plo, phi = block_bootstrap(pred_prof[lt], block=BAND_BLOCK)
        ax.plot(pm, levels, color=c, lw=1.2, marker="o", label=f"+{lt}h")
        if lt == max(lead_times):
            ax.fill_betweenx(levels, plo, phi, color=c, alpha=0.25)
    ax.invert_yaxis()
    ax.set_yticks(levels)
    ax.set_yticklabels(levels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Pressure / hPa")
    ax.grid(True, alpha=0.3)


fig, ax = plt.subplots(figsize=(6, 7))
plot_wbal(ax, "ageo_geo", "|V$_{ag}$| / |V$_g$|")
ax.set_title("Geostrophic wind balance (extra-tropics, |lat| >= 20°)")
ax.legend(fontsize=8)
fig.tight_layout()
save_fig(fig, out_dir, "wind_balance_profile.png")

fig, axes = plt.subplots(1, 3, figsize=(15, 7), sharey=True)
plot_wbal(axes[0], "ageo_geo", "|V$_{ag}$| / |V$_g$|")
plot_wbal(axes[1], "vag", "|V$_{ag}$|  /  m s$^{-1}$")
plot_wbal(axes[2], "vg", "|V$_g$|  /  m s$^{-1}$")
axes[0].set_title("Ratio |V$_{ag}$|/|V$_g$|")
axes[1].set_title("Ageostrophic |V$_{ag}$|")
axes[2].set_title("Geostrophic |V$_g$|")
axes[0].legend(fontsize=8)
fig.suptitle("Geostrophic wind balance components (extra-tropics, |lat| >= 20°)")
fig.tight_layout()
save_fig(fig, out_dir, "wind_balance_components.png")

# --- Global dry air mass conservation ------------------------------------------------
def _exp_scale(arr):                               # power-of-ten scale + exponent for labels
    m = float(np.abs(arr).max())
    e = int(np.floor(np.log10(m))) if m > 0 else 0
    return 10.0 ** e, e


lt_arr = np.array(lead_times, dtype=float)
Md_pred = run.scalar("dry_air_mass", "dryair_Md_pred_{lt}")     # [D, L], kg
Md_truth = run.scalar("dry_air_mass", "dryair_Md_truth_{lt}")
Md_err = run.scalar("dry_air_mass", "dryair_Md_err_{lt}")

res_pred = np.abs(np.diff(Md_pred, axis=1))        # |ΔMd| between consecutive saved leads
res_truth = np.abs(np.diff(Md_truth, axis=1))
lt_mid = lt_arr[1:]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
ax = axes[0]
pm, plo, phi = boot_band(Md_pred / 1e18)
tm, tlo, thi = boot_band(Md_truth / 1e18)
ax.plot(lt_arr, pm, marker='o', color='C3', label='Forecast')
ax.fill_between(lt_arr, plo, phi, color='C3', alpha=0.25)
ax.plot(lt_arr, tm, marker='o', color='black', ls='--', label='ERA5')
ax.fill_between(lt_arr, tlo, thi, color='black', alpha=0.15)
ax.set_xlabel("Lead time (h)")
ax.set_ylabel("Global dry air mass / 10$^{18}$ kg")
ax.set_title("Dry air mass content")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)

ax = axes[1]
rs, re = _exp_scale(np.concatenate([res_pred, res_truth]))
pm, plo, phi = boot_band(res_pred / rs)
tm, tlo, thi = boot_band(res_truth / rs)
ax.plot(lt_mid, pm, marker='o', color='C3', label='Forecast')
ax.fill_between(lt_mid, plo, phi, color='C3', alpha=0.25)
ax.plot(lt_mid, tm, marker='o', color='black', ls='--', label='ERA5')
ax.fill_between(lt_mid, tlo, thi, color='black', alpha=0.15)
ax.set_ylabel(f"|Δ dry air mass| / 10$^{{{re}}}$ kg")
ax.set_xlabel("Lead time (h)")
ax.set_title("Conservation residual (step-to-step |∂M$_d$/∂t|)")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)

ax = axes[2]
es, ee = _exp_scale(Md_err)
m, lo, hi = boot_band(Md_err / es)
ax.plot(lt_arr, m, marker='o', color='C0')
ax.fill_between(lt_arr, lo, hi, color='C0', alpha=0.25)
ax.axhline(0.0, color='black', lw=0.8, ls='--')
ax.set_xlabel("Lead time (h)")
ax.set_ylabel(f"M$_d$ forecast − ERA5 / 10$^{{{ee}}}$ kg")
ax.set_title("Dry air mass error vs ERA5")
ax.grid(True, alpha=0.3)

fig.suptitle("Global dry air mass conservation")
fig.tight_layout()
save_fig(fig, out_dir, "dry_air_mass.png")


def profile_per_date(group, name_fmt, items, lt):
    """[D, len(items)] per-date profile for one lead; name_fmt has {item}/{lt}."""
    return np.column_stack([run.scalar_at(group, name_fmt.replace("{item}", str(i)), lt)
                            for i in items])


def plot_profile(ax, group, name_fmt, items, yvals, xlabel):
    """Vertical-profile plot (pred per lead + truth), bands on ERA5 + longest lead."""
    tm, tlo, thi = block_bootstrap(
        profile_per_date(group, name_fmt.replace("pred", "truth"), items, lead_times[0]),
        block=BAND_BLOCK)
    ax.plot(tm, yvals, color="black", ls="--", lw=1.6, marker="o", label="ERA5 truth")
    ax.fill_betweenx(yvals, tlo, thi, color="black", alpha=0.15)
    for lt, c in zip(lead_times, lt_colors):
        pm, plo, phi = block_bootstrap(profile_per_date(group, name_fmt, items, lt),
                                       block=BAND_BLOCK)
        ax.plot(pm, yvals, color=c, lw=1.2, marker="o", label=f"+{lt}h")
        if lt == max(lead_times):
            ax.fill_betweenx(yvals, plo, phi, color=c, alpha=0.25)
    ax.invert_yaxis()
    ax.set_yticks(yvals)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Pressure / hPa")
    ax.grid(True, alpha=0.3)


# --- Hypsometric residual -------------------------------------------------------------
pairs = run.hyps_pairs()
pair_mid = [0.5 * (int(p.split("-")[0]) + int(p.split("-")[1])) for p in pairs]
fig, axes = plt.subplots(1, 2, figsize=(11, 7), sharey=True)
plot_profile(axes[0], "hypsometric", "hyps_rms_pred_{item}_{lt}", pairs, pair_mid,
             "RMS residual / m$^2$s$^{-2}$")
lt_last = max(lead_times)
resid = profile_per_date("hypsometric", "hyps_rms_pred_{item}_{lt}", pairs, lt_last)
thick = profile_per_date("hypsometric", "hyps_thick_pred_{item}_{lt}", pairs, lt_last)
rm, rlo, rhi = block_bootstrap(100.0 * resid / thick, block=BAND_BLOCK)
resid_t = profile_per_date("hypsometric", "hyps_rms_truth_{item}_{lt}", pairs, lead_times[0])
thick_t = profile_per_date("hypsometric", "hyps_thick_truth_{item}_{lt}", pairs, lead_times[0])
tm, tlo, thi = block_bootstrap(100.0 * resid_t / thick_t, block=BAND_BLOCK)
axes[1].plot(tm, pair_mid, color="black", ls="--", lw=1.6, marker="o", label="ERA5 truth")
axes[1].fill_betweenx(pair_mid, tlo, thi, color="black", alpha=0.15)
axes[1].plot(rm, pair_mid, color="C3", lw=1.2, marker="o", label=f"+{lt_last}h")
axes[1].fill_betweenx(pair_mid, rlo, rhi, color="C3", alpha=0.25)
axes[1].set_xlabel("residual / thickness  (%)")
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=8)
axes[0].set_title("Hypsometric RMS residual")
axes[1].set_title(f"Relative residual (+{lt_last}h)")
axes[0].legend(fontsize=8)
fig.suptitle("Hypsometric (thickness) consistency: Φ vs R$_d$·T̄·ln(p₁/p₂)")
fig.tight_layout()
save_fig(fig, out_dir, "hypsometric_residual.png")

# --- Negative humidity ----------------------------------------------------------------
frac = run.scalar("neg_humidity", "negq_frac_pred_{lt}")
mass = run.scalar("neg_humidity", "negq_mass_pred_{lt}")
frac_t = run.scalar("neg_humidity", "negq_frac_truth_{lt}")
mass_t = run.scalar("neg_humidity", "negq_mass_truth_{lt}")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, (vals, vals_t, ylabel) in zip(axes, [
        (frac, frac_t, "fraction of cells with q < 0"),
        (np.abs(mass), np.abs(mass_t), "|negative moisture mass| / kg")]):
    m, lo, hi = boot_band(vals)
    tm, tlo, thi = boot_band(vals_t)
    ax.plot(lead_times, m, marker='o', color='C3', label='Forecast')
    ax.fill_between(lead_times, lo, hi, color='C3', alpha=0.25)
    ax.plot(lead_times, tm, marker='o', color='black', ls='--', label='ERA5')
    ax.fill_between(lead_times, tlo, thi, color='black', alpha=0.15)
    ax.set_xlabel("Lead time (h)")
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8)
fig.suptitle("Unphysical negative specific humidity")
fig.tight_layout()
save_fig(fig, out_dir, "negative_humidity.png")


# --- PSD (spherical, psd.png), zonal PSD ratio + effective resolution -------------------
wavenum = run.wavenumbers("power spectrum")
wavelength_km = EARTH_CIRCUMFERENCE_KM / wavenum
n_d = len(dates)

psd_truth_mean, psd_pred_mean, ratio_mean = {}, {}, {}
for lt in lead_times:
    t = np.stack([run.spectrum_mean("power spectrum", f"psd_truth_{v}_{{lt}}", lt)
                  for v in KEY_VARS.values()])
    p = np.stack([run.spectrum_mean("power spectrum", f"psd_preds_{v}_{{lt}}", lt)
                  for v in KEY_VARS.values()])
    psd_truth_mean[lt], psd_pred_mean[lt] = t, p
    ratio_mean[lt] = p / t          # ratio of mean PSDs (= ratio of summed PSDs)

sh_wavelength_km = run.sh_wavelength_km()
sh_nyquist_km = float(sh_wavelength_km[len(sh_wavelength_km) // 2])
def _mark_subgrid(ax):
    ax.axvspan(float(min(sh_wavelength_km)), sh_nyquist_km, color="gray", alpha=0.08, zorder=0)
sh_psd_truth_mean, sh_psd_pred_mean = {}, {}
for lt in lead_times:
    sh_psd_truth_mean[lt] = np.stack([run.spectrum_mean("sh power spectrum", f"sh_psd_truth_{v}_{{lt}}", lt)
                                      for v in KEY_VARS.values()])
    sh_psd_pred_mean[lt] = np.stack([run.spectrum_mean("sh power spectrum", f"sh_psd_preds_{v}_{{lt}}", lt)
                                     for v in KEY_VARS.values()])
sh_ratio_mean = {lt: sh_psd_pred_mean[lt] / sh_psd_truth_mean[lt] for lt in lead_times}

fig, axes = plt.subplots(3, 3, figsize=(14, 11))
for i, (label, var) in enumerate(KEY_VARS.items()):
    ax = axes.flat[i]
    ax.loglog(sh_wavelength_km, sh_psd_truth_mean[lead_times[0]][i], color="black", lw=1.6, label="Truth")
    for lt, c in zip(lead_times, lt_colors):
        ax.loglog(sh_wavelength_km, sh_psd_pred_mean[lt][i], color=c, lw=1.0, label=f"+{lt}h")
    ax.set_title(label)
    ax.set_xlabel("Wavelength / km")
    ax.set_ylabel(f"Mean power / {PSD_UNITS.get(label, '')}")
    _mark_subgrid(ax)
    ax.grid(True, which="both", alpha=0.3)
    if i == 0:
        ax.legend(fontsize=8, loc="lower left")
fig.suptitle("Spherical-harmonic PSD")
fig.tight_layout()
save_fig(fig, out_dir, "psd.png")

fig, axes = plt.subplots(3, 3, figsize=(14, 11))
for i, (label, var) in enumerate(KEY_VARS.items()):
    ax = axes.flat[i]
    effective_res_by_lt = [
        effective_resolution_wavelength(sh_ratio_mean[lt][i], sh_wavelength_km, EFF_RES_THRESHOLD)
        for lt in lead_times
    ]
    bars = ax.bar(lead_times, effective_res_by_lt, color=lt_colors, width=0.7)
    ax.bar_label(bars, fmt="%.0f", fontsize=7, padding=2)
    ax.set_title(label)
    ax.set_xticks(lead_times)
    ax.set_xticklabels([f"+{lt}h" for lt in lead_times], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Effective resolution / km")
    ax.grid(True, axis="y", alpha=0.3)
fig.suptitle(f"Effective resolution vs lead time "
             f"(PSD ratio first persistently below {EFF_RES_THRESHOLD})")
fig.tight_layout()
save_fig(fig, out_dir, "effective_resolution.png")

fig, axes = plt.subplots(3, 3, figsize=(14, 11))
for i, (label, var) in enumerate(KEY_VARS.items()):
    ax = axes.flat[i]
    for lt, c in zip(lead_times, lt_colors):
        ax.semilogx(sh_wavelength_km, sh_ratio_mean[lt][i], color=c, lw=1.2, label=f"+{lt}h")
    ax.axhline(1.0, color="black", lw=0.8, ls="--")
    ax.set_title(label)
    ax.set_xlabel("Wavelength / km")
    ax.set_ylabel("Mean power relative to targets")
    ax.set_ylim(0, 1.4)
    _mark_subgrid(ax)
    ax.grid(True, which="both", alpha=0.3)
    if i == 0:
        ax.legend(fontsize=8, loc="lower left")
fig.suptitle("Spherical-harmonic PSD ratio")
fig.tight_layout()
save_fig(fig, out_dir, "psd_ratio.png")

# --- Difference Kinetic Energy (DKE), Selz & Craig (2023) ------------------------------
dke_levels_all = run.levels("dke")
dke_plot_levels = [L for L in (250, 500) if L in dke_levels_all]   # jet + mid-troposphere

if dke_plot_levels:
    dke_wavelength_km = sh_wavelength_km           # SH DKE shares the psd degrees->wavelength axis

    dke_mean = {lt: {L: run.spectrum_mean("dke", f"sh_dke_dke_{L}_{{lt}}", lt)
                     for L in dke_plot_levels} for lt in lead_times}
    bg_mean = {lt: {L: run.spectrum_mean("dke", f"sh_dke_bg_{L}_{{lt}}", lt)
                    for L in dke_plot_levels} for lt in lead_times}

    fig, axes = plt.subplots(1, len(dke_plot_levels),
                             figsize=(7 * len(dke_plot_levels), 5), squeeze=False)
    for j, L in enumerate(dke_plot_levels):
        ax = axes[0, j]
        bg_ref = bg_mean[lead_times[0]][L]
        ax.loglog(dke_wavelength_km, bg_ref, color="black", lw=1.6, label="Background KE")
        ax.loglog(dke_wavelength_km, 2 * bg_ref, color="black", lw=1.0, ls="--",
                  label="Saturation (2xbg)")
        for lt, c in zip(lead_times, lt_colors):
            ax.loglog(dke_wavelength_km, dke_mean[lt][L], color=c, lw=1.2, label=f"+{lt}h")
        ax.set_title(f"{L} hPa")
        ax.set_xlabel("Wavelength / km")
        ax.set_ylabel("KE / m²s⁻²")
        _mark_subgrid(ax)
        ax.grid(True, which="both", alpha=0.3)
        if j == 0:
            ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Difference kinetic energy spectra (Selz & Craig, spherical-harmonic)")
    fig.tight_layout()
    save_fig(fig, out_dir, "dke.png")

    fig, axes = plt.subplots(1, len(dke_plot_levels),
                             figsize=(6 * len(dke_plot_levels), 4.5), squeeze=False)
    for j, L in enumerate(dke_plot_levels):
        ax = axes[0, j]
        scale_by_lt = [predictable_scale_wavelength(dke_mean[lt][L], bg_mean[lt][L],
                                                    dke_wavelength_km)
                       for lt in lead_times]
        bars = ax.bar(lead_times, scale_by_lt, color=lt_colors, width=0.7)
        ax.bar_label(bars, fmt="%.0f", fontsize=7, padding=2)
        ax.set_title(f"{L} hPa")
        ax.set_xticks(lead_times)
        ax.set_xticklabels([f"+{lt}h" for lt in lead_times], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Predictable scale / km")
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Predictable scale vs lead time (DKE reaches half saturation)")
    fig.tight_layout()
    save_fig(fig, out_dir, "predictable_scale.png")

    # DKE error growth: band-summed DKE / saturation vs lead time
    BAND_SPLIT_KM = 1000.0
    small_band = dke_wavelength_km < BAND_SPLIT_KM
    bands = [("All scales", slice(None), "black"),
             (f"Small scale (<{BAND_SPLIT_KM:.0f} km)", small_band, "tab:red"),
             (f"Large scale (>={BAND_SPLIT_KM:.0f} km)", ~small_band, "tab:blue")]

    fig, axes = plt.subplots(1, len(dke_plot_levels),
                             figsize=(6 * len(dke_plot_levels), 4.5), squeeze=False)
    for j, L in enumerate(dke_plot_levels):
        ax = axes[0, j]
        for name, mask, color in bands:
            frac_b = [dke_mean[lt][L][mask].sum() / (2.0 * bg_mean[lt][L][mask].sum())
                      for lt in lead_times]
            ax.plot(lead_times, frac_b, marker="o", color=color, lw=1.4, label=name)
        ax.axhline(1.0, color="gray", ls="--", lw=1.0)
        ax.axhline(0.5, color="gray", ls=":", lw=1.0)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{L} hPa")
        ax.set_xticks(lead_times)
        ax.set_xticklabels([f"+{lt}h" for lt in lead_times])
        ax.set_xlabel("Lead time / h")
        ax.set_ylabel("DKE / saturation  (fraction of predictability lost)")
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("DKE error growth vs lead time (Selz & Craig)")
    fig.tight_layout()
    save_fig(fig, out_dir, "dke_growth.png")

# --- Quantisation-perturbation DKE (referenced to the fp32 forecast) --------------------
if run.has_pert:
    pert_levels = [L for L in (250, 500) if L in run.levels("dke_pert")]
    if pert_levels:
        pwl = sh_wavelength_km
        pdke = {lt: {L: run.spectrum_mean("dke_pert", f"sh_dke_dke_{L}_{{lt}}", lt)
                     for L in pert_levels} for lt in lead_times}
        pbg = {lt: {L: run.spectrum_mean("dke_pert", f"sh_dke_bg_{L}_{{lt}}", lt)
                    for L in pert_levels} for lt in lead_times}

        fig, axes = plt.subplots(1, len(pert_levels),
                                 figsize=(7 * len(pert_levels), 5), squeeze=False)
        for j, L in enumerate(pert_levels):
            ax = axes[0, j]
            bg_ref = pbg[lead_times[0]][L]                       # fp32 (model) KE spectrum
            ax.loglog(pwl, bg_ref, color="black", lw=1.6, label="fp32 background KE")
            ax.loglog(pwl, 2 * bg_ref, color="black", lw=1.0, ls="--", label="Saturation (2xbg)")
            for lt, c in zip(lead_times, lt_colors):
                ax.loglog(pwl, pdke[lt][L], color=c, lw=1.2, label=f"+{lt}h")
            ax.set_title(f"{L} hPa")
            ax.set_xlabel("Wavelength / km")
            ax.set_ylabel("DKE vs fp32 / m²s⁻²")
            ax.grid(True, which="both", alpha=0.3)
            if j == 0:
                ax.legend(fontsize=8, loc="lower right")
        fig.suptitle(f"Quantisation-perturbation DKE - {key} vs fp32")
        fig.tight_layout()
        save_fig(fig, out_dir, "perturbation_dke.png")
else:
    print("Skipping perturbation DKE - no dke_pert group (this is the FP32 baseline).")

SPEARMAN_LT = 120 if 120 in lead_times else lead_times[-1]
specs = metric_registry(run)
cols, labels = [], []
for spec in specs:
    v = per_init_series(run, spec, SPEARMAN_LT)
    v = deseasonalize(v, dates, "monthly_anom")
    if np.std(v) == 0:      # e.g. EffRes pinned at the grid limit - uninformative
        print(f"Spearman: dropping constant column {spec.label!r}")
        continue
    cols.append(v)
    labels.append(spec.label)
X = np.column_stack(cols)
C = spearman_matrix(X)

n = len(labels)
fig, ax = plt.subplots(figsize=(0.42 * n + 3, 0.42 * n + 2.5))
im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(n))
ax.set_xticklabels(labels, rotation=90, fontsize=7)
ax.set_yticks(range(n))
ax.set_yticklabels(labels, fontsize=7)
for i in range(n):
    for j in range(n):
        ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=5,
                color="white" if abs(C[i, j]) > 0.6 else "black")
fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman ρ")
ax.set_title(f"Metric Spearman correlation - monthly anomalies, +{SPEARMAN_LT}h, "
             f"n={len(dates)} inits ({key})")
fig.tight_layout()
save_fig(fig, out_dir, "metric_spearman.png")

print(f"done: {out_dir}")
