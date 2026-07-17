"""OAT per-layer quantisation-sensitivity analysis for the sampled-2020 ablations."""
import argparse
import csv
import glob
import math
import os
import re
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from plot_common import (
    MetricsRun, MetricSpec, metric_registry, per_init_series,
    paired_diff_test, block_bootstrap, svr_test, spearman_matrix,
    ensure_dir, save_fig,
)

EPS = 1e-12
NC_PATH = os.path.join("data", "era5_sampled_2020_4pm_aurora_0p25.nc")
FULL_FP32_KEY = "FP32"
DKE_PERT_LEVEL = 500
DKE_PERT_WL_MAX_KM = 1000.0

HEATMAP_EXCLUDE = ("div/vort", "Hyps")

CSV_FIELDS = ["group", "metric", "lead", "fp32_mean", "group_mean", "mean_delta",
              "ci_lo", "ci_hi", "rel_pct", "svr", "svr_ci_lo", "svr_ci_hi",
              "share", "norm", "rank", "significant", "lag1_autocorr", "n", "n_eff"]


_CLASS_RANK = {"embed": 0, "block": 1, "adaln": 2, "head": 3, "other": 4}
_CLASS_CMAP = {"embed": "Oranges", "block": "Blues", "adaln": "Purples",
               "head": "Reds", "other": "Greys"}


def _group_class(g):
    if re.match(r"b\d+-\d+_", g):
        return "block"
    return g if g in _CLASS_RANK else "other"


def _group_sort_key(g):
    m = re.match(r"b(\d+)-\d+_(attn|mlp)$", g)
    if m:
        return (_CLASS_RANK["block"], int(m.group(1)), {"attn": 0, "mlp": 1}[m.group(2)])
    return (_CLASS_RANK.get(_group_class(g), 4), 0, 0)


def order_groups(groups):
    """Encoder->backbone->decoder order, so figures read as a pass through the model."""
    return sorted(groups, key=_group_sort_key)


def group_color_map(groups):
    """Colour by component class, shade by depth within the class."""
    by_class = defaultdict(list)
    for g in order_groups(groups):
        by_class[_group_class(g)].append(g)
    colors = {}
    for cls, members in by_class.items():
        cmap = matplotlib.colormaps[_CLASS_CMAP[cls]]
        n = len(members)
        for i, g in enumerate(members):
            colors[g] = cmap(0.7 if n == 1 else 0.4 + 0.55 * i / (n - 1))
    return colors


# --- dke_pert: injected-KE band fraction as one more registry metric ----------------
def _dke_pert_band_series(level=DKE_PERT_LEVEL, wl_max_km=DKE_PERT_WL_MAX_KM):
    """[D] sub-`wl_max_km` saturation fraction of the injected KE:
    sum(DKE_pert) / (2*sum(E_bg)) with E_bg the FP32 forecast's own KE spectrum.
    1.0 = the noise this group alone injects has fully decorrelated those scales."""
    def fn(run, lead):
        wl = run.wavelength_km("dke_pert")
        mask = wl < wl_max_km
        dke = run.spectrum_stack("dke_pert", f"dke_dke_{level}_{{lt}}", lead)
        bg = run.spectrum_stack("dke_pert", f"dke_bg_{level}_{{lt}}", lead)
        return dke[:, mask].sum(axis=1) / (2.0 * bg[:, mask].sum(axis=1))
    return fn


def dke_pert_spec(sample_run):
    """The dke_pert registry row, if the OAT runs carry the family."""
    if not sample_run.has_pert or DKE_PERT_LEVEL not in sample_run.levels("dke_pert"):
        return None
    return MetricSpec(f"DKEpert<{DKE_PERT_WL_MAX_KM:.0f}km {DKE_PERT_LEVEL}hPa",
                      "dke_pert", None, "error_pos", None, _dke_pert_band_series())


def spec_series(run, spec, lead):
    """[D] per-init values; the dke_pert metric is identically 0 for runs without the
    family (fp32 vs itself), which makes it behave like every other metric downstream."""
    if spec.group == "dke_pert" and not run.has_pert:
        return np.zeros(len(run.dates))
    return per_init_series(run, spec, lead)


# --- init-date recovery ---------------------------------------------------------------
def recover_dates(init_keys):
    import datetime as _dt
    keys = sorted(init_keys)
    if keys and isinstance(keys[0], str):
        return [_dt.datetime.strptime(k, "%Y%m%d_%H") for k in keys]
    if keys and not isinstance(keys[0], (int, np.integer)):
        return keys                                        # already datetimes
    if os.path.exists(NC_PATH):
        import xarray as xr
        with xr.open_dataset(NC_PATH) as ds:               # lazy: coords only
            times = ds.time.values
        return [times[i].astype("datetime64[s]").tolist() for i in keys]
    full_path = f"all_metrics_{FULL_FP32_KEY}.pt"
    if os.path.exists(full_path):
        full = torch.load(full_path, map_location="cpu", weights_only=False)
        dates = sorted(full.keys())
        if len(dates) == len(keys):
            return dates
    return None


# --- sensitivity records ----------------------------------------------------------------
def pick_svr_method(dates):
    """Deseasonalisation for the SVR denominator, adapted to the init sampling.
    monthly_anom needs >=2 inits per month (with 1/month every anomaly is exactly 0
    and the IQR degenerates); the harmonic fit works from 12 seasonally-spread inits."""
    if dates is None:
        return "iqr_raw"
    counts = defaultdict(int)
    for d in dates:
        counts[d.month] += 1
    return "iqr_monthly_anom" if min(counts.values()) >= 2 else "iqr_harmonic"


def bootstrap_block(n):
    """Moving-block length ~ n^(1/3) (standard rate; gives block=4 at the 48-init
    seasonal runs, matching plot_common.PAIRED_BLOCK, and block=2 at 12 inits)."""
    return max(1, round(n ** (1.0 / 3.0)))


def available_specs(fp32_run, group_run, specs, lead):
    """Specs extractable from both the baseline and a group run at this lead."""
    out = []
    for spec in specs:
        try:
            spec_series(fp32_run, spec, lead)
            spec_series(group_run, spec, lead)
        except (KeyError, ValueError):
            print(f"  dropping metric '{spec.label}' (keys missing)", flush=True)
            continue
        out.append(spec)
    return out


def full_damage_table(scheme, specs, leads):
    paths = [f"all_metrics_{scheme}.pt", f"all_metrics_{FULL_FP32_KEY}.pt"]
    if not all(os.path.exists(p) for p in paths):
        print(f"  full-run files not found ({paths}) -> per-column-max normalisation",
              flush=True)
        return None
    quant = MetricsRun(scheme, torch.load(paths[0], map_location="cpu", weights_only=False))
    fp32 = MetricsRun(FULL_FP32_KEY, torch.load(paths[1], map_location="cpu", weights_only=False))
    block = bootstrap_block(len(quant.dates))
    table, dropped = {}, 0
    for lead in leads:
        if lead not in quant.leads:
            continue
        for spec in specs:
            try:
                q = spec_series(quant, spec, lead)
                f = spec_series(fp32, spec, lead)
            except (KeyError, ValueError):
                continue
            r = paired_diff_test(q, f, block=block)
            if r["significant"]:
                table[(spec.label, lead)] = r["mean_diff"]
            else:
                dropped += 1
    print(f"  share denominator: {paths[0]} - {paths[1]} ({len(table)} significant "
          f"metric/lead cells; {dropped} non-significant -> col-max fallback)", flush=True)
    return table


def sensitivity_records(fp32_run, oat_runs, specs, groups, leads, dates, block,
                        full_table, norm_mode):
    """One row per (group, metric, lead); share/colmax normalisation per column."""
    svr_method = pick_svr_method(dates)
    recs = []
    for lead in leads:
        for spec in specs:
            base = spec_series(fp32_run, spec, lead)
            col = []
            for g in groups:
                val = spec_series(oat_runs[g], spec, lead)
                r = paired_diff_test(val, base, block=block)
                sv = svr_test(val, base, dates, method=svr_method, block=block)
                col.append({
                    "group": g, "metric": spec.label, "lead": lead,
                    "fp32_mean": float(base.mean()), "group_mean": float(val.mean()),
                    "mean_delta": r["mean_diff"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                    "rel_pct": r["rel_diff_pct"],
                    "svr": sv["svr"], "svr_ci_lo": sv["svr_ci_lo"], "svr_ci_hi": sv["svr_ci_hi"],
                    "significant": bool(r["significant"]),
                    "lag1_autocorr": r["lag1_autocorr"], "n": r["n"], "n_eff": r["n_eff"],
                    "share": float("nan"), "norm": "",
                })
            _normalise_column(col, spec, lead, full_table, norm_mode)
            recs.extend(col)
    return recs


def _normalise_column(col, spec, lead, full_table, norm_mode):
    denom, used = None, None
    if norm_mode != "colmax" and full_table is not None:
        d = full_table.get((spec.label, lead))
        if d is not None and abs(d) >= EPS:
            denom, used = d, "share"
    if denom is None:
        m = max((abs(r["mean_delta"]) for r in col), default=0.0)
        denom, used = (m if m >= EPS else 1.0), "colmax"
    for r in col:
        r["share"] = r["mean_delta"] / denom
        r["norm"] = used


def add_ranks(records):
    """`rank` within each (metric, lead) column; 1 = largest |mean_delta|."""
    cols = defaultdict(list)
    for r in records:
        cols[(r["metric"], r["lead"])].append(r)
    for col in cols.values():
        for i, r in enumerate(sorted(col, key=lambda x: -abs(x["mean_delta"])), 1):
            r["rank"] = i


def write_csv(records, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})
    print(f"wrote {path}", flush=True)


def record_index(records):
    return {(r["metric"], r["lead"], r["group"]): r for r in records}


# --- physics / RMSE aggregation --------------------------------------------------------
def is_physics(label):
    return not label.startswith("RMSE ")


def group_aggregates(records, spec_labels, groups, lead):
    """{group: {phys_share, rmse_share, phys_svr, rmse_svr}} - signed mean share,
    mean |SVR|, each averaged over the metric class (nan-safe)."""
    idx = record_index(records)
    phys = [l for l in spec_labels if is_physics(l)]
    rmse = [l for l in spec_labels if not is_physics(l)]
    out = {}
    for g in groups:
        def _vals(labels, key, absval=False):
            v = np.array([idx[(l, lead, g)][key] for l in labels if (l, lead, g) in idx],
                         dtype=np.float64)
            if absval:
                v = np.abs(v)
            return float(np.nanmean(v)) if v.size and not np.all(np.isnan(v)) else float("nan")
        out[g] = {"phys_share": _vals(phys, "share"),
                  "rmse_share": _vals(rmse, "share"),
                  "phys_svr": _vals(phys, "svr", absval=True),
                  "rmse_svr": _vals(rmse, "svr", absval=True)}
    return out


# --- heatmaps ---------------------------------------------------------------------------
def _matrix(records, spec_labels, groups, lead, key):
    idx = record_index(records)
    M = np.full((len(spec_labels), len(groups)), np.nan)
    S = np.zeros_like(M, dtype=bool)
    for i, l in enumerate(spec_labels):
        for j, g in enumerate(groups):
            r = idx.get((l, lead, g))
            if r is not None:
                M[i, j] = r[key]
                S[i, j] = r["significant"]
    return M, S


def _fmt_cell(v):
    if not np.isfinite(v):
        return ""
    return f"{v:.0f}" if abs(v) >= 100 else f"{v:.2g}"


def plot_value_heatmap(records, spec_labels, groups, lead, key, out_path, title,
                       vmax=None, hatch_nonsig=True):
    """Metrics x groups heatmap with the value printed in each cell (numbers, not
    just colour) and non-significant cells hatched."""
    M, S = _matrix(records, spec_labels, groups, lead, key)
    if vmax is None:
        finite = np.abs(M[np.isfinite(M)])
        vmax = float(np.percentile(finite, 95)) if finite.size else 1.0
        vmax = max(vmax, EPS)
    fig, ax = plt.subplots(figsize=(0.55 * len(groups) + 3.5, 0.28 * len(spec_labels) + 2))
    ax.imshow(np.clip(M, -vmax, vmax), aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    for i in range(len(spec_labels)):
        for j in range(len(groups)):
            if hatch_nonsig and not S[i, j]:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       hatch="///", edgecolor="0.6", lw=0))
            if np.isfinite(M[i, j]):
                strong = abs(M[i, j]) > 0.75 * vmax
                ax.text(j, i, _fmt_cell(M[i, j]), ha="center", va="center",
                        fontsize=5, color="w" if strong else "k")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=90, fontsize=7)
    ax.set_yticks(range(len(spec_labels)))
    ax.set_yticklabels(spec_labels, fontsize=7)
    ax.set_title(f"{title} @ {lead} h", fontsize=10)
    fig.colorbar(plt.cm.ScalarMappable(plt.Normalize(-vmax, vmax), "RdBu_r"),
                 ax=ax, shrink=0.7, label=key)
    save_fig(fig, os.path.dirname(out_path), os.path.basename(out_path))


def plot_rank_heatmap(records, spec_labels, groups, lead, out_path):
    M, _ = _matrix(records, spec_labels, groups, lead, "rank")
    fig, ax = plt.subplots(figsize=(0.55 * len(groups) + 3.5, 0.28 * len(spec_labels) + 2))
    ax.imshow(M, aspect="auto", cmap="viridis_r")
    for i in range(len(spec_labels)):
        for j in range(len(groups)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, int(M[i, j]), ha="center", va="center", fontsize=5, color="w")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=90, fontsize=7)
    ax.set_yticks(range(len(spec_labels)))
    ax.set_yticklabels(spec_labels, fontsize=7)
    ax.set_title(f"Sensitivity rank per metric @ {lead} h  (1 = most damaged)", fontsize=10)
    save_fig(fig, os.path.dirname(out_path), os.path.basename(out_path))


# --- metric-vs-lead panels ---------------------------------------------------------------
def _grid(n):
    c = math.ceil(math.sqrt(n))
    return math.ceil(n / c), c


def plot_family_panels(fp32_run, oat_runs, family, specs, groups, leads, block,
                       rec_idx, colors, out_dir):
    rows, cols = _grid(len(specs))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows), squeeze=False)
    flat = list(axes.flat)
    for ax, spec in zip(flat, specs):
        base = np.column_stack([spec_series(fp32_run, spec, lt) for lt in leads])
        bmean, blo, bhi = block_bootstrap(base, block=block)
        ax.plot(leads, bmean, color="k", lw=2, label="fp32")
        ax.fill_between(leads, blo, bhi, color="k", alpha=0.15)
        for g in groups:
            gv = np.column_stack([spec_series(oat_runs[g], spec, lt) for lt in leads])
            gmean, *_ = block_bootstrap(gv, block=block)
            ax.plot(leads, gmean, lw=1, color=colors[g], label=g)
            for j, lt in enumerate(leads):
                if rec_idx[(spec.label, lt, g)]["significant"]:
                    ax.plot(lt, gmean[j], "*", color=colors[g], ms=7)
        ax.set_title(spec.label, fontsize=9)
        ax.set_xlabel("lead (h)")
        ax.grid(True, alpha=0.3)
    for ax in flat[len(specs):]:
        ax.axis("off")
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=6, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"{family}: metric vs lead (* = paired sig vs fp32)")
    fig.tight_layout()
    save_fig(fig, out_dir, f"panel_{family.replace(' ', '_')}.png")


# --- composite / summary figures ----------------------------------------------------------
def plot_layer_summary(records, spec_labels, groups, lead, colors, out_dir):
    """Headline ranking: bar = mean share of full-quant damage over the physics metrics,
    dots = the individual physics metrics, diamond = mean over the RMSE metrics."""
    idx = record_index(records)
    phys = [l for l in spec_labels if is_physics(l)]
    rmse = [l for l in spec_labels if not is_physics(l)]
    norms = {idx[(l, lead, g)]["norm"] for l in spec_labels for g in groups
             if (l, lead, g) in idx}
    xlabel = ("mean share of full-quant damage" if norms == {"share"}
              else "mean normalised damage (mixed share/col-max!)")

    rows = []
    for g in groups:
        pv = np.array([idx[(l, lead, g)]["share"] for l in phys if (l, lead, g) in idx])
        rv = np.array([idx[(l, lead, g)]["share"] for l in rmse if (l, lead, g) in idx])
        rows.append((g, float(np.nanmean(pv)), pv, float(np.nanmean(rv))))
    rows.sort(key=lambda r: r[1])                       # largest at the top of the barh

    fig, ax = plt.subplots(figsize=(8, 0.42 * len(groups) + 2))
    y = np.arange(len(rows))
    for k, (g, pmean, pv, rmean) in enumerate(rows):
        ax.barh(k, pmean, color=colors[g], alpha=0.85, zorder=2)
        ax.plot(pv, np.full(pv.shape, k), "o", ms=3, color="0.25", alpha=0.6, zorder=3)
        ax.plot(rmean, k, "D", ms=6, mfc="w", mec="crimson", mew=1.5, zorder=4)
        ax.text(pmean, k, f" {pmean:.2f}", va="center", fontsize=7,
                ha="left" if pmean >= 0 else "right")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel(f"{xlabel} @ {lead} h")
    ax.set_title(f"Per-layer damage @ {lead} h - bar: mean over physics metrics, "
                 f"dots: individual physics metrics, ◇: mean over RMSE metrics", fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)
    save_fig(fig, out_dir, f"layer_summary_{lead}h.png")


def plot_physics_vs_rmse(records, spec_labels, groups, lead, colors, out_dir):
    """One point per layer group: mean |SVR| over RMSE metrics (x) vs over physics
    metrics (y). Points far above y=x are 'RMSE-blind' layers: they damage physical
    consistency without moving forecast error."""
    agg = group_aggregates(records, spec_labels, groups, lead)
    floor = 1e-2
    fig, ax = plt.subplots(figsize=(7, 6.5))
    for g in groups:
        x = max(agg[g]["rmse_svr"], floor)
        y = max(agg[g]["phys_svr"], floor)
        ax.plot(x, y, "o", ms=8, color=colors[g])
        ax.annotate(g, (x, y), textcoords="offset points", xytext=(5, 3), fontsize=7)
    lims = np.array(ax.get_xlim() + ax.get_ylim())
    lo, hi = max(floor / 2, lims.min() / 2), lims.max() * 2
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("mean |SVR| over RMSE metrics")
    ax.set_ylabel("mean |SVR| over physics metrics")
    ax.set_title(f"Physics vs RMSE damage per layer @ {lead} h", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    save_fig(fig, out_dir, f"physics_vs_rmse_{lead}h.png")


# --- dke_pert figures -----------------------------------------------------------------------
def plot_dke_pert_spectra(oat_runs, groups, full_run, leads, colors, out_dir):
    """Injected-KE saturation spectra: DKE_pert(k) / 2 E_bg(k) at 500 hPa. 1.0 means the
    group's quantisation noise has fully decorrelated the winds at that scale."""
    groups = [g for g in groups if oat_runs[g].has_pert]
    if not groups:
        return
    rows, cols = _grid(len(leads))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.8 * rows), squeeze=False)
    for ax, lead in zip(axes.flat, leads):
        for g in groups:
            run = oat_runs[g]
            wl = run.wavelength_km("dke_pert")
            dke = run.spectrum_mean("dke_pert", f"dke_dke_{DKE_PERT_LEVEL}_{{lt}}", lead)
            bg = run.spectrum_mean("dke_pert", f"dke_bg_{DKE_PERT_LEVEL}_{{lt}}", lead)
            ax.plot(wl, dke / (2.0 * bg), lw=1, color=colors[g], label=g)
        if full_run is not None and full_run.has_pert:
            wl = full_run.wavelength_km("dke_pert")
            dke = full_run.spectrum_mean("dke_pert", f"dke_dke_{DKE_PERT_LEVEL}_{{lt}}", lead)
            bg = full_run.spectrum_mean("dke_pert", f"dke_bg_{DKE_PERT_LEVEL}_{{lt}}", lead)
            ax.plot(wl, dke / (2.0 * bg), lw=2.5, color="k", label="full quant")
        ax.axhline(1.0, color="0.5", ls=":", lw=1)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set_xlabel("wavelength (km)")
        ax.set_ylabel("DKE_pert / 2·E_bg(fp32)")
        ax.set_title(f"+{lead} h", fontsize=9)
        ax.grid(True, which="both", alpha=0.3)
    for ax in list(axes.flat)[len(leads):]:
        ax.axis("off")
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=6, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"KE injected per quantised group vs FP32 forecast, {DKE_PERT_LEVEL} hPa "
                 f"(dotted line = full decorrelation)")
    fig.tight_layout()
    save_fig(fig, out_dir, "dke_pert_spectra.png")


def plot_dke_pert_band(oat_runs, groups, full_run, lead, block, colors, out_dir):
    """Ranked bars: sub-1000 km injected-KE saturation fraction per group with CIs."""
    fn = _dke_pert_band_series()
    rows = []
    for g in groups:
        if not oat_runs[g].has_pert:
            continue
        mean, lo, hi = block_bootstrap(fn(oat_runs[g], lead), block=block)
        rows.append((g, float(mean), float(lo), float(hi)))
    if not rows:
        return
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(groups) + 2))
    for k, (g, mean, lo, hi) in enumerate(rows):
        ax.barh(k, mean, xerr=[[mean - lo], [hi - mean]], color=colors[g],
                alpha=0.85, error_kw={"lw": 1, "ecolor": "0.3"})
        ax.text(hi, k, f" {mean:.3g}", va="center", fontsize=7)
    if full_run is not None and full_run.has_pert:
        fmean, *_ = block_bootstrap(fn(full_run, lead), block=block)
        ax.axvline(float(fmean), color="k", ls="--", lw=1.5,
                   label=f"full quant ({float(fmean):.3g})")
        ax.legend(fontsize=8, loc="lower right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlabel(f"injected-KE saturation fraction, <{DKE_PERT_WL_MAX_KM:.0f} km, "
                  f"{DKE_PERT_LEVEL} hPa (1 = fully decorrelated)")
    ax.set_title(f"Small-scale KE injected by each group alone @ {lead} h", fontsize=10)
    ax.grid(True, axis="x", alpha=0.3)
    save_fig(fig, out_dir, f"dke_pert_band_{lead}h.png")


# --- additivity check ------------------------------------------------------------------------
def plot_additivity(records, spec_labels, groups, lead, full_table, out_dir):
    """Sum of per-group damage vs the full-quant run's damage, per metric. Ratio > 1 =
    sub-additive interactions; < 1 = super-additive interactions and/or damage from the
    never-grouped layers (FiLM / embeddings, quantised in the full run but not OAT'd)."""
    idx = record_index(records)
    labels, ratios = [], []
    for l in spec_labels:
        f = full_table.get((l, lead))
        if f is None or abs(f) < EPS:
            continue
        s = sum(idx[(l, lead, g)]["mean_delta"] for g in groups if (l, lead, g) in idx)
        labels.append(l)
        ratios.append(s / f)
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(8, 0.3 * len(labels) + 2))
    y = np.arange(len(labels))
    ax.barh(y, ratios, color="tab:blue", alpha=0.8)
    for k, v in enumerate(ratios):
        ax.text(v, k, f" {v:.2f}", va="center", fontsize=7)
    ax.axvline(1.0, color="k", ls="--", lw=1, label="perfectly additive")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Σ per-group Δ / full-quant Δ")
    ax.set_title(f"OAT additivity @ {lead} h (shortfall = interactions + never-grouped "
                 f"FiLM/embedding layers)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)
    save_fig(fig, out_dir, f"additivity_{lead}h.png")


# --- per-dir driver -----------------------------------------------------------------------------
def _scheme_of(cfg_dir):
    m = re.match(r"ablations?_([A-Za-z0-9]+)(_|$)", os.path.basename(os.path.normpath(cfg_dir)))
    return m.group(1) if m else None


def analyse_dir(cfg_dir, leads_arg=None, norm_mode="auto"):
    name = os.path.basename(os.path.normpath(cfg_dir))
    out = ensure_dir(f"ablation_analysis_{name}")
    print(f"\n=== {name} -> {out}/ ===", flush=True)

    fp32_run = MetricsRun("fp32", torch.load(os.path.join(cfg_dir, "fp32_metrics.pt"),
                                             map_location="cpu", weights_only=False))
    oat = torch.load(os.path.join(cfg_dir, "oat_results.pt"),
                     map_location="cpu", weights_only=False)
    oat_runs = OrderedDict((g, MetricsRun(g, d)) for g, d in oat.items())
    groups = order_groups(oat_runs)
    for g, run in oat_runs.items():
        assert run.dates == fp32_run.dates, f"{g}: init keys differ from fp32 baseline"
        assert run.leads == fp32_run.leads, f"{g}: leads differ from fp32 baseline"

    leads = leads_arg or fp32_run.leads
    dates = recover_dates(fp32_run.dates)
    block = bootstrap_block(len(fp32_run.dates))
    print(f"groups={len(groups)}  inits={len(fp32_run.dates)}  leads={leads}  "
          f"block={block}  svr_method={pick_svr_method(dates)}", flush=True)

    first = next(iter(oat_runs.values()))
    specs = list(metric_registry(fp32_run))
    dp = dke_pert_spec(first)
    if dp is not None:
        specs.append(dp)
    specs = available_specs(fp32_run, first, specs, leads[0])
    labels = [s.label for s in specs]
    print(f"metrics ({len(specs)}): {labels}", flush=True)

    scheme = _scheme_of(cfg_dir)
    full_table = full_damage_table(scheme, specs, leads) if (
        norm_mode != "colmax" and scheme) else None
    full_run = None
    if os.path.exists(f"all_metrics_{scheme}.pt"):
        full_run = MetricsRun(scheme, torch.load(f"all_metrics_{scheme}.pt",
                                                 map_location="cpu", weights_only=False))

    recs = sensitivity_records(fp32_run, oat_runs, specs, groups, leads, dates, block,
                               full_table, norm_mode)
    add_ranks(recs)
    write_csv(recs, os.path.join(out, "sensitivity.csv"))
    rec_idx = record_index(recs)
    colors = group_color_map(groups)

    heatmap_labels = [l for l in labels
                      if not any(x in l for x in HEATMAP_EXCLUDE)]
    for lead in leads:
        plot_value_heatmap(recs, heatmap_labels, groups, lead, "share",
                           os.path.join(out, f"share_heatmap_{lead}h.png"),
                           "Share of full-quant damage", vmax=1.5)
        plot_value_heatmap(recs, heatmap_labels, groups, lead, "svr",
                           os.path.join(out, f"svr_heatmap_{lead}h.png"),
                           "SVR (Δ / deseasonalised FP32 IQR)")
        plot_rank_heatmap(recs, heatmap_labels, groups, lead,
                          os.path.join(out, f"rank_heatmap_{lead}h.png"))
        plot_layer_summary(recs, labels, groups, lead, colors, out)
        plot_physics_vs_rmse(recs, labels, groups, lead, colors, out)
        if dp is not None:
            plot_dke_pert_band(oat_runs, groups, full_run, lead, block, colors, out)
        if full_table is not None:
            plot_additivity(recs, labels, groups, lead, full_table, out)

    fams = defaultdict(list)
    for s in specs:
        fams[s.group or "derived"].append(s)
    for family, fam_specs in fams.items():
        plot_family_panels(fp32_run, oat_runs, family, fam_specs, groups, leads, block,
                           rec_idx, colors, out)
    if dp is not None:
        plot_dke_pert_spectra(oat_runs, groups, full_run, leads, colors, out)

    return {"scheme": scheme or name, "records": recs, "labels": labels,
            "groups": groups, "leads": leads}


# --- cross-scheme summary --------------------------------------------------------------------------
def cross_scheme(summaries, out):
    """Compare per-layer damage across quant schemes: is the sensitive layer set
    scheme-dependent? Uses each scheme's own share-of-full-damage normalisation so
    schemes with wildly different absolute damage (W4 vs W8) are comparable."""
    schemes = [s["scheme"] for s in summaries]
    groups = order_groups(set.intersection(*(set(s["groups"]) for s in summaries)))
    leads = sorted(set.intersection(*(set(s["leads"]) for s in summaries)))
    if not groups or not leads:
        print("cross-scheme: no common groups/leads - skipped", flush=True)
        return
    aggs = {s["scheme"]: {lt: group_aggregates(s["records"], s["labels"], groups, lt)
                          for lt in leads} for s in summaries}

    with open(os.path.join(out, "cross_scheme.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scheme", "group", "lead", "mean_physics_share", "mean_rmse_share",
                    "mean_physics_abs_svr", "mean_rmse_abs_svr"])
        for sc in schemes:
            for lt in leads:
                for g in groups:
                    a = aggs[sc][lt][g]
                    w.writerow([sc, g, lt, a["phys_share"], a["rmse_share"],
                                a["phys_svr"], a["rmse_svr"]])
    print(f"wrote {os.path.join(out, 'cross_scheme.csv')}", flush=True)

    # grouped bars: mean physics share per group x scheme, one panel per lead
    rows, cols = _grid(len(leads))
    fig, axes = plt.subplots(rows, cols, figsize=(0.5 * len(groups) * cols + 3, 4.2 * rows),
                             squeeze=False)
    width = 0.8 / len(schemes)
    scheme_colors = matplotlib.colormaps["tab10"](np.arange(len(schemes)))
    x = np.arange(len(groups))
    for ax, lt in zip(axes.flat, leads):
        for k, sc in enumerate(schemes):
            v = [aggs[sc][lt][g]["phys_share"] for g in groups]
            ax.bar(x + (k - (len(schemes) - 1) / 2) * width, v, width,
                   color=scheme_colors[k], label=sc)
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=90, fontsize=7)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel("mean physics share")
        ax.set_title(f"+{lt} h", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
    for ax in list(axes.flat)[len(leads):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Per-layer physics damage (share of each scheme's own full-quant damage)")
    fig.tight_layout()
    save_fig(fig, out, "cross_scheme_bars.png")

    # Spearman rank agreement between schemes' per-group damage vectors
    fig, axes = plt.subplots(2, len(leads), figsize=(2.6 * len(leads) + 1.5, 6),
                             squeeze=False)
    for col, lt in enumerate(leads):
        for row, key in enumerate(["phys_share", "rmse_share"]):
            X = np.column_stack([[aggs[sc][lt][g][key] for g in groups] for sc in schemes])
            C = spearman_matrix(X)
            ax = axes[row][col]
            ax.imshow(C, vmin=-1, vmax=1, cmap="RdBu_r")
            for i in range(len(schemes)):
                for j in range(len(schemes)):
                    ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=8)
            ax.set_xticks(range(len(schemes)))
            ax.set_xticklabels(schemes, fontsize=7, rotation=45)
            ax.set_yticks(range(len(schemes)))
            ax.set_yticklabels(schemes, fontsize=7)
            ax.set_title(f"{'physics' if row == 0 else 'RMSE'} @ +{lt} h", fontsize=9)
    fig.suptitle("Spearman agreement of per-layer damage rankings between schemes")
    fig.tight_layout()
    save_fig(fig, out, "rank_agreement.png")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="*",
                    help="ablation output dirs (default: ablation_*_sampled_* with oat_results.pt)")
    ap.add_argument("--leads", default=None, help="comma list, e.g. 24,120")
    ap.add_argument("--norm", choices=["auto", "colmax"], default="auto",
                    help="'auto': share-of-full-quant-damage where a full run exists, "
                         "per-column max elsewhere; 'colmax': force per-column max")
    a = ap.parse_args(argv)

    dirs = a.dirs or sorted(d for d in glob.glob("ablation_*")
                            if os.path.isdir(d)
                            and os.path.exists(os.path.join(d, "oat_results.pt")))
    if not dirs:
        ap.error("no ablation dirs found (looked for ablation_*/oat_results.pt)")
    leads = [int(x) for x in a.leads.split(",")] if a.leads else None

    summaries = [analyse_dir(d, leads, a.norm) for d in dirs]
    if len(summaries) >= 2:
        cross_scheme(summaries, ensure_dir("ablation_analysis_cross_scheme"))


if __name__ == "__main__":
    main()
