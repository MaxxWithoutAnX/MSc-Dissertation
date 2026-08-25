# figure_specs.py
import dataclasses
import gc
import glob
import os
import re
import typing


class Roots(typing.NamedTuple):
    base: str
    results: str

    def path(self, *parts):
        """A path relative to the project root, e.g. an ablation_analysis dir."""
        return os.path.join(self.base, *parts)

    def rpath(self, *parts):
        """A path inside the results dir."""
        return os.path.join(self.base, self.results, *parts)


@dataclasses.dataclass(frozen=True)
class FigureSpec:
    id: str
    number: int
    section: str        # "main" | "appendix"
    model: str          # "aurora" | "stormer" | "cross"
    build: typing.Callable      # (roots: dict[str, Roots], outdir: str) -> str | None
    title: str
    inputs: tuple = ()
    suffix: str = ""

    @property
    def filename(self):
        prefix = "fig" if self.section == "main" else "figA"
        return f"{prefix}{self.number:02d}{self.suffix}_{self.id}.png"


# --------------------------------------------------------------------------- builds

import figure_core as fc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABEL_FAMILY = "wind_balance"
LABEL_LEAD = 120


# ------------------------------------------------------------------ localisation

def _b_localisation_aurora(roots, outdir):
    r = roots["aurora"]
    src = r.path("ablation_analysis_ablations_W8A8", "sensitivity.csv")  # PLURAL 'ablations'
    if not os.path.exists(src):
        return None
    data = fc.prepare_localisation(src, lead=120)
    return fc.draw_localisation(data, outdir, "fig03_localisation_aurora.png",
                                "Aurora: distortion per layer group, by axis",
                                120)


def _b_localisation_stormer(roots, outdir):
    r = roots["stormer"]
    src = r.path("ablation_analysis_ablation_W8A8", "sensitivity.csv")  # SINGULAR 'ablation'
    if not os.path.exists(src):
        return None
    data = fc.prepare_localisation(src, lead=120)
    return fc.draw_localisation(data, outdir, "figA03_localisation_stormer.png",
                                "Stormer: distortion per layer group, by axis",
                                120)


# ------------------------------------------------------------------ damage removed

def _b_damage_removed_aurora(roots, outdir):
    r = roots["aurora"]
    data = fc.prepare_damage_removed(
        r.rpath("harness_damage_removed.csv"),
        labels_csv=r.rpath("harness_damage_removed_labels.csv"),
        labels_family=LABEL_FAMILY, labels_lead=LABEL_LEAD)
    return fc.draw_damage_removed(
        data, outdir, "fig04_damage_removed_aurora.png",
        "Protecting the physics-selected group removes most of the balance damage",
)


def _b_damage_removed_stormer(roots, outdir):
    r = roots["stormer"]
    data = fc.prepare_damage_removed(
        r.rpath("harness_damage_removed.csv"),
        labels_csv=r.rpath("stormer_damage_removed_labels.csv"),
        labels_family=LABEL_FAMILY, labels_lead=LABEL_LEAD)
    return fc.draw_damage_removed(
        data, outdir, "figA11_damage_removed_stormer.png",
        "Stormer: damage removed per configuration")


# ------------------------------------------------------------------ paired effects

def _b_paired_effects_aurora(roots, outdir):
    """ENDPOINT ORDER IS LOAD-BEARING: the composite CSV must come first. Falling back to
    the balance-only file scores the physics guide on one of the two axes it was trading
    off, which on Frontier A reverses the sign of the result."""
    r = roots["aurora"]
    data = fc.prepare_paired_effects([
        r.rpath("harness_results_composite.csv"),
        r.rpath("harness_results_paired.csv")],
        labels_csv=r.rpath("harness_results_composite_labels.csv"),
        labels_lead=LABEL_LEAD)
    return fc.draw_paired_effects(data, outdir, "figA01_paired_effects_aurora.png",
                                  "Aurora: paired effects at matched budget")


def _b_paired_effects_stormer(roots, outdir):
    r = roots["stormer"]
    data = fc.prepare_paired_effects([
        r.rpath("stormer_composite.csv"),
        r.rpath("stormer_paired_wind_balance.csv")],
        labels_csv=r.rpath("stormer_composite_labels.csv"),
        labels_lead=LABEL_LEAD)
    return fc.draw_paired_effects(data, outdir, "figA02_paired_effects_stormer.png",
                                  "Stormer: paired effects at matched budget")


# ------------------------------------------------------------------ axis decomposition

def _b_axis_decomposition_aurora(roots, outdir):
    r = roots["aurora"]
    data = fc.prepare_axis_decomposition(r.rpath("harness_results_paired.csv"),
                                         r.rpath("harness_results_paired_dry_air_mass.csv"))
    return fc.draw_axis_decomposition(
        data, outdir, "figA04_axis_decomposition_aurora.png",
        "Distortion decomposed into balance and conservation axes")


def _b_axis_decomposition_stormer(roots, outdir):
    r = roots["stormer"]
    data = fc.prepare_axis_decomposition(r.rpath("stormer_paired_wind_balance.csv"),
                                         r.rpath("stormer_paired_dry_air_mass.csv"))
    return fc.draw_axis_decomposition(
        data, outdir, "figA05_axis_decomposition_stormer.png",
        "Stormer: distortion decomposed into balance and conservation axes")


# ------------------------------------------------------------------ axis robustness

def _b_axis_robustness(roots, outdir):
    r = roots["aurora"]
    rows = fc.read_csv(r.rpath("harness_axis_variants.csv"))
    if not rows:
        print("  skip axis_robustness (no harness_axis_variants.csv)")
        return None
    by = {}
    for row in rows:
        by.setdefault(row["variant"], {})[row["tag"]] = float(row["value"])
    variants = [v for v in ("dry_only", "dry_and_negq", "multi_family") if v in by]
    pairs = [("W8A8_span1", "W8A8_rmse_span1"), ("W8A8_knee", "W8A8_rmse_knee")]
    series = []
    for (p, q), col in zip(pairs, (fc.C["physics"], fc.C["accent"])):
        vals = []
        for v in variants:
            a, b = by[v].get(p), by[v].get(q)
            vals.append(b / a if a and b and a > 0 else float("nan"))
        series.append((f"{p.replace('W8A8_', '')} vs {q.replace('W8A8_', '')}",
                       vals, None, col))
    span1 = [v for v in series[0][1] if v == v]      # NaN-safe: NaN != NaN
    rng = (f"{min(span1):.1f}x-{max(span1):.1f}x across definitions" if span1
           else "across definitions")
    return fc.draw_dotplot(
        variants, series, outdir, "figA06_axis_robustness.png",
        "Physics advantage under three consistency-axis definitions",
        "physics advantage  (x, log;  >1 = physics better)",
        logx=True, refline=1.0,
        annotate=f"span1 advantage ranges {rng}:\n"
                 "the sign and order of magnitude are invariant, the point estimate is not")


# ------------------------------------------------------------------ activation blindness

def _b_activation_blindness(roots, outdir):
    r = roots["aurora"]
    lead = 120
    tbl = fc.read_csv(r.path("activation_analysis", "activation_stats_table_28.csv")) or \
        fc.read_csv(r.path("activation_analysis", "activation_stats_table.csv"))
    if not tbl:
        return None
    try:
        from allocator import load_distortion_table
        d = load_distortion_table(
            {"W8A8": r.path("ablation_analysis_ablations_W8A8", "sensitivity.csv")},
            lead=lead, min_effect_frac=0.0)
    except Exception as e:                                   # pragma: no cover
        print(f"  skip activation_blindness ({e})")
        return None

    amap = {"encoder/misc": "film", "decoder/misc": "film",
            "encoder/attn": "enc0_attn", "encoder/mlp": "enc0_mlp",
            "decoder/attn": "dec0_attn", "decoder/mlp": "dec0_mlp"}
    agg = {}
    for row in tbl:
        g = amap.get(row["group"])
        if g:
            agg.setdefault(g, []).append(float(row["outlier_ratio"]))
    pts = [(g, float(np.median(v)), d[g]["W8A8"]["balance"])
           for g, v in agg.items() if g in d]
    if not pts:
        return None

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for g, o, b in pts:
        ax.scatter(o, max(b, 1e-3), s=90,
                   color=fc.C["warn"] if g == "film" else fc.C["physics"],
                   edgecolor="white", zorder=3)
        ax.annotate(g, (o, max(b, 1e-3)), textcoords="offset points", xytext=(7, 3),
                    fontsize=8.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("activation outlier ratio  (the standard PTQ diagnostic, log)")
    ax.set_ylabel(f"measured physics damage @{lead}h  (SVR, log)")
    ax.set_title("Activation outlier statistics versus physics damage, per layer group",
                 fontsize=12, fontweight="bold")
    ax.grid(True, which="both", ls="-", lw=0.4, color="#ececec")
    fc.style(ax)
    return fc.save(fig, outdir, "figA07_activation_blindness.png")


# ------------------------------------------------------------------ calibration drift

def _b_calibration_drift(roots, outdir):
    r = roots["aurora"]

    def load(s):
        return {row["layer"]: row for row in
                fc.read_csv(r.path("activation_analysis",
                                   f"activation_stats_table{s}.csv"))}
    a10, a28 = load("_10"), load("_28")
    common = sorted(set(a10) & set(a28))
    if not common:
        return None
    x = np.array([float(a10[l]["outlier_ratio"]) for l in common])
    y = np.array([float(a28[l]["outlier_ratio"]) for l in common])
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[ok], y[ok]
    rho = np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1]

    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    lim = [min(x.min(), y.min()) * 0.8, max(x.max(), y.max()) * 1.2]
    ax.plot(lim, lim, color="#444", lw=0.9, ls="--", zorder=1)
    ax.scatter(x, y, s=26, color=fc.C["physics"], alpha=0.55, edgecolor="none", zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("activation outlier ratio, 10-step rollout (log)")
    ax.set_ylabel("activation outlier ratio, 28-step rollout (log)")
    ax.set_title("Calibration statistics versus rollout depth",
                 fontsize=12, fontweight="bold")
    moved = int(np.sum(np.abs(y / x - 1) > 0.10))
    ax.text(0.04, 0.94, f"rank-$\\rho$ = {rho:.3f}\nmedian ratio = {np.median(y/x):.3f}\n"
                        f"{moved}/{len(x)} layers move >10%",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc"))
    ax.grid(True, which="both", ls="-", lw=0.4, color="#ececec")
    fc.style(ax)
    return fc.save(fig, outdir, "figA08_calibration_drift.png")


# ------------------------------------------------------------------ cross-model


def _b_replication_forest(roots, outdir):
    a, s = roots["aurora"], roots["stormer"]
    d = fc.prepare_replication_forest(a.rpath("harness_results_composite.csv"),
                                      s.rpath("stormer_composite.csv"),
                                      labels_csv_by_model={
                                          "Aurora": a.rpath(
                                              "harness_results_composite_labels.csv"),
                                          "Stormer": s.rpath(
                                              "stormer_composite_labels.csv")},
                                      labels_lead=LABEL_LEAD)
    if not d["labels"]:
        return None
    fig, ax = plt.subplots(figsize=(7.8, 0.44 * len(d["labels"]) + 2.2))
    y = np.arange(len(d["labels"]))
    col = [fc.C["aurora"] if m == "Aurora" else fc.C["stormer"] for m in d["model"]]
    if d.get("samples"):
        rng = np.random.default_rng(0)
        for i, c in zip(y, col):
            fc._draw_strips(ax, [d["samples"][i]], [i], c, width=0.34, rng=rng)
    ax.scatter(d["ratio"], y, s=60, zorder=5, color=col,
               edgecolor="white", linewidth=1.0)
    ax.axvline(1.0, color="#444", lw=0.9, ls="--")
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(fc.label_with_n(d["labels"], d.get("samples")), fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("physics-guided advantage on max(balance, conservation)  (log)\n"
                  ">1 = physics better")
    ax.set_title("Effect sizes on both architectures\n"
                 "Aurora: Swin U-Net + Perceiver, 0.25 deg   |   "
                 "Stormer: plain ViT + adaLN, 1.40625 deg", fontsize=10.5,
                 fontweight="bold")
    fc.style(ax)
    return fc.save(fig, outdir, "fig05_replication_forest.png")


def _b_concentration(roots, outdir):
    a, s = roots["aurora"], roots["stormer"]
    lead = 120
    curves = fc.prepare_concentration({
        "Aurora": a.path("ablation_analysis_ablations_W8A8", "sensitivity.csv"),
        "Stormer": s.path("ablation_analysis_ablation_W8A8", "sensitivity.csv"),
    }, lead=lead)
    if not curves:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for model, c in curves.items():
        if not c["x"]:
            continue
        ax.plot([0] + c["x"], [0] + c["y"], marker="o", ms=4,
                color=fc.C["aurora"] if model == "Aurora" else fc.C["stormer"],
                label=f"{model}  ({c['n_groups']} groups, top-1 = {c['top1']:.0%})")
    ax.plot([0, 1], [0, 1], color="#999", lw=0.9, ls="--", label="uniform (no localisation)")
    ax.set_xlabel("fraction of layer groups, ordered most-damaging first")
    ax.set_ylabel(f"cumulative share of balance distortion @{lead}h")
    ax.set_title("Cumulative share of physics damage by layer group, both models\n"
                 "the shape replicates even though the named groups do not", fontsize=11,
                 fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fc.style(ax)
    ax.grid(True, axis="y", ls="-", lw=0.4, color="#ececec")
    return fc.save(fig, outdir, "fig02_concentration.png")


_TRANSFER_AXES = (("balance", "balance"), ("conservation", "conservation"),
                  ("standard", "RMSE"))

def _b_surrogate_transfer(roots, outdir):
    a, s = roots["aurora"], roots["stormer"]
    sources = {
        "Aurora": ([a.path("harness_configs_corrected.csv"),
                    a.path("harness_configs_standard.csv")],
                   [a.rpath("harness_results.csv"), a.rpath("harness_axis_standard.csv")]),
        "Stormer": ([s.path("stormer_harness_configs.csv"),
                     s.path("stormer_harness_configs_standard.csv")],
                    [s.rpath("harness_results.csv"), s.rpath("harness_axis_standard.csv")]),
    }
    fig, axs = plt.subplots(1, len(_TRANSFER_AXES), figsize=(15.2, 5.4))
    drew = False
    for ax, (axis, label) in zip(axs, _TRANSFER_AXES):
        n_panel, resid = 0, []
        for model, (cfg, res) in sources.items():
            d = fc.prepare_transfer(cfg, res, axis, normalise=True)
            if len(d["tags"]) < 2:
                continue
            drew = True
            n_panel += 1
            colour = fc.C["aurora"] if model == "Aurora" else fc.C["stormer"]
            ax.scatter(d["predicted"], d["measured"], s=34, alpha=0.85,
                       color=colour, edgecolor="white", linewidth=0.6,
                       label=f"{model}  rho={d['rho']:.3f}, slope={d['slope']:.3f}  "
                             f"(n={d['n']})")
            resid.append((model, d["residual"], colour))
        if not n_panel:
            ax.text(0.5, 0.5, f"no {label} data\n(sidecar CSVs not generated)",
                    ha="center", va="center", fontsize=9, color=fc.C["neutral"],
                    transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            lims = ax.get_xlim()
            ax.plot(lims, lims, color="#999", lw=0.9, ls="--")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.legend(frameon=False, fontsize=7.5, loc="upper left")
            fc.draw_transfer_residual_inset(ax, resid)
        ax.set_xlabel("predicted damage, % of uniform floor\n(additive OAT surrogate)")
        ax.set_ylabel("measured damage, % of uniform floor\n(held-out 2021)")
        ax.set_title(label, fontsize=11, fontweight="bold")
        fc.style(ax)
    if not drew:
        plt.close(fig)
        return None
    fig.suptitle("Predicted versus measured damage as a share of uniform-floor damage, "
                 "both models\nslope 1 = calibrated in level;  slope < 1 = compressed",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fc.save(fig, outdir, "figA09_surrogate_transfer.png")


# ------------------------------------------------------------------ Stormer-specific

_STORMER_LEADS = (24, 72, 120, 168)


def _b_lead_robustness_stormer(roots, outdir):
    r = roots["stormer"]
    fam = fc._block_rows(fc.read_csv(r.rpath("stormer_lead_robustness.csv")), "2")
    desc = fc._block_rows(fc.read_csv(r.rpath("stormer_lead_by_pair.csv")), "2")
    if not fam:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    by_pair = {}
    for row in desc:
        by_pair.setdefault(row["pair"], {})[int(row["lead"])] = float(row["ratio"])
    tested = fam[0]["pair"]
    for pair, cells in sorted(by_pair.items()):
        if pair == tested:
            continue
        xs = sorted(cells)
        ax.plot(xs, [cells[x] for x in xs], color="#c9c9c9", lw=1.0, zorder=1)

    leads = [int(row["lead"]) for row in fam]
    val = np.array([float(row["ratio"]) for row in fam])
    ax.plot(leads, val, color=fc.C["physics"], lw=2.0, marker="o", zorder=3,
            label=f"{tested}  (declared family, m=4)")
    ax.axhline(1.0, color="#444", lw=0.9, ls="--")
    ax.set_yscale("log")
    ax.set_xticks(list(_STORMER_LEADS))
    ax.set_xlabel("forecast lead time (h)")
    ax.set_ylabel("physics advantage  (RMSE / physics distortion, log)")
    ax.set_title("Stormer: physics advantage by lead time\n"
                 "bold = the pre-declared pair; grey = the other pairs, descriptive",
                 fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fc.style(ax)
    return fc.save(fig, outdir, "figA10_lead_robustness_stormer.png")


def _knee_removed_pct(damage_removed_csv, tag="W8A8_knee"):
    data = fc.prepare_damage_removed(damage_removed_csv)
    if tag not in data["tags"]:
        return None
    return data["removed"][data["tags"].index(tag)]


def _b_adaln_bound(roots, outdir):
    r = roots["stormer"]
    margin = 5.0
    rows = [row for row in fc.read_csv(r.path("harness_results_adaln",
                                              "adaln_marginal.csv"))
            if row.get("endpoint") == "composite" and row.get("pct_of_floor")]
    if not rows:
        return None
    labels, vals = [], []
    for row in rows:
        labels.append(f"{row['lead']} h")
        vals.append(float(row["pct_of_floor"]))
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    ax.axvline(0.0, color="#444", lw=0.8, ls="--", zorder=1)
    ax.scatter(vals, y, s=52, color=fc.C["physics"], edgecolor="white", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    lo, hi = min(vals), max(vals)
    pad = max(0.25, 0.18 * (hi - lo))
    ax.set_xlim(min(-pad, lo - pad), hi + pad)
    ax.set_ylabel("lead time")
    ax.set_xlabel("share of total quantisation damage recovered (%)")
    knee_pct = _knee_removed_pct(r.rpath("harness_damage_removed.csv"))
    title = f"adaln recovers ~{max(vals):.1f}% of quantisation damage"
    if knee_pct is not None:
        title += f"; embed recovers {knee_pct:.0f}%"
    ax.set_title(title, fontsize=10.5, fontweight="bold")
    fc.style(ax)
    for lab, v in zip(labels, vals):
        flag = "" if abs(v) <= margin else "   <-- OUTSIDE the margin"
        print(f"      {lab:<8} {v:+.2f}%{flag}")
    if knee_pct is not None:
        print(f"      protecting embed alone removes {knee_pct:.1f}% by the same measure")
    return fc.save(fig, outdir, "fig07_adaln_bound.png")


# --------------------------------------------------------------------------- registry

REGISTRY: list = [
    # ---- main body
    FigureSpec(id="concentration", number=2, section="main", model="cross",
               build=_b_concentration,
               inputs=("ablation_analysis_ablations_W8A8/sensitivity.csv",
                       "ablation_analysis_ablation_W8A8/sensitivity.csv"),
               title="Cumulative share of physics damage by layer group, both models"),
    FigureSpec(id="localisation_aurora", number=3, section="main", model="aurora",
               build=_b_localisation_aurora,
               inputs=("ablation_analysis_ablations_W8A8/sensitivity.csv",),
               title="Aurora: distortion per layer group, by axis"),
    FigureSpec(id="damage_removed_aurora", number=4, section="main", model="aurora",
               build=_b_damage_removed_aurora,
               inputs=("harness_damage_removed.csv",
                       "harness_damage_removed_labels.csv"),
               title="Aurora: damage removed per configuration, balance axis"),
    FigureSpec(id="replication_forest", number=5, section="main", model="cross",
               build=_b_replication_forest,
               inputs=("harness_results_composite.csv",
                       "stormer_composite.csv",
                       "harness_results_composite_labels.csv",
                       "stormer_composite_labels.csv"),
               title="Effect sizes on both architectures"),
    FigureSpec(id="adaln_bound", number=7, section="main", model="stormer",
               build=_b_adaln_bound,
               inputs=("harness_results_adaln/adaln_marginal.csv",
                       "harness_damage_removed.csv"),
               title="Stormer: incremental effect of protecting adaln after embed"),

    # ---- appendix
    FigureSpec(id="paired_effects_aurora", number=1, section="appendix", model="aurora",
               build=_b_paired_effects_aurora,
               inputs=("harness_results_composite.csv",
                       "harness_results_paired.csv",
                       "harness_results_composite_labels.csv"),
               title="Aurora: paired effects at matched budget"),
    FigureSpec(id="paired_effects_stormer", number=2, section="appendix", model="stormer",
               build=_b_paired_effects_stormer,
               inputs=("stormer_composite.csv",
                       "stormer_paired_wind_balance.csv",
                       "stormer_composite_labels.csv"),
               title="Stormer: paired effects at matched budget"),
    FigureSpec(id="localisation_stormer", number=3, section="appendix", model="stormer",
               build=_b_localisation_stormer,
               inputs=("ablation_analysis_ablation_W8A8/sensitivity.csv",),
               title="Stormer: distortion per layer group, by axis"),
    FigureSpec(id="axis_decomposition_aurora", number=4, section="appendix", model="aurora",
               build=_b_axis_decomposition_aurora,
               inputs=("harness_results_paired.csv",
                       "harness_results_paired_dry_air_mass.csv"),
               title="Distortion decomposed into balance and conservation axes"),
    FigureSpec(id="axis_decomposition_stormer", number=5, section="appendix", model="stormer",
               build=_b_axis_decomposition_stormer,
               inputs=("stormer_paired_wind_balance.csv",
                       "stormer_paired_dry_air_mass.csv"),
               title="Stormer: distortion decomposed into balance and conservation axes"),
    FigureSpec(id="axis_robustness", number=6, section="appendix", model="aurora",
               build=_b_axis_robustness,
               inputs=("harness_axis_variants.csv",),
               title="Physics advantage under three consistency-axis definitions"),
    FigureSpec(id="activation_blindness", number=7, section="appendix", model="aurora",
               build=_b_activation_blindness,
               # the second entry is the FALLBACK the build tries when _28 is absent
               inputs=("activation_analysis/activation_stats_table_28.csv",
                       "activation_analysis/activation_stats_table.csv",
                       "ablation_analysis_ablations_W8A8/sensitivity.csv"),
               title="Activation outlier statistics versus physics damage, per layer group"),
    FigureSpec(id="calibration_drift", number=8, section="appendix", model="aurora",
               build=_b_calibration_drift,
               inputs=("activation_analysis/activation_stats_table_10.csv",
                       "activation_analysis/activation_stats_table_28.csv"),
               title="Calibration statistics versus rollout depth"),
    FigureSpec(id="surrogate_transfer", number=9, section="appendix", model="cross",
               build=_b_surrogate_transfer,
               inputs=("harness_configs_corrected.csv",
                       "harness_configs_standard.csv",
                       "harness_results_n48/harness_results.csv",
                       "harness_results_n48/harness_axis_standard.csv",
                       "stormer_harness_configs.csv",
                       "stormer_harness_configs_standard.csv",
                       "harness_results_n47/harness_results.csv",
                       "harness_results_n47/harness_axis_standard.csv"),
               title="Predicted versus measured damage as a share of uniform-floor damage",
),
    FigureSpec(id="lead_robustness_stormer", number=10, section="appendix", model="stormer",
               build=_b_lead_robustness_stormer,
               inputs=("stormer_lead_robustness.csv",
                       "stormer_lead_by_pair.csv"),
               title="Stormer: physics advantage by lead time"),
    FigureSpec(id="damage_removed_stormer", number=11, section="appendix", model="stormer",
               build=_b_damage_removed_stormer,
               inputs=("harness_damage_removed.csv",
                       "stormer_damage_removed_labels.csv"),
               title="Stormer: damage removed per configuration"),
]


import csv as _csv


RMSE_COMPRESSION_TITLE = "RMSE degradation and balance distortion versus lead time"

def _rmse_compression_figure(r, outdir, name, title):
    import torch

    import rmse_compression as rc
    pt_path = r.rpath("harness_results.pt")
    floored = r.rpath("harness_results_by_lead_floored.csv")
    gated = os.path.exists(floored)
    if not gated:
        floored = r.rpath("harness_results_by_lead.csv")
    if not (os.path.exists(pt_path) and os.path.exists(floored)):
        print(f"  skip {name} (need harness_results.pt + a by_lead CSV)")
        return None
    pt = torch.load(pt_path, map_location="cpu", mmap=True, weights_only=False)
    leads = [24, 72, 120, 168]
    with open(floored) as f:
        bal = {(row["tag"], int(row["lead"])): float(row["balance"])
               for row in _csv.DictReader(f)}
    spec = [("uniform W8A8 (floor)", "W8A8_floor", "#D55E00", "s"),
            ("physics-protected (span1)", "W8A8_span1", "#0072B2", "o"),
            ("bf16 ceiling", "ceiling", "#333333", "^")]
    series = []
    for lab, tag, col, mk in spec:
        if tag not in pt or (tag, leads[0]) not in bal:
            continue
        series.append((lab,
                       [rc.rmse_degradation(pt, tag, L) for L in leads],
                       [bal[(tag, L)] for L in leads], col, mk))
    del pt
    gc.collect()
    note = None
    xcsv = r.rpath("harness_rmse_crossing.csv")
    if os.path.exists(xcsv):
        by_lead = {int(row["lead"]): row for row in fc.read_csv(xcsv)}

        def _gap(row):
            return f"{float(row['diff_pp']):+.2f} pp"

        neg = sorted(L for L, row in by_lead.items() if float(row["diff_pp"]) < 0)
        if neg:
            note = f"({_gap(by_lead[neg[0]])})"
        elif by_lead:
            narrowest = min(by_lead, key=lambda L: abs(float(by_lead[L]["diff_pp"])))
            note = (f"the lines never cross, but by {narrowest} h the gap\n"
                    f"has narrowed to {_gap(by_lead[narrowest])}")
    series = fc.balance_as_percent_of_floor(series, floor_label=spec[0][0])
    per_init = fc.read_per_init(r.rpath("harness_rmse_lead_per_init.csv"))
    per_var = fc.read_per_init_var(r.rpath("harness_rmse_lead_per_init_var.csv"))
    bands, agreement, agreement_of = None, None, ("", "")
    if per_init:
        bands = {sp[0]: b for sp, b in
                 ((sp, fc.pooled_boot_ci(per_var, sp[1], leads)) for sp in spec) if b}
        agreement = fc.paired_sign_agreement(per_init, spec[0][1], spec[1][1], leads)
        if agreement:
            agreement_of = (spec[0][0], spec[1][0])
    return fc.draw_rmse_compression(leads, series, outdir, name, title,
                                    crossing_note=note, balance_percent=True,
                                    rmse_bands=bands, agreement=agreement,
                                    agreement_of=agreement_of)


def _b_rmse_compression(roots, outdir):
    """Aurora's fig01. Inputs: harness_results.pt, harness_results_by_lead_floored.csv
    (with harness_results_by_lead.csv as the ungated fallback), harness_rmse_crossing.csv."""
    return _rmse_compression_figure(roots["aurora"], outdir, "fig01_rmse_compression.png",
                                    RMSE_COMPRESSION_TITLE)


REGISTRY.append(
    FigureSpec(id="rmse_compression", number=1, section="main", model="aurora",
               build=_b_rmse_compression,
               inputs=("harness_results.pt", "harness_results_by_lead_floored.csv"),
               title=RMSE_COMPRESSION_TITLE,
))


# ------------------------------------------------------------------ lead divergence

LEAD_DIVERGENCE_TITLE = "Physics advantage versus lead time, both models"

LEAD_DIVERGENCE_TAIL = (
    "This trend is POST-HOC and "
    "hypothesis-generating: it was not among the comparisons named in advance and is NOT A "
    "REPLICATION. What replicates is the pre-declared comparison itself -- physics beats "
    "RMSE at all four leads on both models.")


def _slope_magnitude_sentence(data, models=("Aurora", "Stormer")):
    vals = {}
    for model in models:
        s = fc.parse_slope(data.get("slope_text", {}).get(model, ""))
        if not s:
            continue
        try:
            vals[model] = float(str(s[0]).replace("+", ""))
        except (TypeError, ValueError):
            continue
    if len(vals) < 2:
        return ""
    (big, bv), (small, sv) = sorted(vals.items(), key=lambda kv: -abs(kv[1]))
    if abs(sv) < 1e-9:
        return (f" {big}'s slope is the only one with any size to it ({bv:+.3f} against "
                f"{small}'s {sv:+.3f}).")
    return (f" {big}'s slope is {abs(bv) / abs(sv):.0f}x the size of {small}'s "
            f"({bv:+.3f} against {sv:+.3f}), so the contrast is between one slope worth "
            f"reading and one close to flat -- not between two trends of opposite sign.")


def _b_lead_divergence(roots, outdir):
    a, s = roots["aurora"], roots["stormer"]
    data = fc.prepare_lead_divergence({
        "Aurora": a.rpath("aurora_lead_robustness.csv"),
        "Stormer": s.rpath("stormer_lead_robustness.csv")})
    return fc.draw_lead_divergence(
        data, outdir, "fig06_lead_divergence.png",
        LEAD_DIVERGENCE_TITLE)


REGISTRY.append(
    FigureSpec(id="lead_divergence", number=6, section="main", model="cross",
               build=_b_lead_divergence,
               inputs=("aurora_lead_robustness.csv",
                       "stormer_lead_robustness.csv"),
               title=LEAD_DIVERGENCE_TITLE,
))


# ------------------------------------------------ endpoint choice sensitivity

ENDPOINT_FOREST_TITLE = "Effect sizes under two endpoint definitions"


def _b_endpoint_forest(roots, outdir):
    a, s = roots["aurora"], roots["stormer"]
    data = fc.prepare_endpoint_forest({
        "Aurora": {"balance": a.rpath("harness_results_paired.csv"),
                   "composite": a.rpath("harness_results_composite.csv")},
        "Stormer": {"balance": s.rpath("stormer_paired_wind_balance.csv"),
                    "composite": s.rpath("stormer_composite.csv")},
    })
    if not data["labels"]:
        return None
    return fc.draw_endpoint_forest(
        data, outdir, "figA13_endpoint_dependence.png", ENDPOINT_FOREST_TITLE,
        "balance", "composite",)


REGISTRY.append(
    FigureSpec(id="endpoint_dependence", number=13, section="appendix", model="cross",
               build=_b_endpoint_forest,
               inputs=("harness_results_paired.csv",
                       "harness_results_composite.csv",
                       "stormer_paired_wind_balance.csv",
                       "stormer_composite.csv"),
               title=ENDPOINT_FOREST_TITLE))


# ------------------------------------------------ surrogate validity condition

ADDITIVITY_TITLE = "Summed single-group effects versus the measured joint effect"


def _reversing_schemes(roots):
    a, s = roots["aurora"], roots["stormer"]
    data = fc.prepare_endpoint_forest({
        "Aurora": {"balance": a.rpath("harness_results_paired.csv"),
                   "composite": a.rpath("harness_results_composite.csv")},
        "Stormer": {"balance": s.rpath("stormer_paired_wind_balance.csv"),
                    "composite": s.rpath("stormer_composite.csv")}})
    out = set()
    for i in fc.endpoint_reversals(data, "balance", "composite"):
        tag = data["labels"][i].split()[-1]
        scheme = fc._scheme_of(tag)
        out.add((data["model"][i], "W4" if scheme == "W4W8" else scheme))
    return out


def _b_additivity(roots, outdir):
    rows = fc.prepare_additivity({"Aurora": roots["aurora"], "Stormer": roots["stormer"]})
    if not rows:
        return None
    fc.write_additivity_csv(rows, os.path.join(outdir, "additivity_by_scheme.csv"))
    broken = _reversing_schemes(roots)

    def cell(model, scheme, bucket):
        for r in rows:
            if (r["model"], r["scheme"], r["bucket"]) == (model, scheme, bucket):
                return r
        return None

    aw4, a8 = cell("Aurora", "W4", "physics"), cell("Aurora", "W8A8", "physics")
    sw4 = cell("Stormer", "W4", "physics")
    dropped = sum(r["n_dropped"] for r in rows)
    bits = []
    if aw4 and a8 and sw4:
        bits.append(
            "Aurora's W4 table is the only cell where the assumption fails: median ratio "
            "{aw4:.2f} with {ak}/{an} metrics inside the +/-20% band, against {a8:.2f} "
            "({a8k}/{a8n}) at W8A8 and {sw4:.2f} ({sk}/{sn}) for Stormer at W4. Aurora's W4W8 "
            "frontier is exactly where physics-guided allocation reverses sign (figA13); "
            "Stormer's W4W8, whose surrogate stays additive, does not reverse. Marked cells "
            "are ALL SIGN FLIPS -- pairs landing on opposite sides of 1.0 on the two "
            "endpoints.".format(
                aw4=aw4["median"], ak=aw4["n_ok"], an=aw4["n"],
                a8=a8["median"], a8k=a8["n_ok"], a8n=a8["n"],
                sw4=sw4["median"], sk=sw4["n_ok"], sn=sw4["n"]))
    bits.append(
        "Two further readings: the physics axes are more additive than RMSE in six of the "
        "seven scheme cells, so the OAT surrogate is better founded for physics-guided "
        "allocation than for the RMSE-guided baseline it is measured against; and ratios sit "
        "above 1 almost everywhere -- interactions are sub-additive, so the surrogate "
        "OVER-predicts damage and errs conservative. With figA09 (rank transfers, level does "
        "not) the surrogate is correct in RANK and conservative in LEVEL. "
        "THE SEVENTH CELL GOES THE OTHER WAY. At Stormer W4 the RMSE bucket is the more "
        "additive one on both measures -- median |log10 ratio| 0.0426 against physics' "
        "0.0501, and 6/6 metrics inside the band against 10/12 -- so 'consistently' would be "
        "an overstatement of the project's own data.")
    bits.append(
        "EACH ROW SHOWS ITS SAMPLE, not a summary alone: a box and whiskers where the cell "
        "has at least ten metrics and the points themselves where it does not. That "
        "distinction is load-bearing here. Stormer's W8 and W8A8 RMSE cells rest on a SINGLE "
        "metric each, which a median and an interquartile bar rendered exactly as "
        "authoritatively as Aurora's fourteen-metric physics cells. The physics rows are also "
        "pseudo-replicated -- Aurora's fourteen come from about five metric families -- "
        "though that is not what drives the result: re-aggregating family-first moves the "
        "physics medians by at most 0.02 (0.910 to 0.923 at Aurora W8, 1.123 to 1.128 at "
        "W8A8) and leaves every RMSE median unchanged. It does collapse the RMSE cells to "
        "n=1 throughout, which is why figA31 pools schemes rather than splitting them.")
    if dropped:
        bits.append(
            "{d} metric-rows had a non-positive ratio (OAT sum and full-quant delta of "
            "OPPOSITE SIGN) and are excluded: unplottable on a log axis, and a sign error is "
            "a different failure from a scale error.".format(d=dropped))
    return fc.draw_additivity(rows, outdir, "figA15_additivity.png", ADDITIVITY_TITLE,
                              broken=broken, boxes=True)


REGISTRY.append(
    FigureSpec(id="additivity", number=15, section="appendix", model="cross",
               build=_b_additivity,
               inputs=("ablation_analysis_*/additivity_120h.csv",
                       "harness_results_paired.csv",
                       "harness_results_composite.csv",
                       "stormer_paired_wind_balance.csv",
                       "stormer_composite.csv"),
               title=ADDITIVITY_TITLE))


# ------------------------------------------------ lead trend as a model property

LEAD_SLOPES_TITLE = "Lead-time slopes by configuration pair"


def _aurora_by_lead_sources(a):
    primary = a.rpath("harness_results_by_lead.csv")
    patch = a.path("harness_results_corrected", "harness_results_by_lead.csv")
    if not os.path.exists(primary) or not os.path.exists(patch):
        return [primary]
    seen = {}
    for row in fc.read_csv(primary):
        tag = row.get("tag", "")
        if not tag.endswith(fc.SUPERSEDED_SUFFIX):
            seen.setdefault(tag, set()).add(str(row.get("lead", "")))
    complete = seen and all(len(v) >= 4 for v in seen.values())
    if complete:
        return [primary]
    print(f"  lead_slopes: {os.path.basename(primary)} is missing leads -- patching with "
          f"harness_results_corrected/ (mixes that run's n; expected for the merged dir)")
    return [primary, patch]


def _b_lead_slopes(roots, outdir):
    a, s = roots["aurora"], roots["stormer"]
    data = fc.prepare_lead_slopes({
        "Aurora": _aurora_by_lead_sources(a),
        "Stormer": s.rpath("harness_results_by_lead.csv"),
    })
    if not any(data.values()):
        return None
    fc.write_lead_slopes_csv(data, os.path.join(outdir, "lead_slopes_by_pair.csv"))

    return fc.draw_lead_slopes(data, outdir, "figA14_lead_slopes_by_pair.png",
                               LEAD_SLOPES_TITLE,
)


REGISTRY.append(
    FigureSpec(id="lead_slopes_by_pair", number=14, section="appendix", model="cross",
               build=_b_lead_slopes,
               inputs=("harness_results_merged/harness_results_by_lead.csv",
                       "harness_results_2021_stormer/harness_results_by_lead.csv"),
               title=LEAD_SLOPES_TITLE))


COST_SCOPE_TAGS = ("ceiling", "W8A8_floor", "W8A8_knee", "W8A8_rmse_knee", "W8A8_span1",
                   "W8A8_rmse_span1", "W8A8_span2", "W8A8_span3", "W4W8_floor")


def _b_cost_scope(roots, outdir):
    r = roots["aurora"]
    pts = fc.prepare_cost_scope(r.rpath("harness_results.csv"), COST_SCOPE_TAGS)
    return fc.draw_cost_scope(
        pts, outdir, "figA12_cost_scope.png",
        "Aurora: measured model size and latency versus balance distortion")


REGISTRY.append(
    FigureSpec(id="cost_scope", number=12, section="appendix", model="aurora",
               build=_b_cost_scope,
               inputs=("harness_results.csv",),
               title="Aurora: measured model size and latency versus balance distortion",
))


# ------------------------------------------- RMSE vs lead across the config families

RMSE_LEAD_FAMILIES_TITLE = "RMSE degradation versus lead time, by configuration family"

RMSE_LEAD_ROLES = [
    ("floor", "uniform floor", "#D55E00", "s"),
    ("knee", "knee", "#E69F00", "D"),
    ("span1", "physics span1", "#0072B2", "o"),
    ("span2", "physics span2", "#4C9FD4", "^"),
    ("span3", "physics span3", "#9ECAE1", "v"),
]

RMSE_LEAD_GUIDED_ROLES = [
    ("rmse_knee", "RMSE knee", "#E69F00", "D"),
    ("rmse_span1", "RMSE span1", "#0072B2", "o"),
    ("rmse_span2", "RMSE span2", "#4C9FD4", "^"),
    ("rmse_span3", "RMSE span3", "#9ECAE1", "v"),
]

RMSE_LEAD_SCHEMES = [("W8A8", "W8A8"), ("W4W8", "W4W8")]

RMSE_LEAD_VARS = ["geopotential_500", "temperature_850", "2m_temperature",
                  "mean_sea_level_pressure", "specific_humidity_700",
                  "u_component_of_wind_500", "v_component_of_wind_500"]

RMSE_LEAD_LEADS = (24, 72, 120, 168)


def _rmse_lead_cells_for(pt, leads):
    ceil = fc.prepare_rmse_lead_families(pt, ["ceiling"], leads, RMSE_LEAD_VARS)
    cols = []
    for scheme, col_label in RMSE_LEAD_SCHEMES:
        tags = [f"{scheme}_{role}" for role, _, _, _ in RMSE_LEAD_ROLES]
        got = fc.prepare_rmse_lead_families(pt, tags, leads, RMSE_LEAD_VARS)
        series = [(label, got[f"{scheme}_{role}"], colour, marker)
                  for role, label, colour, marker in RMSE_LEAD_ROLES
                  if f"{scheme}_{role}" in got]
        if series and "ceiling" in ceil:
            series.append(("bf16 ceiling", ceil["ceiling"], "#333333", "x"))
        cols.append((col_label, series or None))
    return cols


def _per_init_spread(path, leads):
    if not os.path.exists(path):
        return {}
    acc = {}
    for row in fc.read_csv(path):
        acc.setdefault(row["tag"], {}).setdefault(int(row["lead"]), []).append(
            float(row["deg_pct"]))
    out = {}
    for tag, by_lead in acc.items():
        if any(L not in by_lead for L in leads):
            continue
        sizes = {len(by_lead[L]) for L in leads}
        if len(sizes) != 1:
            continue
        out[tag] = [by_lead[L] for L in leads]
    return out


def _rmse_lead_cells_from_csv(path, leads, guided=(), spread=None):
    by_tag = {}
    for row in fc.read_csv(path):
        by_tag.setdefault(row["tag"], {})[int(row["lead"])] = float(row["deg_pct"])

    def series_for(tag):
        d = by_tag.get(tag)
        if not d or any(L not in d for L in leads):
            return None
        return [d[L] for L in leads]

    ceil = series_for("ceiling")
    cols = []
    for scheme, col_label in RMSE_LEAD_SCHEMES:
        series = []
        # (role table, linestyle). Physics arms solid, RMSE-guided twins dashed.
        for table, ls in ((RMSE_LEAD_ROLES, "-"), (guided, "--")):
            for role, label, colour, marker in table:
                tag = f"{scheme}_{role}"
                vals = series_for(tag)
                if vals is None:
                    continue
                sample = spread.get(tag) if spread and role.endswith("span1") else None
                series.append((label, vals, colour, marker, ls, sample) if sample
                              else (label, vals, colour, marker, ls))
        if series and ceil is not None:
            series.append(("bf16 ceiling", ceil, "#333333", "x"))
        cols.append((col_label, series or None))
    return cols


def _b_rmse_lead_families(roots, outdir):
    import torch

    leads = list(RMSE_LEAD_LEADS)
    cells = []
    for key, row_label in (("aurora", "Aurora"), ("stormer", "Stormer")):
        r = roots.get(key)
        if not r:
            continue
        csv_path = r.rpath("harness_rmse_lead_families.csv")
        if os.path.exists(csv_path):
            cells.append((row_label, _rmse_lead_cells_from_csv(csv_path, leads)))
            continue
        p = r.rpath("harness_results.pt")
        if not os.path.exists(p):
            print(f"  rmse_lead_families: no {key} CSV or harness_results.pt, row omitted")
            continue
        print(f"  rmse_lead_families: {key} has no {os.path.basename(csv_path)}, falling "
              f"back to loading the .pt (run run_rmse_lead_families.py to avoid this)")
        pt = torch.load(p, map_location="cpu", mmap=True, weights_only=False)
        cells.append((row_label, _rmse_lead_cells_for(pt, leads)))
        del pt
    if not cells:
        print("  skip rmse_lead_families (no input for either model)")
        return None
    return fc.draw_rmse_lead_grid(
        cells, leads, outdir, "figA16_rmse_lead_families.png",
        RMSE_LEAD_FAMILIES_TITLE)


REGISTRY.append(
    FigureSpec(id="rmse_lead_families", number=16, section="appendix", model="cross",
               build=_b_rmse_lead_families,
               inputs=("harness_results.pt",),
               title=RMSE_LEAD_FAMILIES_TITLE,
))


RMSE_LEAD_GUIDED_TITLE = ("RMSE degradation versus lead time, physics-guided against "
                          "RMSE-guided")
def _b_rmse_lead_families_guided(roots, outdir):
    leads = list(RMSE_LEAD_LEADS)
    cells = []
    for key, row_label in (("aurora", "Aurora"), ("stormer", "Stormer")):
        r = roots.get(key)
        if not r:
            continue
        csv_path = r.rpath("harness_rmse_lead_families.csv")
        if not os.path.exists(csv_path):
            print(f"  rmse_lead_families_guided: no {key} CSV "
                  f"(run run_rmse_lead_families.py), row omitted")
            continue
        cells.append((row_label, _rmse_lead_cells_from_csv(
            csv_path, leads, guided=RMSE_LEAD_GUIDED_ROLES,
            spread=_per_init_spread(r.rpath("harness_rmse_lead_per_init.csv"), leads))))
    if not cells:
        print("  skip rmse_lead_families_guided (no CSV for either model)")
        return None
    return fc.draw_rmse_lead_grid(
        cells, leads, outdir, "figA16b_rmse_lead_families_guided.png",
        RMSE_LEAD_GUIDED_TITLE)


REGISTRY.append(
    FigureSpec(id="rmse_lead_families_guided", number=16, suffix="b", section="appendix",
               model="cross", build=_b_rmse_lead_families_guided,
               inputs=("harness_rmse_lead_families.csv",),
               title=RMSE_LEAD_GUIDED_TITLE,
))


CONCENTRATION_AXIS_TITLE = ("Damage concentration curve, by guide axis\n"
                            "(each curve normalised by its own total)")
CONCENTRATION_AXES = (("balance", "balance", "-", "o", "physics"),
                      ("conservation", "conservation", "-.", "^", "accent"),
                      ("standard", "RMSE", "--", "s", "rmse"))


def _b_concentration_by_axis(roots, outdir):
    a, s = roots["aurora"], roots["stormer"]
    lead = 120
    srcs = {"Aurora": a.path("ablation_analysis_ablations_W8A8", "sensitivity.csv"),
            "Stormer": s.path("ablation_analysis_ablation_W8A8", "sensitivity.csv")}
    curves = {axis: fc.prepare_concentration(srcs, lead=lead, axis=axis)
              for axis, _, _, _, _ in CONCENTRATION_AXES}
    models = [m for m in ("Aurora", "Stormer")
              if any(curves[axis].get(m) for axis, _, _, _, _ in CONCENTRATION_AXES)]
    if not models:
        return None

    fig, panels = plt.subplots(1, len(models), figsize=(5.7 * len(models), 5.0),
                               sharey=True, squeeze=False)
    for panel, model in zip(panels[0], models):
        panel.plot([0, 1], [0, 1], color="#999", lw=0.9, ls=":",
                   label="uniform (no localisation)")
        n_groups = 0
        for axis, axis_label, ls, marker, colour_key in CONCENTRATION_AXES:
            c = curves[axis].get(model)
            if not c or not c["x"]:
                continue
            n_groups = c["n_groups"]
            top = fc.concentration_top_group(srcs[model], lead=lead, axis=axis)
            panel.plot([0] + c["x"], [0] + c["y"], ls=ls, lw=1.7, marker=marker, ms=4.4,
                       color=fc.C[colour_key], mew=1.0, mec="white",
                       label=f"{axis_label}  (top-1 {c['top1']:.0%}, {top})")
        panel.set_xlim(-0.02, 1.02)
        panel.set_ylim(0, 1.05)
        panel.set_title(f"{model}   ({n_groups} layer groups)", fontsize=11,
                        fontweight="bold")
        panel.legend(frameon=False, fontsize=7.6, loc="lower right")
        fc.style(panel)
        panel.grid(True, axis="y", ls="-", lw=0.4, color="#ececec")
    panels[0][0].set_ylabel(f"cumulative share of that axis' distortion @{lead}h")
    fig.supxlabel("fraction of layer groups, ordered most-damaging first", fontsize=10)
    fig.suptitle(CONCENTRATION_AXIS_TITLE, fontsize=12, fontweight="bold", y=1.10)
    return fc.save(fig, outdir, "figA17_concentration_by_axis.png")


REGISTRY.append(
    FigureSpec(id="concentration_by_axis", number=17, section="appendix", model="cross",
               build=_b_concentration_by_axis,
               inputs=("ablation_analysis_ablations_W8A8/sensitivity.csv",
                       "ablation_analysis_ablation_W8A8/sensitivity.csv"),
               title=CONCENTRATION_AXIS_TITLE,
))


# --------------------------------------------------- transfer to un-optimised metrics

HELDOUT_TITLE = "Effect sizes on held-out metric families"

HELDOUT_TIERS = [
    ("held_out", "never guided, never scored", "#0072B2"),
    ("variant", "spectral re-parameterisations", "#4C9FD4"),
    ("partial", "in the guide, not in the endpoint", "#E69F00"),
    ("objective", "the reported endpoints", "#999999"),
]


def _b_heldout_families(roots, outdir):
    r = roots["aurora"]
    src = r.rpath("harness_heldout_families.csv")
    rows = fc.read_csv(src)
    if not rows:
        print("  skip heldout_families (no harness_heldout_families.csv -- "
              "run run_heldout_families.py)")
        return None

    by_tier = {}
    for row in rows:
        by_tier.setdefault(row["tier"], []).append(row)
    order = [(t, lab, col) for t, lab, col in HELDOUT_TIERS if by_tier.get(t)]
    labels = [row["family"] for t, _, _ in order for row in by_tier[t]]
    if not labels:
        return None

    series = []
    for tier, tier_label, colour in order:
        vals = []
        for t2, _, _ in order:
            for row in by_tier[t2]:
                vals.append(float(row["ratio"]) if t2 == tier else float("nan"))
        series.append((tier_label, vals, None, colour))

    return fc.draw_dotplot(
        labels, series, outdir, "figA18_heldout_families.png",
        HELDOUT_TITLE,
        "physics advantage on this family  (x, log;  >1 = physics less distorted)",
        logx=True, refline=1.0,)


REGISTRY.append(
    FigureSpec(id="heldout_families", number=18, section="appendix", model="aurora",
               build=_b_heldout_families,
               inputs=("harness_heldout_families.csv",),
               title=HELDOUT_TITLE,
))


# ------------------------------------------------------------------ damage plane

AXIS_DISPLACEMENT_TITLE = ("Displacement along the balance and conservation axes, by guide")


def _b_axis_displacement(roots, outdir):
    r = roots["aurora"]
    src = r.rpath("harness_results.csv")
    rows = [row for row in fc.read_csv(src)
            if row.get("floor") in ("W8A8", "bf16") and row.get("balance")]
    if not rows:
        print("  skip axis_displacement (no W8A8 rows in harness_results.csv)")
        return None
    by = {row["tag"]: (float(row["balance"]), float(row["conservation"])) for row in rows}

    FLOOR = 1e-4

    def xy(tag):
        v = by.get(tag)
        return (max(v[0], FLOOR), max(v[1], FLOOR)) if v else None

    def track(tags, colour, label, marker, offsets):
        """`offsets` is per-point and hand-set: the two knees land within a few percent of
        each other beside the floor marker, so a single shared offset stacks four labels on
        top of one another in the top-right corner."""
        pts = [(t, xy(t)) for t in tags]
        pts = [(t, p) for t, p in pts if p]
        if not pts:
            return
        ax.plot([p[0] for _, p in pts], [p[1] for _, p in pts],
                color=colour, lw=1.4, alpha=0.55, zorder=2)
        ax.scatter([p[0] for _, p in pts], [p[1] for _, p in pts], s=95, marker=marker,
                   color=colour, edgecolor="white", linewidth=1.4, zorder=4, label=label)
        for (t, p), off in zip(pts, offsets):
            ax.annotate(t.replace("W8A8_", ""), p, textcoords="offset points", xytext=off,
                        fontsize=7.5, color=colour, fontweight="bold")

    fig, ax = plt.subplots(figsize=(7.6, 6.2))

    rnd = [xy(f"W8A8_rand_{i}") for i in range(4)]
    rnd = [p for p in rnd if p]
    if rnd:
        ax.scatter([p[0] for p in rnd], [p[1] for p in rnd], s=58, facecolor="none",
                   edgecolor="#9a9a9a", linewidth=1.1, zorder=3,
                   label=f"random, matched cost (n={len(rnd)})")

    track(["W8A8_knee", "W8A8_span1", "W8A8_span2", "W8A8_span3"],
          fc.C["physics"], "physics-guided", "o",
          [(-14, 12), (9, -2), (9, -2), (9, -2)])
    track(["W8A8_rmse_knee", "W8A8_rmse_span1", "W8A8_rmse_span2", "W8A8_rmse_span3"],
          fc.C["rmse"], "RMSE-guided", "s",
          [(6, -14), (-30, -14), (7, 6), (7, 6)])

    pio = xy("W8A8_protect_io")
    if pio:
        ax.scatter(*pio, s=125, marker="D", color=fc.C["accent"], edgecolor="white",
                   linewidth=1.4, zorder=5, label="positional (protect first/last)")
        ax.annotate("protect_io", pio, textcoords="offset points", xytext=(9, -12),
                    fontsize=7.5, color="#333333")

    for tag, lab, mk in (("W8A8_floor", "uniform W8A8 floor", "X"),
                         ("ceiling", "bf16 ceiling", "*")):
        p = xy(tag)
        if p:
            ax.scatter(*p, s=170, marker=mk, color="#333333", zorder=5, label=lab)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("balance distortion @120 h  (SVR, log - lower is better)")
    ax.set_ylabel("conservation distortion @120 h  (SVR, log - lower is better)")
    ax.set_title(AXIS_DISPLACEMENT_TITLE + "\nboth guides march left; only the physics guide "
                 "also comes down", fontsize=11.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(True, which="both", ls="-", lw=0.4, color="#ececec")
    fc.style(ax)
    ax.grid(True, axis="y", ls="-", lw=0.4, color="#ececec")

    # The stall, measured rather than asserted, so a regenerated panel cannot contradict it.
    p3, r3 = by.get("W8A8_span3"), by.get("W8A8_rmse_span3")
    if p3 and r3:
        print(f"    span3: physics (bal {p3[0]:.3f}, cons {p3[1]:.4f})  vs  "
              f"RMSE (bal {r3[0]:.3f}, cons {r3[1]:.4f})  -- RMSE is "
              f"{p3[0]/r3[0]:.1f}x BETTER on balance and {r3[1]/p3[1]:.0f}x WORSE on "
              f"conservation")
    return fc.save(fig, outdir, "fig08_axis_displacement.png")


REGISTRY.append(
    FigureSpec(id="axis_displacement", number=8, section="main", model="aurora",
               build=_b_axis_displacement,
               inputs=("harness_results.csv",),
               title=AXIS_DISPLACEMENT_TITLE,
))


SPECTRAL_TEST_TITLE = "Power spectral density ratio versus wavenumber"
def _b_spectral_test(roots, outdir):
    r = roots["aurora"]
    data = fc.prepare_spectral_test(r.rpath("harness_spectral_psd.csv"),
                                    r.rpath("harness_spectral_amplification.csv"))
    return fc.draw_spectral_test(data, outdir, "figA19_spectral_test.png",
                                 SPECTRAL_TEST_TITLE)


REGISTRY.append(
    FigureSpec(id="spectral_test", number=19, section="appendix", model="aurora",
               build=_b_spectral_test,
               inputs=("harness_spectral_psd.csv",
                       "harness_spectral_amplification.csv"),
               title=SPECTRAL_TEST_TITLE,
))


ALLOCATION_FRONTIER_TITLE = "Measured balance distortion versus allocation budget"
def _b_allocation_frontier(roots, outdir):
    r = roots["aurora"]
    data = fc.prepare_allocation_frontier(r.rpath("harness_results.csv"), axis="balance")
    return fc.draw_allocation_frontier(data, outdir, "figA20_allocation_frontier.png",
                                       ALLOCATION_FRONTIER_TITLE)


REGISTRY.append(
    FigureSpec(id="allocation_frontier", number=20, section="appendix", model="aurora",
               build=_b_allocation_frontier,
               inputs=("harness_results.csv",),
               title=ALLOCATION_FRONTIER_TITLE,
))


BALANCE_PROFILE_TITLE = "Ageostrophic / geostrophic wind ratio by pressure level"
def _b_balance_profile(roots, outdir):
    r = roots["aurora"]
    series = fc.prepare_balance_profile(r.rpath("harness_balance_profile.csv"))
    return fc.draw_balance_profile(series, outdir, "figA21_balance_profile.png",
                                   BALANCE_PROFILE_TITLE)


REGISTRY.append(
    FigureSpec(id="balance_profile", number=21, section="appendix", model="aurora",
               build=_b_balance_profile,
               inputs=("harness_balance_profile.csv",),
               title=BALANCE_PROFILE_TITLE,
))


PER_VARIABLE_RHO_TITLE = "Rank correlation between per-variable RMSE and balance damage"
def _b_per_variable_rho(roots, outdir):
    r = roots["aurora"]
    rows = fc.prepare_per_variable_rho(
        r.rpath("harness_per_variable_rmse.csv"),
        r.rpath("harness_per_variable_rho_jackknife.csv"))
    return fc.draw_per_variable_rho(rows, outdir, "figA22_per_variable_rho.png",
                                    PER_VARIABLE_RHO_TITLE)


REGISTRY.append(
    FigureSpec(id="per_variable_rho", number=22, section="appendix", model="aurora",
               build=_b_per_variable_rho,
               inputs=("harness_per_variable_rmse.csv",
                       "harness_per_variable_rho_jackknife.csv"),
               title=PER_VARIABLE_RHO_TITLE,))


ANTICONTROL_TITLE = "Physics-guided versus over-funded RMSE-guided arms, by endpoint"
def _b_anticontrol(roots, outdir):
    r = roots["aurora"]
    rows = fc.prepare_anticontrol(
        r.rpath("harness_results_anticontrol.csv"),
        r.rpath("harness_results_anticontrol_composite.csv"),
        balance_labels_csv=r.rpath("harness_results_anticontrol_labels.csv"),
        composite_labels_csv=r.rpath(
            "harness_results_anticontrol_composite_labels.csv"),
        labels_family=LABEL_FAMILY, labels_lead=LABEL_LEAD)
    return fc.draw_anticontrol(rows, outdir, "figA24_anticontrol.png",
                               ANTICONTROL_TITLE)


REGISTRY.append(
    FigureSpec(id="anticontrol", number=24, section="appendix", model="aurora",
               build=_b_anticontrol,
               inputs=("harness_results_anticontrol.csv",
                       "harness_results_anticontrol_composite.csv",
                       "harness_results_anticontrol_labels.csv",
                       "harness_results_anticontrol_composite_labels.csv"),
               title=ANTICONTROL_TITLE,))


CROSS_QUANTISER_TITLE = "Aurora: share of balance damage per layer group, by quantisation scheme"
def _b_cross_quantiser(roots, outdir):
    r = roots["aurora"]
    data = fc.prepare_cross_quantiser(
        r.path("ablation_analysis_cross_scheme", "cross_scheme.csv"), lead=120)
    return fc.draw_cross_quantiser(data, outdir, "figA25_cross_quantiser.png",
                                   CROSS_QUANTISER_TITLE)


REGISTRY.append(
    FigureSpec(id="cross_quantiser", number=25, section="appendix", model="aurora",
               build=_b_cross_quantiser,
               inputs=("ablation_analysis_cross_scheme/cross_scheme.csv",),
               title=CROSS_QUANTISER_TITLE,))


# ------------------------------------------------- Stormer half of four figures

COST_SCOPE_STORMER_TITLE = "Stormer: measured model size and latency versus balance distortion"
def _b_cost_scope_stormer(roots, outdir):
    r = roots["stormer"]
    pts = fc.prepare_cost_scope(r.rpath("harness_results.csv"), COST_SCOPE_TAGS)
    return fc.draw_cost_scope(pts, outdir, "figA27_cost_scope_stormer.png",
                              COST_SCOPE_STORMER_TITLE)


REGISTRY.append(
    FigureSpec(id="cost_scope_stormer", number=27, section="appendix", model="stormer",
               build=_b_cost_scope_stormer, inputs=("harness_results.csv",),
               title=COST_SCOPE_STORMER_TITLE,))


ALLOCATION_FRONTIER_STORMER_TITLE = "Stormer: measured balance distortion versus allocation budget"
def _b_allocation_frontier_stormer(roots, outdir):
    """Stormer's FLOP-share / weight-byte allocation frontier, from harness_results.csv."""
    r = roots["stormer"]
    data = fc.prepare_allocation_frontier(r.rpath("harness_results.csv"), axis="balance")
    return fc.draw_allocation_frontier(data, outdir,
                                       "figA28_allocation_frontier_stormer.png",
                                       ALLOCATION_FRONTIER_STORMER_TITLE,
                                       legend_loc="lower left")


REGISTRY.append(
    FigureSpec(id="allocation_frontier_stormer", number=28, section="appendix",
               model="stormer", build=_b_allocation_frontier_stormer,
               inputs=("harness_results.csv",),
               title=ALLOCATION_FRONTIER_STORMER_TITLE,
))


ALLOCATION_FRONTIER_STORMER_BAL_TITLE = (
    "Stormer: measured balance distortion versus allocation cost, "
    "balance-scalarisation points only")
def _b_allocation_frontier_stormer_balance_arm(roots, outdir):
    r = roots["stormer"]
    data = fc.prepare_allocation_frontier(r.rpath("harness_results.csv"), axis="balance",
                                          exclude=("W8A8_span2",))
    return fc.draw_allocation_frontier(data, outdir,
                                       "figA28b_allocation_frontier_stormer_balance_arm.png",
                                       ALLOCATION_FRONTIER_STORMER_BAL_TITLE,
                                       legend_loc="lower left")


REGISTRY.append(
    FigureSpec(id="allocation_frontier_stormer_balance_arm", number=28, suffix="b",
               section="appendix", model="stormer",
               build=_b_allocation_frontier_stormer_balance_arm,
               inputs=("harness_results.csv",),
               title=ALLOCATION_FRONTIER_STORMER_BAL_TITLE,
))


CROSS_QUANTISER_STORMER_TITLE = "Stormer: share of balance damage per layer group, by quantisation scheme"
def _b_cross_quantiser_stormer(roots, outdir):
    """Does Stormer's localisation survive a change of quantiser? Reads the same
    ablation_analysis_cross_scheme/cross_scheme.csv layout Aurora uses -- identical header,
    Stormer's own eleven layer groups."""
    r = roots["stormer"]
    data = fc.prepare_cross_quantiser(
        r.path("ablation_analysis_cross_scheme", "cross_scheme.csv"), lead=120)
    return fc.draw_cross_quantiser(data, outdir, "figA29_cross_quantiser_stormer.png",
                                   CROSS_QUANTISER_STORMER_TITLE)


REGISTRY.append(
    FigureSpec(id="cross_quantiser_stormer", number=29, section="appendix", model="stormer",
               build=_b_cross_quantiser_stormer,
               inputs=("ablation_analysis_cross_scheme/cross_scheme.csv",),
               title=CROSS_QUANTISER_STORMER_TITLE,
))


RMSE_COMPRESSION_STORMER_TITLE = "Stormer: RMSE degradation and balance distortion versus lead time"
def _b_rmse_compression_stormer(roots, outdir):
    return _rmse_compression_figure(roots["stormer"], outdir,
                                    "figA30_rmse_compression_stormer.png",
                                    RMSE_COMPRESSION_STORMER_TITLE)


REGISTRY.append(
    FigureSpec(id="rmse_compression_stormer", number=30, section="appendix", model="stormer",
               build=_b_rmse_compression_stormer,
               inputs=("harness_results.pt", "harness_results_by_lead.csv",
                       "harness_rmse_crossing.csv"),
               title=RMSE_COMPRESSION_STORMER_TITLE,
))


# ------------------------------------------------ axis carrying the additivity

ADDITIVITY_BY_AXIS_TITLE = "Additivity of the summed single-group effects, by axis"


def _b_additivity_by_axis(roots, outdir):
    models = {"Aurora": roots["aurora"], "Stormer": roots["stormer"]}
    per_scheme = fc.prepare_additivity(models, buckets=fc.AXIS_BUCKETS, family_first=True)
    pooled = fc.prepare_additivity(models, buckets=fc.AXIS_BUCKETS, pooled=True,
                                   family_first=True)
    if not pooled:
        return None
    fc.write_additivity_csv(pooled + per_scheme,
                            os.path.join(outdir, "additivity_by_axis.csv"))

    cal = {(c["model"], c["bucket"]): c for c in fc.additivity_calibration(per_scheme)}
    contrasts = fc.additivity_bucket_contrast(per_scheme)

    bits = []
    for model in models:
        parts = []
        for bucket in ("balance", "conservation", "RMSE"):
            c = cal.get((model, bucket))
            if c:
                parts.append(f"{bucket} {c['median_abs_log10']:.3f} (n={c['n']})")
        if parts:
            bits.append(f"{model} calibration error, median |log10 ratio|: "
                        + "; ".join(parts) + ".")
    for c in contrasts:
        bits.append(
            "{m}: {a} minus {b} = {d:+.3f} log units, paired across {n} schemes.".format(
                m=c["model"], a=c["bucket_a"], b=c["bucket_b"],
                d=c["median_diff"], n=c["n_schemes"]))
    bits.append(
        "WEIGHTING, AND ITS LIMIT. Each family votes once (median across its labels), not "
        "each metric row: Aurora's 11 balance rows come from only 4 families, six of them "
        "one wind_balance diagnostic pair at three levels whose ratios are near copies, and "
        "counting them separately both outvotes the other families and halves every CI. But "
        "the same rule OVER-collapses RMSE, whose 6 rows are 6 genuinely distinct variables "
        "(Z500, T850, Q700, U500, MSLP, T2M) reduced to a single vote because they share one "
        "family. RMSE's true calibration error therefore lies between the two weightings "
        "-- 0.125 raw-row against 0.206 family-first on Aurora -- and the balance-versus-RMSE "
        "gap should be read as bounded, not as a point estimate. What is robust to the "
        "choice is that BALANCE is the most additive axis on both models; the "
        "conservation-versus-RMSE ordering is NOT, and reverses between weightings on "
        "Stormer.")
    bits.append(
        "WHAT THE SPREAD MARKS ARE, AND WHY MOST ARE NOT BOXES. Each cell shows the sample "
        "of per-(scheme, family) additivity ratios behind its estimate; the thick tick is "
        "the median. A cell with at least {k} values is drawn as a box (IQR) with 1.5-IQR "
        "whiskers; every smaller cell is drawn as the RAW POINTS instead. That is not a "
        "stylistic choice -- only ONE of the six pooled cells (Aurora balance, n=14) reaches "
        "the threshold; the others hold 9, 7, 4, 4 and 3 values. Quartiles computed from "
        "four numbers render exactly as authoritative as quartiles from forty, so drawing "
        "them as boxes would manufacture precision this design does not have. The honest "
        "summary of this figure is that the additivity sample is too small for boxplots, "
        "and the points say so on the page. Note also that the spread is ACROSS FAMILIES "
        "AND SCHEMES, not uncertainty on the median: a wide cell means the families "
        "disagree about how additive the axis is, which is a different claim from an "
        "imprecise estimate.".format(k=fc.BOX_MIN_N))
    bits.append(
        "Bold rows pool every scheme's RAW ratios; faint rows are the individual schemes and "
        "are context, not independent estimates. Conservation contributes 3 metric rows per "
        "scheme against balance's 11, which is why the pooled row is the one to read. "
        "Paired contrasts carry a bootstrap CI and NO p-value: at 4 schemes a signed-rank "
        "test is not powered to reject, and this is descriptive rather than a declared "
        "family.")

    ordered, faint = [], set()
    for model in models:
        ordered += [r for r in pooled if r["model"] == model]
        for r in per_scheme:
            if r["model"] != model:
                continue
            ordered.append(dict(r, scheme=f"  {r['scheme']}"))
            faint.add((model, f"  {r['scheme']}"))
    return fc.draw_additivity(ordered, outdir, "figA31_additivity_by_axis.png",
                              ADDITIVITY_BY_AXIS_TITLE,
                              faint=faint, wide=True, boxes=True)


REGISTRY.append(
    FigureSpec(id="additivity_by_axis", number=31, section="appendix", model="cross",
               build=_b_additivity_by_axis,
               inputs=("ablation_analysis_*/additivity_120h.csv",),
               title=ADDITIVITY_BY_AXIS_TITLE))


# ------------------------------------------------- the reproducibility floor

def _scheme_csvs(root):
    out = {}
    for pattern in ("ablation_analysis_ablation_*", "ablation_analysis_ablations_*"):
        for d in sorted(glob.glob(root.path(pattern))):
            csvp = os.path.join(d, "sensitivity.csv")
            if not os.path.exists(csvp):
                continue
            scheme = re.sub(r"^ablation_analysis_ablations?_", "", os.path.basename(d))
            out.setdefault(scheme, csvp)
    return out


REPRODUCIBILITY_FLOOR_TITLE = (
    "Aurora: layer-group effect sizes against the measured reproducibility floor")


def _b_reproducibility_floor(roots, outdir):
    r = roots["aurora"]
    data = fc.prepare_reproducibility_floor(
        r.path("noise_floor_ens_n48", "axis_null_p95.csv"),
        _scheme_csvs(r), lead=120)
    return fc.draw_reproducibility_floor(
        data, outdir, "figA23_reproducibility_floor.png", REPRODUCIBILITY_FLOOR_TITLE)


REGISTRY.append(
    FigureSpec(id="reproducibility_floor", number=23, section="appendix", model="aurora",
               build=_b_reproducibility_floor,
               inputs=("noise_floor_ens_n48/axis_null_p95.csv",
                       "ablation_analysis_ablations_*/sensitivity.csv"),
               title=REPRODUCIBILITY_FLOOR_TITLE))


REPRODUCIBILITY_FLOOR_STORMER_TITLE = (
    "Stormer: layer-group effect sizes against the measured reproducibility floor")


def _b_reproducibility_floor_stormer(roots, outdir):
    r = roots["stormer"]
    data = fc.prepare_reproducibility_floor(
        r.path("stormer_noise_floor_ens", "axis_null_p95.csv"),
        _scheme_csvs(r), lead=120)
    return fc.draw_reproducibility_floor(
        data, outdir, "figA26_reproducibility_floor_stormer.png",
        REPRODUCIBILITY_FLOOR_STORMER_TITLE)


REGISTRY.append(
    FigureSpec(id="reproducibility_floor_stormer", number=26, section="appendix",
               model="stormer", build=_b_reproducibility_floor_stormer,
               inputs=("stormer_noise_floor_ens/axis_null_p95.csv",
                       "ablation_analysis_ablation_*/sensitivity.csv"),
               title=REPRODUCIBILITY_FLOOR_STORMER_TITLE))


RMSE_SCORECARD_TITLE = ("Percent change in RMSE versus FP32 by variable, level "
                        "and lead time")
PHYSICS_SCORECARD_TITLE = ("Percent change in physical-consistency diagnostics versus "
                           "FP32 by metric and lead time")


def _b_rmse_scorecard(roots, outdir):
    """The AIWP-conventional scorecard: rows are (variable, level), one metric."""
    r = roots["aurora"]
    data = fc.prepare_scorecard(r.rpath("scorecard.csv"), "rmse")
    return fc.draw_scorecard(data, outdir, "fig09_rmse_scorecard.png",
                             RMSE_SCORECARD_TITLE)


REGISTRY.append(
    FigureSpec(id="rmse_scorecard", number=9, section="main", model="aurora",
               build=_b_rmse_scorecard, inputs=("scorecard.csv",),
               title=RMSE_SCORECARD_TITLE))


def _b_physics_scorecard(roots, outdir):
    """fig09's counterpart on the physics diagnostics, same schemes and leads."""
    r = roots["aurora"]
    data = fc.prepare_scorecard(r.rpath("scorecard.csv"), "physics")
    return fc.draw_scorecard(data, outdir, "fig10_physics_scorecard.png",
                             PHYSICS_SCORECARD_TITLE)


REGISTRY.append(
    FigureSpec(id="physics_scorecard", number=10, section="main", model="aurora",
               build=_b_physics_scorecard, inputs=("scorecard.csv",),
               title=PHYSICS_SCORECARD_TITLE))


RMSE_SCORECARD_STORMER_TITLE = ("Stormer: percent change in RMSE versus FP32 by "
                                "variable, level and lead time")
PHYSICS_SCORECARD_STORMER_TITLE = ("Stormer: percent change in physical-consistency "
                                   "diagnostics versus FP32 by metric and lead time")


def _b_rmse_scorecard_stormer(roots, outdir):
    """fig09's twin, on the same rows so the two can be read side by side."""
    r = roots["stormer"]
    data = fc.prepare_scorecard(r.rpath("scorecard.csv"), "rmse")
    return fc.draw_scorecard(data, outdir, "figA32_rmse_scorecard_stormer.png",
                             RMSE_SCORECARD_STORMER_TITLE)


REGISTRY.append(
    FigureSpec(id="rmse_scorecard_stormer", number=32, section="appendix",
               model="stormer", build=_b_rmse_scorecard_stormer,
               inputs=("scorecard.csv",), title=RMSE_SCORECARD_STORMER_TITLE))


def _b_physics_scorecard_stormer(roots, outdir):
    """fig10's twin. Every cell is a change against that model's OWN FP32 run, so the"""
    r = roots["stormer"]
    data = fc.prepare_scorecard(r.rpath("scorecard.csv"), "physics")
    return fc.draw_scorecard(data, outdir, "figA33_physics_scorecard_stormer.png",
                             PHYSICS_SCORECARD_STORMER_TITLE)


REGISTRY.append(
    FigureSpec(id="physics_scorecard_stormer", number=33, section="appendix",
               model="stormer", build=_b_physics_scorecard_stormer,
               inputs=("scorecard.csv",), title=PHYSICS_SCORECARD_STORMER_TITLE))


def specs_for(section=None, model=None):
    return [s for s in REGISTRY
            if (section is None or s.section == section)
            and (model is None or s.model == model)]
