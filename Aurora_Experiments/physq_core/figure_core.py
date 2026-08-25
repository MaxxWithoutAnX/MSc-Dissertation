# figure_core.py
import csv
import math
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import numpy as np

# Okabe-Ito derived; distinguishable in greyscale and to the common colour-vision deficiencies.
C = {"physics": "#0072B2", "rmse": "#D55E00", "neutral": "#666666",
     "accent": "#009E73", "warn": "#CC79A7", "aurora": "#0072B2", "stormer": "#E69F00"}

COMPOSITE_ENDPOINT = "pre-declared endpoint  max(balance, conservation)"


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(l for l in f if not l.startswith("#")))


def read_label_samples(path, key_field, value_field, family=None, lead=None):
    if not path or not os.path.exists(path):
        return {}
    rows = read_csv(path)
    wanted = {"family": None if family is None else str(family),
              "lead": None if lead is None else str(lead)}
    out, seen = {}, {}
    for r in rows:
        skip = False
        for col, want in wanted.items():
            if col not in r:
                continue
            if want is not None and str(r[col]) != want:
                skip = True
        if skip:
            continue
        try:
            v = _safe_float(r[value_field])
        except (KeyError, TypeError):
            continue
        if v is None:
            continue
        k = r[key_field]
        out.setdefault(k, []).append(v)
        for col in wanted:
            if col in r:
                seen.setdefault(k, {}).setdefault(col, set()).add(r[col])
    for k, cols in seen.items():
        for col, vals in cols.items():
            if wanted[col] is None and len(vals) > 1:
                pass
    return out


def samples_for(by_key, keys):
    if not by_key:
        return None
    return [by_key.get(k, []) for k in keys]


def read_per_init(path):
    out = {}
    for r in read_csv(path):
        try:
            out.setdefault((r["tag"], int(r["lead"])), {})[r["init"]] = float(r["deg_pct"])
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _init_order(d):
    """Init keys in chronological order -- block resampling assumes time order."""
    try:
        return sorted(d, key=int)
    except (TypeError, ValueError):
        return sorted(d)


def per_init_boot_ci(per_init, tag, leads, n_boot=2000, ci=95):
    """Moving-block bootstrap CI on the MEAN per-init degradation, one per lead.
    Same estimator as the plots_*/ RMSE bands (plot_common.block_bootstrap).
    Based off GraphCast paper reporting RMSE spread."""
    from plot_common import block_bootstrap
    out = []
    for L in leads:
        d = per_init.get((tag, L)) or {}
        if len(d) < 4:
            return None
        vals = np.asarray([d[k] for k in _init_order(d)], dtype=np.float64)
        block = max(1, round(len(vals) ** (1.0 / 3.0)))
        _, lo, hi = block_bootstrap(vals, block=block, n_boot=n_boot, ci=ci)
        out.append((float(lo), float(hi)))
    return out


def per_init_iqr(per_init, tag, leads):
    out = []
    for L in leads:
        vals = list((per_init.get((tag, L)) or {}).values())
        if len(vals) < 4:
            return None
        q1, _, q3 = np.percentile(vals, [25, 50, 75])
        out.append((float(q1), float(q3)))
    return out


def paired_sign_agreement(per_init, tag_a, tag_b, leads):
    out = []
    for L in leads:
        a, b = per_init.get((tag_a, L)) or {}, per_init.get((tag_b, L)) or {}
        shared = sorted(set(a) & set(b))
        if not shared:
            return None
        out.append((sum(1 for k in shared if a[k] > b[k]), len(shared)))
    return out


def label_with_n(labels, samples):
    if not samples:
        return list(labels)
    return [f"{lab}  n={len(sm)}" if sm else lab for lab, sm in zip(labels, samples)]


def save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, name)
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}")
    return p


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis="x", ls="-", lw=0.4, color="#ececec")
    ax.set_axisbelow(True)


def _truthy(v):
    return str(v).strip().lower() in ("true", "1", "yes")


def _block_rows(rows, block):
    return [r for r in rows if str(r.get("block", block)) in ("", str(block))]


# ------------------------------------------------------------------ localisation

def prepare_localisation(sensitivity_csv, lead=120, scheme="W8A8", min_effect_frac=0.0):
    from allocator import load_distortion_table
    d = load_distortion_table({scheme: sensitivity_csv}, lead=lead,
                              min_effect_frac=min_effect_frac)
    groups = sorted(d, key=lambda g: -d[g][scheme]["balance"])
    try:
        std = load_distortion_table({scheme: sensitivity_csv}, lead=lead,
                                    axes=("standard",), min_effect_frac=min_effect_frac)
        rmse = [std[g][scheme]["standard"] if g in std else None for g in groups]
        if all(v is None for v in rmse):
            rmse = None
    except Exception as e:                                          # pragma: no cover
        print(f"  localisation: no standard axis ({e}); drawing consistency axes only")
        rmse = None
    return {"groups": groups,
            "balance": [d[g][scheme]["balance"] for g in groups],
            "conservation": [d[g][scheme]["conservation"] for g in groups],
            "rmse": rmse}


def _rank_divergence_note(data):
    rmse = data.get("rmse")
    if not rmse or any(v is None for v in rmse):
        return None
    groups = data["groups"]
    # `groups` is already balance-descending, so its index IS the balance rank.
    order = sorted(range(len(groups)), key=lambda i: -rmse[i])
    rmse_rank = {i: r + 1 for r, i in enumerate(order)}
    worst = 0                       # the group RMSE most under-ranks
    for i in range(len(groups)):
        if rmse_rank[i] - (i + 1) > rmse_rank[worst] - (worst + 1):
            worst = i
    top_rmse = order[0]
    if rmse_rank[worst] - (worst + 1) <= 0:
        return None
    return ("RMSE and the physics axes disagree on the group that matters: "
            f"{groups[worst]} is rank {worst + 1} of {len(groups)} on balance but rank "
            f"{rmse_rank[worst]} on RMSE, while RMSE's own top pick ({groups[top_rmse]}) "
            f"is rank {top_rmse + 1} on balance.")


CAPTIONS_ON_FIGURE = False


def _cap(caption):
    return caption if (CAPTIONS_ON_FIGURE and caption) else ""


def draw_localisation(data, outdir, name, title, lead=120):
    if not data["groups"]:
        print(f"  skipped {name} (no groups in prepared data)")
        return None
    series = [("wind balance", data["balance"], None, C["physics"]),
              ("conservation", data["conservation"], None, C["accent"])]
    if data.get("rmse"):
        series.append(("RMSE (standard axis)", data["rmse"], None, C["rmse"]))
    return draw_dotplot(
        data["groups"], series, outdir, name, title,
        f"distortion @{lead}h  (SVR, log; higher = more damage)",
        logx=True, refline=1.0, annotate=_rank_divergence_note(data))


# ------------------------------------------------------------------ damage removed

def prepare_damage_removed(csv_path, block="2", labels_csv=None,
                           labels_family=None, labels_lead=None):
    rows = _block_rows(read_csv(csv_path), block)
    rows = sorted(rows, key=lambda r: float(r["removed_pct"]))
    tags = [r["tag"] for r in rows]
    return {"tags": tags,
            "removed": [float(r["removed_pct"]) for r in rows],
            "samples": samples_for(
                read_label_samples(labels_csv, "tag", "removed_pct",
                                   family=labels_family, lead=labels_lead), tags),
            "is_rmse": [("rmse" in t or "probe" in t) for t in tags]}


def draw_damage_removed(data, outdir, name, title, caption=None):
    if not data["tags"]:
        print(f"  skipped {name} (no tags in prepared data)")
        return None
    v = np.array(data["removed"])
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(v) + 1.8))
    ax.barh(np.arange(len(v)), v,
            color=[C["rmse"] if r else C["physics"] for r in data["is_rmse"]])
    _draw_strips(ax, data.get("samples"), np.arange(len(v)), "#333", width=0.42)
    import matplotlib.patches as mpatches
    ax.legend(handles=[mpatches.Patch(color=C["physics"], label="physics-guided"),
                       mpatches.Patch(color=C["rmse"], label="RMSE-guided / probe")],
              frameon=False, fontsize=8.5, loc="lower right")
    ax.set_yticks(np.arange(len(v)))
    ax.set_yticklabels(data["tags"], fontsize=8)
    ax.set_xlabel("physics damage removed vs the uniform floor  (%)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    style(ax)
    if caption:
        frac = min(0.34, 0.055 * (1 + len(caption) // 150))
        fig.tight_layout(rect=[0, frac, 1, 1])
        fig.text(0.5, frac * 0.88, caption, ha="center", va="top", fontsize=8.5, wrap=True)
    return save(fig, outdir, name)


# ------------------------------------------------------------------ paired effects

def prepare_paired_effects(csv_paths, block="2", labels_csv=None,
                           labels_family=None, labels_lead=None,
                           labels_value_field="contribution"):
    for i, p in enumerate(csv_paths):
        rows = _block_rows(read_csv(p), block)
        if rows:
            pairs = [r["pair"] for r in rows]
            return {"pairs": pairs,
                    "ratio": [float(r["ratio"]) for r in rows],
                    "samples": samples_for(
                        read_label_samples(labels_csv, "pair", labels_value_field,
                                           family=labels_family, lead=labels_lead), pairs),
                    "endpoint": COMPOSITE_ENDPOINT if i == 0 else "single axis only",
                    "source": p}
    return {"pairs": [], "ratio": [], "samples": None, "endpoint": "", "source": ""}


def draw_paired_effects(data, outdir, name, title):
    if not data["pairs"]:
        print(f"  skipped {name} (no pairs in prepared data)")
        return None
    lab = label_with_n([p.replace("|", "\nvs ") for p in data["pairs"]],
                       data.get("samples"))
    fig, ax = plt.subplots(figsize=(7.6, 0.62 * len(lab) + 2.0))
    y = np.arange(len(lab))
    _draw_strips(ax, data.get("samples"), y, C["physics"], width=0.44)
    ax.scatter(data["ratio"], y, s=58, zorder=5, color=C["physics"],
               edgecolor="white", linewidth=1.0)
    ax.axvline(1.0, color="#444", lw=0.9, ls="--")
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(lab, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("physics-guided advantage  (RMSE distortion / physics distortion, log)\n"
                  ">1 = physics better;  small marks = per-label terms averaging to the point")
    ax.set_title(f"{title}\n{data['endpoint']}", fontsize=11, fontweight="bold")
    style(ax)
    return save(fig, outdir, name)


# ------------------------------------------------------------------ axis decomposition

def _safe_float(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def prepare_axis_decomposition(balance_csv, conservation_csv, block="2",
                               balance_labels_csv=None, conservation_labels_csv=None,
                               balance_labels_family=None, conservation_labels_family=None,
                               labels_lead=None):
    b = {r["pair"]: r for r in _block_rows(read_csv(balance_csv), block)}
    c = {r["pair"]: r for r in _block_rows(read_csv(conservation_csv), block)}
    pairs = [p for p in b if p in c]
    return {"pairs": pairs,
            "balance": [float(b[p]["ratio"]) for p in pairs],
            "conservation": [float(c[p]["ratio"]) for p in pairs],
            "balance_samples": samples_for(
                read_label_samples(balance_labels_csv, "pair", "contribution",
                                   family=balance_labels_family, lead=labels_lead), pairs),
            "conservation_samples": samples_for(
                read_label_samples(conservation_labels_csv, "pair", "contribution",
                                   family=conservation_labels_family, lead=labels_lead),
                pairs)}


def draw_axis_decomposition(data, outdir, name, title):
    if not data["pairs"]:
        print(f"  skipped {name} (no pairs in prepared data)")
        return None
    labels = [q.replace("|", "\nvs ") for q in data["pairs"]]
    return draw_dotplot(
        labels,
        [("balance axis", data["balance"], data.get("balance_samples"), C["physics"]),
         ("conservation axis", data["conservation"], data.get("conservation_samples"),
          C["accent"])],
        outdir, name, title,
        "ratio per axis  (>1 = physics better on that axis, log)",
        logx=True, refline=1.0)


# ------------------------------------------------------------------ shared: dot plots

def draw_dotplot(labels, series, outdir, name, title, xlabel,
                 logx=True, refline=1.0, annotate=None, caption=None):
    annotate = _cap(annotate)          # a sentence on the axes is still a caption
    if not labels or not series:
        print(f"  skipped {name} (no data)")
        return None
    n, k = len(labels), len(series)
    fig, ax = plt.subplots(figsize=(7.8, 0.52 * n + 2.0))
    y = np.arange(n)
    # Offset multiple series so their dots do not overprint at equal values.
    offs = np.linspace(-0.16, 0.16, k) if k > 1 else [0.0]
    rng = np.random.default_rng(0)     # seeded: an unseeded jitter diffs on every rebuild
    for (lab, vals, samples, col), off in zip(series, offs):
        yy = y + off
        _draw_strips(ax, samples, yy, col, width=0.26 / max(k, 1), rng=rng)
        ax.scatter(vals, yy, s=54, color=col, edgecolor="white", linewidth=1.0,
                   zorder=5, label=lab)
    if refline is not None:
        ax.axvline(refline, color="#444", lw=0.9, ls="--", zorder=1)
    if logx:
        ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    _legend_rows = 0 if k <= 1 else (1 if k <= 2 else 2)
    ax.set_title(title, fontsize=11, fontweight="bold",
                 pad=(6.0 + 14.0 * _legend_rows) if (annotate or caption) else 6.0)
    has_labels = k > 1 or bool(series[0][0])
    if has_labels:
        if annotate or caption:
            ax.legend(frameon=False, fontsize=8.5, loc="lower left",
                      bbox_to_anchor=(0.0, 1.005), ncol=min(k, 2), borderaxespad=0.0)
        else:
            ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    if annotate:
        bottom, top = ax.get_ylim()
        band = 1.3
        ax.set_ylim(bottom + band, top)
        import matplotlib.transforms as mtransforms
        trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(0.02, bottom + band * 0.5, annotate, transform=trans, fontsize=8,
                va="center", ha="left",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc"))
    style(ax)
    if caption:
        frac = min(0.34, 0.05 * (1 + len(caption) // 150))
        fig.tight_layout(rect=[0, frac, 1, 1])
        fig.text(0.5, frac * 0.92, caption, ha="center", va="top", fontsize=8.5, wrap=True)
    return save(fig, outdir, name)


# ------------------------------------------------------------------ RMSE compression

def balance_as_percent_of_floor(series, floor_label):
    floor = next((s for s in series if s[0] == floor_label), None)
    if floor is None:
        return series
    denom = floor[2]
    out = []
    for lab, rmse, bal, col, mk in series:
        out.append((lab, rmse,
                    [100.0 * b / d if d else float("nan") for b, d in zip(bal, denom)],
                    col, mk))
    return out


def draw_rmse_compression(leads, series, outdir, name, title, crossing_note=None,
                          balance_percent=False, rmse_bands=None, agreement=None,
                          agreement_of=("", "")):
    if not leads or not series:
        print(f"  skipped {name} (no data)")
        return None
    fig, (axR, axB) = plt.subplots(1, 2, figsize=(12.6, 5.4))
    for lab, rmse, bal, col, mk in series:
        band = (rmse_bands or {}).get(lab)
        if band:
            blo, bhi = [b[0] for b in band], [b[1] for b in band]
            axR.fill_between(leads, blo, bhi, color=col, alpha=0.10, lw=0, zorder=2)
            for edge in (blo, bhi):
                axR.plot(leads, edge, "-", color=col, lw=0.8, alpha=0.55, zorder=3)
        axR.plot(leads, rmse, "-", color=col, marker=mk, ms=7, lw=2,
                 mec="white", mew=1, label=lab, zorder=5)
        axB.plot(leads, bal, "-", color=col, marker=mk, ms=7, lw=2,
                 mec="white", mew=1, label=lab, zorder=5)
    axR.axhline(0, color="#888", lw=1, ls=":")
    axR.set_title("RMSE degradation vs fp32  (headline variables)",
                  fontsize=11.5, fontweight="bold", pad=24 if agreement else None)
    axR.set_ylabel("RMSE degradation (%)")
    if agreement:
        blend = mtransforms.blended_transform_factory(axR.transData, axR.transAxes)
        for L, (k, tot) in zip(leads, agreement):
            axR.text(L, 1.012, f"{k}/{tot}", transform=blend, ha="center", va="bottom",
                     fontsize=8.5, color="#444", fontweight="bold")
    note_lines = []
    if rmse_bands:
        note_lines.append("band = 95% moving-block bootstrap CI on the pooled "
                          "degradation  (2000 resamples, block = 4 inits ~ 1 month)")
        a, b = agreement_of
        if agreement and a and b:
            note_lines.append(f"n/N above = initialisations (of {agreement[0][1]}) where "
                              f"{a} is worse than {b}")
        lo, hi = axR.get_ylim()
        axR.set_ylim(lo, hi + 0.06 * (hi - lo))
    axB.set_yscale("log")
    if balance_percent:
        axB.set_title("balance damage  (% of uniform floor)", fontsize=11.5,
                      fontweight="bold")
        axB.set_ylabel("balance damage, % of uniform floor (log)")
        axB.axhline(100.0, color="#444", lw=0.6, alpha=0.5)
    else:
        axB.set_title("balance distortion  (SVR)", fontsize=11.5, fontweight="bold")
        axB.set_ylabel("balance distortion (SVR, log)")
        axB.axhline(1.0, color="#444", lw=0.6, alpha=0.5)
    for ax in (axR, axB):
        # axR's label may already have been set to the band caveat above; do not clobber it.
        if not ax.get_xlabel():
            ax.set_xlabel("lead time (h)")
        ax.set_xticks(list(leads))
        ax.grid(True, which="both", ls="-", lw=0.4, color="#ececec")
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axR.legend(frameon=False, fontsize=9, loc="upper right")
    if len(series) >= 2 and len(leads) >= 2:
        crossing_note = _cap(crossing_note)
        cross = next((k for k in range(len(leads))
                      if series[0][1][k] < series[1][1][k]), None)
        if CAPTIONS_ON_FIGURE and cross is not None and cross > 0:
            x_last, x_prev = leads[cross], leads[cross - 1]
            y_floor = series[0][1][cross]
            ymin, ymax = axR.get_ylim()
            pad = 0.24 * (ymax - ymin)
            axR.set_ylim(ymin - pad, ymax)
            label = f"from {x_last}h onward: the floor scores better on RMSE"
            if crossing_note:
                label += f"\n{crossing_note}"
            axR.annotate(
                label,
                xy=(x_last, y_floor), xytext=(x_prev + 0.5 * (x_last - x_prev), ymin - 0.55 * pad),
                ha="center", va="center", fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color="#777777", lw=1.0,
                                connectionstyle="arc3,rad=0.15"),
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc"), zorder=6)
        elif crossing_note:
            ymin, ymax = axR.get_ylim()
            pad = 0.24 * (ymax - ymin)
            axR.set_ylim(ymin - pad, ymax)
            axR.annotate(
                crossing_note,
                xy=(0.5, 0.0), xycoords="axes fraction",
                xytext=(0.5, 0.055), textcoords="axes fraction",
                ha="center", va="bottom", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc"), zorder=6)
    fig.suptitle(title, fontsize=11.5, fontweight="bold", y=1.02)
    if note_lines:
        fig.tight_layout(rect=(0, 0.045 * len(note_lines), 1, 1))
        fig.text(0.008, 0.006, "\n".join(note_lines), ha="left", va="bottom",
                 fontsize=8.5, color="#444")
    else:
        fig.tight_layout()
    return save(fig, outdir, name)


# ------------------------------------------------------------------ lead divergence

def prepare_lead_divergence(csv_by_model, block="2"):
    models, leads, ratio, slope = [], {}, {}, {}
    for m, path in csv_by_model.items():
        rows = _block_rows(read_csv(path), block)
        if not rows:
            continue
        models.append(m)
        leads[m] = [int(r["lead"]) for r in rows]
        ratio[m] = [float(r["ratio"]) for r in rows]
        slope[m] = ""
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    if not line.startswith("#"):
                        break
                    if "POST-HOC" in line:
                        slope[m] = line.lstrip("# ").strip()
    return {"models": models, "leads": leads, "ratio": ratio, "slope_text": slope}


def parse_slope(provenance_line):
    if not provenance_line:
        return None
    m = re.search(r"log-log slope\s*([+-]?[\d.]+)"
                  r"(?:\s*\[\s*([+-]?[\d.]+)\s*,\s*([+-]?[\d.]+)\s*\])?",
                  provenance_line)
    return (m.group(1), m.group(2), m.group(3)) if m else None


def n_inits_of(results_csv):
    """Largest n_inits in a harness_results.csv, or None. The figure caption states the
    sample each arm rests on, and that number moves when a run is extended."""
    vals = []
    for r in read_csv(results_csv):
        try:
            vals.append(int(float(r["n_inits"])))
        except (KeyError, TypeError, ValueError):
            continue
    return max(vals) if vals else None


def draw_lead_divergence(data, outdir, name, title, caption=""):
    if not data["models"]:
        print(f"  skipped {name} (no model series)")
        return None
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for m in data["models"]:
        col = C.get(m.lower(), C["neutral"])
        x, y = data["leads"][m], data["ratio"][m]
        ax.plot(x, y, "-", color=col, marker="o", ms=7, lw=2.0, mec="white", mew=1,
                label=m, zorder=4)
    ax.axhline(1.0, color="#444", lw=0.9, ls="--", zorder=1)
    ax.set_yscale("log")
    ax.set_xticks(sorted({v for m in data["models"] for v in data["leads"][m]}))
    ax.set_xlabel("forecast lead time (h)")
    ax.set_ylabel("physics advantage  (RMSE / physics distortion, log)")
    ax.set_title(title, fontsize=11.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    style(ax)
    fig.tight_layout(rect=(0.0, 0.16 if caption else 0.0, 1.0, 1.0))
    if caption:
        fig.text(0.5, 0.05, caption, ha="center", va="center", fontsize=8.5, wrap=True)
    return save(fig, outdir, name)


# ------------------------------------------------------------------ cross-model

_XMODELS = ("Aurora", "Stormer")


def _x_rows(path, block="2"):
    """Thin wrapper: read + apply the legacy block filter, delegating the filter itself to
    _block_rows so the two projects cannot drift on what it means. See _block_rows: it is a
    no-op on any CSV written since the bootstrap was removed."""
    return _block_rows(read_csv(path), block)


def prepare_endpoint_forest(csv_by_model_endpoint, block="2", labels_csv_by_model=None,
                            labels_family_by_endpoint=None, labels_lead=None):
    out = {"labels": [], "model": [], "endpoints": [], "by_endpoint": {}}
    endpoints = []
    for spec in csv_by_model_endpoint.values():
        for e in spec:
            if e not in endpoints:
                endpoints.append(e)
    out["endpoints"] = endpoints
    for e in endpoints:
        out["by_endpoint"][e] = {"ratio": [], "samples": []}
    any_sample = False
    for model in _XMODELS:
        spec = csv_by_model_endpoint.get(model)
        if not spec:
            continue
        tables = {e: {r["pair"]: r for r in _x_rows(p, block)} for e, p in spec.items()}
        label_spec = (labels_csv_by_model or {}).get(model) or {}
        fam_of = labels_family_by_endpoint or {}
        samples = {e: read_label_samples(label_spec.get(e), "pair", "contribution",
                                         family=fam_of.get(e), lead=labels_lead)
                   for e in endpoints}
        any_sample = any_sample or any(samples.values())
        first = tables[endpoints[0]]
        for pair in first:
            if not all(pair in tables[e] for e in endpoints):
                continue
            out["labels"].append(f"{model}  {pair.split('|')[0]}")
            out["model"].append(model)
            for e in endpoints:
                d = out["by_endpoint"][e]
                d["ratio"].append(float(tables[e][pair]["ratio"]))
                d["samples"].append(samples[e].get(pair, []))
    if not any_sample:
        for e in endpoints:
            out["by_endpoint"][e]["samples"] = None
    return out


def endpoint_reversals(data, endpoint_a, endpoint_b, ref=1.0):
    a, b = data["by_endpoint"][endpoint_a], data["by_endpoint"][endpoint_b]
    out = []
    for i, (x, y) in enumerate(zip(a["ratio"], b["ratio"])):
        if (x < ref) == (y < ref):
            continue
        out.append(i)
    return out


def draw_endpoint_forest(data, outdir, name, title, endpoint_a, endpoint_b, caption=""):
    if not data["labels"]:
        print(f"  skipped {name} (no pairs common to both endpoints)")
        return None
    n = len(data["labels"])
    fig, ax = plt.subplots(figsize=(8.6, 0.50 * n + 3.0))
    y = np.arange(n)
    a, b = data["by_endpoint"][endpoint_a], data["by_endpoint"][endpoint_b]
    flips = set(endpoint_reversals(data, endpoint_a, endpoint_b))

    for i in y:
        ax.plot([a["ratio"][i], b["ratio"][i]], [i, i],
                color=C["warn"] if i in flips else "#c9c9c9",
                lw=2.4 if i in flips else 1.4, zorder=2,
                solid_capstyle="round")
    srng = np.random.default_rng(0)
    for (d, e, col, mk) in ((a, endpoint_a, C["neutral"], "s"),
                            (b, endpoint_b, C["physics"], "o")):
        _draw_strips(ax, d.get("samples"), y, col, width=0.34, marker=mk, rng=srng)
        ax.scatter(d["ratio"], y, s=62, marker=mk, zorder=5,
                   color=col, edgecolor=col, linewidth=1.4, label=e)
    for i in flips:
        ax.annotate("reversal", (max(a["ratio"][i], b["ratio"][i]), i),
                    textcoords="offset points", xytext=(9, -3),
                    fontsize=7.5, color=C["warn"], fontweight="bold")
    ax.axvline(1.0, color="#444", lw=0.9, ls="--", zorder=1)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(data["labels"], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("physics-guided advantage  (log)\n>1 = physics better")
    ax.set_title(title, fontsize=11.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", title="endpoint",
              title_fontsize=8.5)
    style(ax)
    if caption:
        fig.tight_layout(rect=(0.0, 0.15, 1.0, 1.0))
        fig.text(0.5, 0.055, caption, ha="center", va="center", fontsize=8.2, wrap=True)
    else:
        fig.tight_layout()
    return save(fig, outdir, name)


# ------------------------------------------------ additive surrogate validity

import glob

SCHEME_ORDER = ("W8", "W8A8", "W8A8_sq", "W4")


def _scheme_of(tag):
    for scheme in ("W8A8_sq", "W8A8", "W4W8", "W4", "W8"):
        if tag.startswith(scheme):
            return scheme
    return tag.split("_")[0]


def _scheme_of_dir(path):
    base = os.path.basename(path.rstrip(os.sep))
    base = base.replace("ablation_analysis_", "", 1)
    for pre in ("ablations_", "ablation_"):
        if base.startswith(pre):
            return base[len(pre):]
    return base


DEFAULT_BUCKETS = {"balance": "physics", "conservation": "physics", "standard": "RMSE"}
AXIS_BUCKETS = {"balance": "balance", "conservation": "conservation", "standard": "RMSE"}

POOLED_SCHEME = "all schemes pooled"


def _median(values):
    v = sorted(values)
    return v[len(v) // 2] if v else float("nan")


def prepare_additivity(roots_by_model, lead=120, tol_col="additive_ok", buckets=None,
                       pooled=False, family_first=False):
    from allocator import _family_of_metric
    from plot_common import AGG_CLASS
    buckets = buckets or DEFAULT_BUCKETS
    bucket_order = list(dict.fromkeys(buckets.values()))
    out = []
    for model, root in roots_by_model.items():
        cells = {}
        for d in sorted(glob.glob(root.path("ablation_analysis_*"))):
            real_scheme = _scheme_of_dir(d)
            scheme = POOLED_SCHEME if pooled else real_scheme
            rows = read_csv(os.path.join(d, f"additivity_{lead}h.csv"))
            if not rows:
                continue
            for r in rows:
                bucket = buckets.get(AGG_CLASS.get(_family_of_metric(r.get("metric", ""))))
                if bucket is None:
                    continue
                key = (scheme, bucket)
                c = cells.setdefault(key, {"vals": [], "ok": 0, "dropped": 0,
                                           "by_family": {}})
                try:
                    v = float(r["ratio"])
                except (KeyError, ValueError, TypeError):
                    continue
                if not math.isfinite(v) or v <= 0:
                    c["dropped"] += 1
                    continue
                c["vals"].append(v)
                c["ok"] += _truthy(r.get(tol_col, ""))
                c["by_family"].setdefault(
                    (real_scheme, _family_of_metric(r.get("metric", ""))), []).append(v)
        order = [s for s in SCHEME_ORDER if any(k[0] == s for k in cells)]
        order += sorted({k[0] for k in cells} - set(order))
        for scheme in order:
            for bucket in bucket_order:
                c = cells.get((scheme, bucket))
                if not c or not c["vals"]:
                    continue
                if family_first:
                    c = dict(c, vals=[_median(v) for v in c["by_family"].values()])
                    c["ok"] = sum(1 for v in c["vals"] if abs(v - 1.0) <= 0.2)
                v = sorted(c["vals"])
                out.append({"model": model, "scheme": scheme, "bucket": bucket,
                            "median": v[len(v) // 2],
                            "q1": v[len(v) // 4], "q3": v[(3 * len(v)) // 4],
                            "n": len(v), "n_ok": c["ok"], "n_dropped": c["dropped"],
                            "vals": v})
    return out


# ------------------------------------------------ comparing additivity ACROSS axes

def _median_abs_log10(ratios):
    v = sorted(abs(math.log10(r)) for r in ratios if r > 0 and math.isfinite(r))
    return v[len(v) // 2] if v else float("nan")


def additivity_calibration(rows):
    grouped = {}
    for r in rows:
        grouped.setdefault((r["model"], r["bucket"]), []).extend(r.get("vals", []))
    return [{"model": model, "bucket": bucket, "n": len(vals),
             "median_abs_log10": _median_abs_log10(vals)}
            for (model, bucket), vals in grouped.items()]


def additivity_bucket_contrast(rows):
    per_scheme = {}
    for r in rows:
        per_scheme.setdefault(r["model"], {}).setdefault(r["scheme"], {})[r["bucket"]] = \
            _median_abs_log10(r.get("vals", []))
    out = []
    for model, schemes in per_scheme.items():
        names = list(dict.fromkeys(b for s in schemes.values() for b in s))
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                diffs = [s[a] - s[b] for s in schemes.values()
                         if a in s and b in s
                         and math.isfinite(s[a]) and math.isfinite(s[b])]
                if not diffs:
                    continue
                out.append({"model": model, "bucket_a": a, "bucket_b": b,
                            "n_schemes": len(diffs),
                            "median_diff": float(np.median(diffs))})
    return out


_BUCKET_STYLE = {
    "physics": (C["physics"], "o"),
    "balance": (C["physics"], "o"),
    "conservation": (C["accent"], "D"),
    "RMSE": (C["rmse"], "s"),
}
_BUCKET_FALLBACK = (C["neutral"], "^")
_BUCKET_SPREAD = 0.30      # total vertical span of one row's buckets, in y-tick units


def _bucket_layout(bucket_names):
    names = list(dict.fromkeys(bucket_names))
    n = len(names)
    out = {}
    for i, b in enumerate(names):
        col, mk = _BUCKET_STYLE.get(b, _BUCKET_FALLBACK)
        off = 0.0 if n == 1 else _BUCKET_SPREAD * (i / (n - 1) - 0.5)
        out[b] = {"color": col, "marker": mk, "offset": off}
    return out


SOLID_ALPHA = 0.75      # rows carrying the estimate being claimed
FAINT_ALPHA = 0.28      # rows that are context only (figA31's per-scheme rows)

BOX_MIN_N = 10


def _box_stats(values, whis=1.5):
    v = sorted(float(x) for x in values)
    q1, med, q3 = (float(np.percentile(v, p)) for p in (25, 50, 75))
    iqr = q3 - q1
    lo_fence, hi_fence = q1 - whis * iqr, q3 + whis * iqr
    inside = [x for x in v if lo_fence <= x <= hi_fence] or v
    return {"med": med, "q1": q1, "q3": q3,
            "whislo": min(inside), "whishi": max(inside),
            "fliers": [x for x in v if x < lo_fence or x > hi_fence],
            "label": ""}


def _draw_row_distribution(ax, vals, y, colour, marker, alpha, width, rng, dim=False):
    vals = [v for v in (_safe_float(x) for x in vals) if v is not None]
    if not vals:
        return
    if len(vals) >= BOX_MIN_N:
        bp = ax.bxp([_box_stats(vals)], positions=[y], widths=width,
                    orientation="horizontal", patch_artist=True, showfliers=True,
                    manage_ticks=False, zorder=4)
        for patch in bp["boxes"]:
            patch.set(facecolor=colour, edgecolor=colour, alpha=alpha * 0.45, linewidth=1.0)
        for key in ("whiskers", "caps"):
            for a in bp[key]:
                a.set(color=colour, linewidth=1.1, alpha=alpha)
        for a in bp["medians"]:
            a.set(color=colour, linewidth=2.0, alpha=1.0)
        for a in bp["fliers"]:
            a.set(marker=marker, markersize=3.4, markerfacecolor="none",
                  markeredgecolor=colour, alpha=alpha)
        return
    jit = (rng.random(len(vals)) - 0.5) * width * 0.55
    ax.scatter(list(vals), [y + j for j in jit], s=20 if dim else 34, marker=marker,
               zorder=4, facecolor="none" if dim else colour, edgecolor=colour,
               linewidth=0.9, alpha=alpha if dim else 0.85)
    if len(vals) < 2:
        return
    med = float(np.median(list(vals)))
    ax.plot([med, med], [y - width / 2, y + width / 2], color=colour,
            lw=1.2 if dim else 2.4, zorder=5, alpha=alpha if dim else 1.0,
            solid_capstyle="butt")


def _draw_strips(ax, samples, ypos, colour, width, rng=None, marker="o", alpha=0.85):
    if not samples:
        return
    rng = rng if rng is not None else np.random.default_rng(0)
    for yi, s in zip(ypos, samples):
        if s is not None and len(s):
            _draw_row_distribution(ax, s, yi, colour, marker, alpha, width=width,
                                   rng=rng, dim=True)


def draw_additivity(rows, outdir, name, title, caption="", broken=(), tol=0.2, faint=(),
                    wide=False, boxes=False):
    if not rows:
        print(f"  skipped {name} (no additivity CSVs found)")
        return None
    cells = []
    for r in rows:
        key = (r["model"], r["scheme"])
        if key not in cells:
            cells.append(key)
    fig, ax = plt.subplots(figsize=(11.6 if wide else 8.4, 0.52 * len(cells) + 3.0))
    y = {k: i for i, k in enumerate(cells)}

    ax.axvspan(1.0 - tol, 1.0 + tol, color=C["accent"], alpha=0.10, zorder=0)
    ax.axvline(1.0, color="#444", lw=1.0, ls="--", zorder=1)
    layout = _bucket_layout([r["bucket"] for r in rows])
    faint = set(faint)
    rng = np.random.default_rng(0)      # seeded: an unseeded jitter diffs on every rebuild
    for bucket, s in layout.items():
        col, mk, off = s["color"], s["marker"], s["offset"]
        first = True
        for r in rows:
            if r["bucket"] != bucket:
                continue
            key = (r["model"], r["scheme"])
            dim = key in faint
            alpha = FAINT_ALPHA if dim else SOLID_ALPHA
            i = y[key] + off
            at = r["q3"]
            if boxes and r.get("vals"):
                at = max(float(v) for v in r["vals"])
            if boxes and r.get("vals"):
                _draw_row_distribution(ax, r["vals"], i, col, mk, alpha,
                                       _BUCKET_SPREAD * 0.62, rng, dim=dim)
                if first and not dim:
                    ax.scatter([], [], s=62, color=col, marker=mk, label=bucket)
            else:
                ax.plot([r["q1"], r["q3"]], [i, i], color=col, lw=1.0 if dim else 1.6,
                        zorder=3, solid_capstyle="round", alpha=alpha)
                ax.scatter([r["median"]], [i], s=30 if dim else 62, color=col,
                           edgecolor="white", linewidth=0.7 if dim else 1.1, marker=mk,
                           zorder=4, alpha=alpha,
                           label=bucket if (first and not dim) else None)
            first = first and dim
            if not dim:
                ax.annotate(f"{r['n_ok']}/{r['n']}", (at, i), fontsize=6.8, color=col,
                            textcoords="offset points", xytext=(6, -2.5), va="center")

    labels = []
    for m, s in cells:
        mark = "   << allocation reverses" if (m, s) in set(broken) else ""
        labels.append(f"{m}  {s}{mark}")
    ax.set_xscale("log")
    ax.set_yticks(range(len(cells)))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("additivity ratio   sum of one-at-a-time deltas / full-quantisation delta  (log)\n"
                  f"1.0 = perfectly additive;  >1 = surrogate OVER-predicts damage;  "
                  f"shaded = +/-{tol:.0%};  n/N = metrics within tolerance")
    ax.set_title(title, fontsize=11.5, fontweight="bold", pad=22 if wide else None)
    if wide:
        ax.legend(frameon=False, fontsize=8.5, ncol=len(layout), loc="lower left",
                  bbox_to_anchor=(0.0, 1.005), title="metric class", title_fontsize=8.5)
    else:
        ax.legend(frameon=False, fontsize=8.5, loc="lower right", title="metric class",
                  title_fontsize=8.5)
    style(ax)
    if caption:
        fig.tight_layout(rect=(0.0, 0.17, 1.0, 1.0))
        fig.text(0.5, 0.06, caption, ha="center", va="center", fontsize=8.2, wrap=True)
    else:
        fig.tight_layout()
    return save(fig, outdir, name)


def write_additivity_csv(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "scheme", "class", "n_metrics", "n_within_tol",
                    "median_ratio", "q1", "q3", "n_dropped_nonpositive"])
        for r in rows:
            w.writerow([r["model"], r["scheme"], r["bucket"], r["n"], r["n_ok"],
                        f"{r['median']:.4f}", f"{r['q1']:.4f}", f"{r['q3']:.4f}",
                        r["n_dropped"]])
    print(f"  wrote {path}")
    return path


# ------------------------------------------------ lead trend across all pairs

SUPERSEDED_SUFFIX = "__superseded"


def prepare_lead_slopes(by_lead_csv_by_model, axis="balance", leads=(24, 72, 120, 168),
                        declared=("W8A8_span1", "W8A8_rmse_span1")):
    out = {}
    for model, paths in by_lead_csv_by_model.items():
        if isinstance(paths, str):
            paths = [paths]
        by_tag = {}
        for path in paths:
            for r in read_csv(path):
                tag = r.get("tag", "")
                if not tag or tag.endswith(SUPERSEDED_SUFFIX):
                    continue
                try:
                    by_tag.setdefault(tag, {})[int(float(r["lead"]))] = float(r[axis])
                except (KeyError, ValueError, TypeError):
                    continue
        rows = []
        for tag in sorted(by_tag):
            twin = _rmse_twin(tag)
            if twin is None or twin not in by_tag:
                continue
            xs, ys, series = [], [], []
            for L in leads:
                p, q = by_tag[tag].get(L), by_tag[twin].get(L)
                if p is None or q is None or p <= 0 or q <= 0:
                    series.append(None)
                    continue
                ratio = q / p
                series.append(ratio)
                xs.append(math.log(L))
                ys.append(math.log(ratio))
            if len(xs) < 3:                 # a slope through two points is not a trend
                continue
            rows.append({"pair": tag, "ratios": series, "leads": list(leads),
                         "slope": _ols_slope(xs, ys),
                         "scheme": _scheme_of(tag),
                         "declared": tag == declared[0]})
        out[model] = rows
    return out


def _rmse_twin(tag):
    """The RMSE-guided counterpart of a physics-guided tag, or None if `tag` is not a
    physics-guided span/knee. Anticontrols (`rmseup`), uniform endpoints, random and probe
    controls have no matched twin and are excluded here rather than silently paired."""
    if any(k in tag for k in ("rmse", "floor", "ceiling", "rand", "probe")):
        return None
    for scheme in ("W8A8_sq_", "W8A8_", "W4W8_"):
        if tag.startswith(scheme):
            return scheme + "rmse_" + tag[len(scheme):]
    return None


def _ols_slope(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx


def draw_lead_slopes(data, outdir, name, title, caption=""):
    rows = [(m, r) for m in _XMODELS for r in data.get(m, [])]
    if not rows:
        print(f"  skipped {name} (no pairs with an RMSE twin)")
        return None
    fig, ax = plt.subplots(figsize=(7.8, 0.46 * len(rows) + 2.6))
    y = np.arange(len(rows))
    for i, (m, r) in enumerate(rows):
        col = C["aurora"] if m == "Aurora" else C["stormer"]
        ax.scatter(r["slope"], i, s=88 if r["declared"] else 58,
                   color=col, edgecolor="black" if r["declared"] else "white",
                   linewidth=1.6 if r["declared"] else 1.0, zorder=4)
    ax.axvline(0.0, color="#444", lw=0.9, ls="--", zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{m}  {r['pair']}" + ("   [declared]" if r["declared"] else "")
                        for m, r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("log-log slope of physics advantage vs lead, BALANCE axis\n"
                  "<0 = advantage shrinks with lead;  >0 = it grows;  "
                  "ringed = the same pair fig06 reports (fig06 scores the composite "
                  "endpoint, so its slope differs)")
    ax.set_title(title, fontsize=11.5, fontweight="bold")
    style(ax)
    if caption:
        fig.tight_layout(rect=(0.0, 0.17, 1.0, 1.0))
        fig.text(0.5, 0.06, caption, ha="center", va="center", fontsize=8.2, wrap=True)
    else:
        fig.tight_layout()
    return save(fig, outdir, name)


def write_lead_slopes_csv(data, path):
    """The numbers behind draw_lead_slopes, so the table can be cited in the text."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "pair", "scheme", "declared",
                    *[f"ratio_{L}h" for L in (24, 72, 120, 168)], "loglog_slope"])
        for model in _XMODELS:
            for r in data.get(model, []):
                w.writerow([model, r["pair"], r["scheme"], r["declared"],
                            *["" if v is None else f"{v:.6g}" for v in r["ratios"]],
                            f"{r['slope']:.6g}"])
    print(f"  wrote {path}")
    return path


def prepare_replication_forest(aurora_csv, stormer_csv, labels_csv_by_model=None,
                               labels_family=None, labels_lead=None,
                               labels_value_field="contribution"):
    labels = labels_csv_by_model or {}
    out = {"labels": [], "ratio": [], "model": [], "samples": []}
    any_sample = False
    for model, path in zip(_XMODELS, (aurora_csv, stormer_csv)):
        by_pair = read_label_samples(labels.get(model), "pair", labels_value_field,
                                     family=labels_family, lead=labels_lead)
        any_sample = any_sample or bool(by_pair)
        for r in _x_rows(path):
            out["labels"].append(f"{model}  {r['pair'].split('|')[0]}")
            out["ratio"].append(float(r["ratio"]))
            out["model"].append(model)
            out["samples"].append(by_pair.get(r["pair"], []))
    if not any_sample:
        out["samples"] = None
    return out


def _cumulative(values):
    v = np.sort(np.asarray([max(float(x), 0.0) for x in values], dtype=np.float64))[::-1]
    total = float(v.sum())
    if total <= 0:
        return {"x": [], "y": [], "n_groups": int(v.size), "top1": float("nan")}
    cum = np.cumsum(v) / total
    xs = (np.arange(1, v.size + 1) / v.size)
    return {"x": list(xs), "y": list(cum), "n_groups": int(v.size), "top1": float(cum[0])}


def _distortion_by_group(path, lead, scheme, axis):
    from allocator import load_distortion_table
    d = load_distortion_table({scheme: path}, lead=lead, axes=(axis,), min_effect_frac=0.0)
    return {g: d[g][scheme][axis] for g in d}


def prepare_concentration(sensitivity_by_model, lead=120, scheme="W8A8", axis="balance"):
    out = {}
    for model, path in sensitivity_by_model.items():
        if not os.path.exists(path):
            print(f"  skip concentration for {model} (no {path})")
            continue
        out[model] = _cumulative(list(_distortion_by_group(path, lead, scheme, axis).values()))
    return out


def concentration_top_group(path, lead=120, scheme="W8A8", axis="balance"):
    if not os.path.exists(path):
        return ""
    by_group = _distortion_by_group(path, lead, scheme, axis)
    return max(by_group, key=by_group.get) if by_group else ""


def _spearman(a, b):
    def rank(x):
        o = np.argsort(np.asarray(x), kind="mergesort")
        r = np.empty(len(x), dtype=float)
        r[o] = np.arange(1, len(x) + 1)
        return r
    ra, rb = rank(a) - np.mean(rank(a)), rank(b) - np.mean(rank(b))
    d = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def _loglog(predicted, measured):
    p = np.asarray(predicted, dtype=np.float64)
    m = np.asarray(measured, dtype=np.float64)
    ok = (p > 0) & (m > 0) & np.isfinite(p) & np.isfinite(m)
    p, m = p[ok], m[ok]
    if p.size < 2:
        return {"rho": float("nan"), "slope": float("nan"), "n": int(p.size), "ok": ok}
    lx, ly = np.log(p), np.log(m)
    lx0 = lx - lx.mean()
    slope = float((lx0 * (ly - ly.mean())).sum() / (lx0 ** 2).sum())
    return {"rho": _spearman(p, m), "slope": slope, "n": int(p.size), "ok": ok}


def _merge_by_tag(paths):
    if isinstance(paths, (str, bytes, os.PathLike)):
        paths = [paths]
    out = {}
    for p in paths:
        for r in read_csv(p):
            tag = r.get("tag")
            if tag is None:
                continue
            merged = dict(r)
            merged.update(out.get(tag, {}))     # earlier file wins on shared keys
            out[tag] = merged
    return out


UNQUANTISED_FLOOR = "bf16"


def _floor_reference_tags(pred):
    out = {}
    for tag, row in pred.items():
        floor = row.get("floor", "")
        if floor and floor != UNQUANTISED_FLOOR and not (row.get("config") or "").strip():
            out.setdefault(floor, tag)
    return out


def _transfer_residual(xs, ys, ok):
    """log10(measured/predicted) for the points `_loglog` kept. [] if none."""
    if not len(ok):
        return []
    p = np.asarray(xs, dtype=np.float64)[ok]
    m = np.asarray(ys, dtype=np.float64)[ok]
    return [float(v) for v in np.log10(m / p)]


def draw_transfer_residual_inset(ax, series, loc=(0.55, 0.115, 0.42, 0.27)):
    if not series:
        return None
    ins = ax.inset_axes(loc)
    rng = np.random.default_rng(0)
    for i, (label, vals, colour) in enumerate(series):
        if not len(vals):
            continue
        _draw_row_distribution(ins, list(vals), i, colour, "o", SOLID_ALPHA, 0.52, rng)
    ins.axvline(0.0, color="#555", lw=0.9, ls="--", zorder=2)
    for i, (label, vals, colour) in enumerate(series):
        if not len(vals):
            continue
        ins.text(0.015, i + 0.30, label, transform=ins.get_yaxis_transform(),
                 fontsize=6.5, color=colour, ha="left", va="bottom", zorder=6)
    ins.set_yticks([])
    ins.set_ylim(-0.5, len(series) - 0.05)
    ins.tick_params(axis="x", labelsize=6.5, length=2)
    ins.set_xlabel("log$_{10}$(measured / predicted)", fontsize=6.5, labelpad=1)
    ins.patch.set_alpha(0.92)
    for side in ("top", "right"):
        ins.spines[side].set_visible(False)
    return ins


def prepare_transfer(configs_csv, results_csv, axis, normalise=False):
    pred = _merge_by_tag(configs_csv)
    meas = _merge_by_tag(results_csv)
    refs = _floor_reference_tags(pred) if normalise else {}
    tags, xs, ys = [], [], []
    for t in sorted(set(pred) & set(meas)):
        try:
            x, y = float(pred[t][axis]), float(meas[t][axis])
        except (KeyError, ValueError):
            continue
        if normalise:
            ref = refs.get(pred[t].get("floor", ""))
            if ref is None or ref not in meas:
                continue          # no reference: percent undefined
            try:
                fx, fy = float(pred[ref][axis]), float(meas[ref][axis])
            except (KeyError, ValueError):
                continue
            if not (fx > 0 and fy > 0):
                continue
            x, y = 100.0 * x / fx, 100.0 * y / fy
        tags.append(t)
        xs.append(x)
        ys.append(y)
    fit = _loglog(xs, ys)
    ok = np.asarray(fit["ok"], dtype=bool)
    out = {"tags": tags, "predicted": xs, "measured": ys,
           "rho": fit["rho"], "slope": fit["slope"], "n": fit["n"],
           "normalised": bool(normalise),
           "residual": _transfer_residual(xs, ys, ok),
           "fit_tags": [t for t, keep in zip(tags, ok) if keep]}
    if normalise:
        raw = prepare_transfer(configs_csv, results_csv, axis)
        out["rho_raw"], out["slope_raw"] = raw["rho"], raw["slope"]
    return out


# ------------------------------------------------------------------ scope: measured cost

def prepare_cost_scope(results_csv, tags=None):
    rows = {r["tag"]: r for r in read_csv(results_csv)}
    out = []
    for t in (tags or list(rows)):
        r = rows.get(t)
        if not r:
            continue
        try:
            out.append({"tag": t,
                        "balance": float(r["balance"]),
                        "size_gb": float(r["model_size_gb"]),
                        "latency": float(r["latency_s_per_step"])})
        except (KeyError, TypeError, ValueError):
            continue
    return out


def draw_cost_scope(points, outdir, name, title, caption="", ceiling_tag="ceiling"):
    if not points:
        print(f"  skipped {name} (no cost points)")
        return None
    ceil = next((p for p in points if p["tag"] == ceiling_tag), None)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2))
    for ax, key, xlabel in ((axes[0], "size_gb", "model size (GB, lower = better)"),
                            (axes[1], "latency", "latency (s/step, lower = better)")):
        plotted = [p for p in points if p["tag"] != ceiling_tag]
        xs = [p[key] for p in plotted]
        tol = 0.02 * (max(xs) - min(xs)) if len(xs) > 1 else 0.0
        clusters = {}
        for p in sorted(plotted, key=lambda q: (q[key], -q["balance"])):
            hit = next((c for c in clusters if abs(c - p[key]) <= tol), None)
            clusters.setdefault(p[key] if hit is None else hit, []).append(p)
        for members in clusters.values():
            for rank, p in enumerate(members):
                phys = ("rmse" not in p["tag"] and "rand" not in p["tag"]
                        and "probe" not in p["tag"])
                ax.scatter(p[key], max(p["balance"], 1e-3), s=64, zorder=4,
                           edgecolor="white", linewidth=1.0,
                           color=C["physics"] if phys else C["rmse"])
                ax.annotate(p["tag"].replace("W8A8_", "").replace("W4W8_", "W4:"),
                            (p[key], max(p["balance"], 1e-3)), textcoords="offset points",
                            xytext=(7, 4 - 13 * rank), fontsize=7)
        if ceil:
            ax.axvline(ceil[key], color="#444", ls=":", lw=1.3, zorder=1)
            ax.axhline(max(ceil["balance"], 1e-3), color="#444", ls=":", lw=1.3, zorder=1)
            ax.text(ceil[key], 0.55, " bf16 ceiling", rotation=90, fontsize=8, color="#444",
                    va="center", ha="left",
                    transform=ax.get_xaxis_transform())
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("measured balance distortion @120h  (SVR, log)")
        ax.grid(True, which="both", ls="-", lw=0.4, color="#ececec")
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle(title, fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0.16 if caption else 0.0, 1, 0.99])
    if caption:
        fig.text(0.5, 0.055, caption, ha="center", va="top", fontsize=8.5, wrap=True)
    return save(fig, outdir, name)


# --------------------------------------------- RMSE vs lead across the config families

def _harness_cases(pt, tag):
    from rmse_compression import tag_data
    return tag_data(pt, tag)


def rc_default_agg():
    """rmse_compression.DEFAULT_AGG, imported lazily so figure_core keeps no import cycle."""
    from rmse_compression import DEFAULT_AGG
    return DEFAULT_AGG


def _pooled_fracs_for(pt, tag, lead, variables, ref):
    from rmse_compression import pooled_fracs
    return pooled_fracs(pt, tag, lead, [f"w_rmse_{v}_{lead}" for v in variables], ref=ref)


def read_per_init_var(path):
    """{(tag, lead): (inits, vars, R_tag[D,V], R_ref[D,V])} from a *_per_init_var.csv."""
    acc = {}
    for r in read_csv(path):
        try:
            key = (r["tag"], int(r["lead"]))
            acc.setdefault(key, {}).setdefault(r["init"], {})[r["var"]] = (
                float(r["r_cfg"]), float(r["r_ref"]))
        except (KeyError, TypeError, ValueError):
            continue
    out = {}
    for key, by_init in acc.items():
        inits = sorted(by_init)
        varnames = sorted({v for d in by_init.values() for v in d})
        try:
            q = np.array([[by_init[i][v][0] for v in varnames] for i in inits])
            f = np.array([[by_init[i][v][1] for v in varnames] for i in inits])
        except KeyError:
            continue                      # ragged (tag, lead): drop rather than mis-pair
        out[key] = (inits, varnames, q, f)
    return out


def pooled_boot_ci(per_init_var, tag, leads, n_boot=2000, ci=95, rng_seed=0):
    """[(lo, hi) per lead]: moving-block bootstrap CI on the POOLED degradation."""
    out = []
    for L in leads:
        got = per_init_var.get((tag, L))
        if got is None:
            return None
        _, _, q, f = got
        D = q.shape[0]
        if D < 4:
            return None
        block = max(1, min(round(D ** (1.0 / 3.0)), D))
        rng = np.random.default_rng(rng_seed)
        n_blocks = int(np.ceil(D / block))
        starts = rng.integers(0, D - block + 1, size=(n_boot, n_blocks))
        idx = (starts[..., None] + np.arange(block)).reshape(n_boot, -1)[:, :D]
        qs = np.sqrt((q[idx] ** 2).mean(axis=1))          # [n_boot, V]
        fs = np.sqrt((f[idx] ** 2).mean(axis=1))
        vals = 100.0 * (qs / fs - 1.0).mean(axis=1)       # [n_boot]
        alpha = (100 - ci) / 2
        lo, hi = np.percentile(vals, [alpha, 100 - alpha])
        out.append((float(lo), float(hi)))
    return out


def prepare_rmse_lead_families(pt, tags, leads, variables, ref="FP32", per_init=False,
                               agg=None):
    ref_cases = sorted(_harness_cases(pt, ref).keys()) if ref in pt else []
    if not ref_cases:
        return {}
    out = {}
    for tag in tags:
        if tag not in pt:
            continue
        cases = _harness_cases(pt, tag)
        if not set(ref_cases) <= set(cases.keys()):
            continue
        series = []
        for lead in leads:
            per_case = []
            for c in ref_cases:
                fracs = []
                for v in variables:
                    key = f"w_rmse_{v}_{lead}"
                    f = float(_harness_cases(pt, ref)[c][lead]["RMSE"][key])
                    q = float(cases[c][lead]["RMSE"][key])
                    fracs.append((q - f) / f)
                per_case.append(float(np.mean(fracs)))
            pct = [100.0 * v for v in per_case]
            if per_init:
                series.append(pct)
                continue
            if (agg or rc_default_agg()) == "pooled":
                series.append(100.0 * float(np.mean(
                    _pooled_fracs_for(pt, tag, lead, variables, ref))))
            else:
                series.append(float(np.mean(pct)))
        out[tag] = series
    return out


def draw_rmse_lead_grid(cells, leads, outdir, name, title, caption=""):
    if not cells:
        print(f"  skipped {name} (no cells)")
        return None
    nrow, ncol = len(cells), max(len(cols) for _, cols in cells)
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.5 * nrow), squeeze=False)
    seen = {}
    for r, (row_label, cols) in enumerate(cells):
        for c in range(ncol):
            ax = axs[r][c]
            col_label, series = cols[c] if c < len(cols) else ("", None)
            if r == 0 and col_label:
                ax.set_title(col_label, fontsize=11, fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"{row_label}\nRMSE degradation (%)", fontsize=9.5)
            if not series:
                ax.text(0.5, 0.5, "not run\nin this sweep", ha="center", va="center",
                        fontsize=9.5, color="#999999", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
                continue
            for entry in series:
                label, values, colour, marker = entry[:4]
                ls = entry[4] if len(entry) > 4 else "-"
                samples = entry[5] if len(entry) > 5 else None
                dashed = ls != "-"
                if samples:
                    w = 0.055 * (max(leads) - min(leads))
                    for L, vals in zip(leads, samples):
                        if len(vals) < BOX_MIN_N:
                            continue
                        bp = ax.bxp([_box_stats(vals)], positions=[L], widths=w,
                                    patch_artist=True, showfliers=False,
                                    manage_ticks=False, zorder=2)
                        for p in bp["boxes"]:
                            p.set(facecolor=colour, edgecolor=colour, alpha=0.18,
                                  linewidth=0.7)
                        for k in ("whiskers", "caps", "medians"):
                            for a in bp[k]:
                                a.set(color=colour, linewidth=0.8, alpha=0.5)
                line, = ax.plot(leads, values, ls, color=colour, marker=marker, ms=5.5,
                                lw=1.7, mec="white", mew=0.8, label=label, zorder=5,
                                markerfacecolor="white" if dashed else colour,
                                alpha=0.9 if dashed else 1.0)
                seen.setdefault(label, line)
            ax.axhline(0, color="#888", lw=1, ls=":", zorder=1)
            ax.set_xticks(list(leads))
            ax.set_xlabel("lead time (h)", fontsize=9)
            ax.grid(True, ls="-", lw=0.4, color="#ececec")
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
    fig.suptitle(title, fontsize=12.5, fontweight="bold", y=1.01)
    n_lines = (max(1, math.ceil(len(caption) / max(1.0, 23.0 * fig.get_figwidth())))
               if caption else 0)
    cap_in = 0.17 * n_lines + 0.30            # caption text + legend row, in inches
    frac = min(0.34, cap_in / fig.get_figheight())
    fig.tight_layout(rect=[0, frac, 1, 0.99])
    handles = list(seen.values())
    if handles:
        fig.legend(handles, list(seen.keys()), frameon=False, fontsize=9,
                   loc="lower center", ncol=min(len(handles), 6),
                   bbox_to_anchor=(0.5, frac - 0.30 / fig.get_figheight()))
    if caption:
        fig.text(0.5, frac - 0.34 / fig.get_figheight(), caption,
                 ha="center", va="top", fontsize=8.5, wrap=True)
    return save(fig, outdir, name)


# ------------------------------------------------------------------ spectral mechanism test

SPECTRAL_FAMILY = "power spectrum"
SPECTRAL_VARS = (("Z500", "geopotential_500"), ("U500", "u_component_of_wind_500"))
SPECTRAL_SERIES = (("uniform W8A8 (floor)", "W8A8_floor", C["rmse"]),
                   ("physics-protected (span1)", "W8A8_span1", C["physics"]))
SPECTRAL_SERIES_WITH_REF = (("fp32 (reference)", "FP32", "#000000", "-"),
                            ("uniform W8A8 (all-int8)", "W8A8_floor", C["rmse"], "--"),
                            ("physics-protected (span1)", "W8A8_span1", C["physics"], "-"))


def prepare_spectral_test(psd_csv, amp_csv=None):
    rows = read_csv(psd_csv)
    if not rows:
        return {"panels": []}
    label_of = {tag: label for label, tag, _ in SPECTRAL_SERIES}
    colour_of = {tag: colour for _, tag, colour in SPECTRAL_SERIES}
    amp = {r["var"]: r for r in read_csv(amp_csv)} if amp_csv else {}
    panels = []
    for var_label, _ in SPECTRAL_VARS:
        sel = [r for r in rows if r.get("var") == var_label]
        if not sel:
            continue
        curves, k_ref = [], None
        for _, tag, colour in SPECTRAL_SERIES:
            pts = [r for r in sel if r.get("tag") == tag]
            if not pts:
                continue
            pts.sort(key=lambda r: float(r["k"]))
            k = np.array([float(r["k"]) for r in pts])
            curves.append((label_of.get(tag, tag), colour_of.get(tag, colour),
                           np.array([float(r["ratio"]) for r in pts])))
            k_ref = k if k_ref is None else k_ref
        if not curves:
            continue
        a = amp.get(var_label)
        panels.append({
            "var_label": var_label, "k": k_ref, "curves": curves,
            "plain": float(a["plain_frac"]) if a else None,
            "grad": float(a["grad_frac"]) if a else None,
            "amplification": float(a["amplification"]) if a else None})
    return {"panels": panels}


def prepare_balance_profile(path, series=SPECTRAL_SERIES_WITH_REF):
    rows = read_csv(path)
    if not rows:
        return []
    out = []
    for label, tag, colour, style_ in series:
        sel = [r for r in rows if r.get("tag") == tag]
        if not sel:
            continue
        sel.sort(key=lambda r: -float(r["level"]))
        out.append({"label": label, "colour": colour, "ls": style_,
                    "levels": np.array([float(r["level"]) for r in sel]),
                    "mean": np.array([float(r["mean"]) for r in sel]),
                    "lo": np.array([float(r["lo"]) for r in sel]),
                    "hi": np.array([float(r["hi"]) for r in sel])})
    return out


def draw_balance_profile(series, outdir, name, title, caption=""):
    """Absolute-severity companion to the SVR figures: the diagnostic in its own units."""
    if not series:
        print(f"  skipped {name} (no balance-profile extract)")
        return None
    fig, ax = plt.subplots(figsize=(6.8, 7.4))
    for s in series:
        ax.plot(s["mean"], s["levels"], s["ls"], color=s["colour"], marker="o", ms=5, lw=2,
                mec="white", mew=0.8, label=s["label"], zorder=5)
        ax.fill_betweenx(s["levels"], s["lo"], s["hi"],
                         color=s["colour"], alpha=0.15, zorder=2)
    levels = series[0]["levels"]
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.set_yticks(levels)
    ax.set_yticklabels([f"{int(v)}" for v in levels], fontsize=8)
    ax.set_ylabel("pressure level (hPa)")
    ax.set_xlabel("ageostrophic / geostrophic wind ratio  @120 h")
    ax.text(0.5, -0.085, "band = 95% moving-block bootstrap CI on the mean  "
            "(2000 resamples, monthly blocks)", transform=ax.transAxes,
            ha="center", va="top", fontsize=8.5, color="#444")
    ax.set_title(title, fontsize=11.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    style(ax)
    ax.grid(True, which="both", ls="-", lw=0.4, color="#ececec")
    fig.tight_layout()
    if caption:
        frac = min(0.34, 0.055 * (1 + len(caption) // 150))
        fig.subplots_adjust(bottom=frac)
        fig.text(0.5, frac * 0.5, caption, ha="center", va="top", fontsize=8.5, wrap=True)
    return save(fig, outdir, name)


def draw_spectral_test(data, outdir, name, title, caption=""):
    """Two log-x panels of PSD(config)/PSD(fp32). Returns the path, or None."""
    panels = data.get("panels") or []
    if not panels:
        print(f"  skipped {name} (no spectral extract -- run run_figure_pt_extracts.py; "
              f"the '{SPECTRAL_FAMILY}' family may also be absent from the .pt)")
        return None
    fig, axes = plt.subplots(1, len(panels), figsize=(6.3 * len(panels), 5.4), squeeze=False)
    for ax, p in zip(axes[0], panels):
        for label, colour, ratio in p["curves"]:
            ax.plot(p["k"], ratio, "-", color=colour, lw=2, label=label)
        ax.axhline(1.0, color="#888", ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("wavenumber k  (large scale -> small scale)")
        ax.set_ylabel(f"PSD(config) / PSD(fp32)   [{p['var_label']}]")
        ax.set_title(p["var_label"], fontsize=11, fontweight="bold")
        style(ax)
        ax.grid(True, which="both", ls="-", lw=0.4, color="#ececec")
        if p["plain"] is not None and p["grad"] is not None:
            same_sign = p["plain"] * p["grad"] > 0
            tail = (f"  (x{abs(p['amplification']):.0f})"
                    if same_sign and p["amplification"] is not None
                    else "  (opposite sign)")
            ax.annotate(f"floor: plain {100 * p['plain']:+.1f}%,  "
                        f"$k^2$-weighted {100 * p['grad']:+.1f}%{tail}",
                        xy=(0.02, 0.02), xycoords="axes fraction", fontsize=8.5,
                        color="#444")
    axes[0][0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle(title, fontsize=11.5, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if caption:
        frac = min(0.34, 0.055 * (1 + len(caption) // 150))
        fig.subplots_adjust(bottom=frac)
        fig.text(0.5, frac * 0.5, caption, ha="center", va="top", fontsize=8.5, wrap=True)
    return save(fig, outdir, name)


# ------------------------------------------------------------------ allocation-cost frontier

FRONTIER_A_PHYS = ("W8A8_floor", "W8A8_knee", "W8A8_span1", "W8A8_span2", "W8A8_span3")
FRONTIER_A_RMSE = ("W8A8_floor", "W8A8_rmse_knee", "W8A8_rmse_span1", "W8A8_rmse_span2",
                   "W8A8_rmse_span3")
FRONTIER_B_PHYS = ("W4W8_floor", "W4W8_knee", "W4W8_span1", "W4W8_span2", "W4W8_span3")
FRONTIER_B_RMSE = ("W4W8_floor", "W4W8_rmse_span1", "W4W8_rmse_span2", "W4W8_rmse_span3")
FRONTIER_RANDOM = ("W8A8_rand_0", "W8A8_rand_1", "W8A8_rand_2", "W8A8_rand_3")
FRONTIER_PROBES = ("W8A8_probe_divergent", "W8A8_probe_rmse_favoured")


def prepare_allocation_frontier(results_csv, axis="balance", exclude=()):
    rows = {r["tag"]: r for r in read_csv(results_csv)}
    exclude = set(exclude)

    def series(tags, need_cost=True, honour_exclude=True):
        """`need_cost=False` for the probe controls: they carry NO budget (predicted_cost is
        empty in harness_results.csv by construction), so requiring it drops them silently
        while the legend still advertises them. Their x is presentational either way."""
        out = []
        for t in tags:
            r = rows.get(t)
            if not r or (honour_exclude and t in exclude):
                continue
            try:
                y = float(r[axis])
            except (KeyError, ValueError, TypeError):
                continue
            try:
                x = float(r["predicted_cost"])
            except (KeyError, ValueError, TypeError):
                if need_cost:
                    continue
                x = None
            out.append((x, y))
        return out

    def dropped(tags):
        return series([t for t in dict.fromkeys(tags) if t in exclude], honour_exclude=False)

    return {"axis": axis, "excluded_tags": sorted(exclude),
            "panels": [
                {"title": "W8A8 family  (8-bit backbone)", "logx": False,
                 "xlabel": "allocation cost  (share of quantisable Linear FLOPs at bf16;  "
                           "0 = all-W8A8,  1 = all-bf16)",
                 "physics": series(FRONTIER_A_PHYS), "rmse": series(FRONTIER_A_RMSE),
                 "random": series(FRONTIER_RANDOM),
                 "probe": series(FRONTIER_PROBES, need_cost=False),
                 "excluded": dropped(FRONTIER_A_PHYS + FRONTIER_A_RMSE)},
                {"title": "W4W8 family  (4-bit backbone)", "logx": True,
                 "xlabel": "allocation cost  (model weight bytes, log;  6.3e8 = all-W4)",
                 "physics": series(FRONTIER_B_PHYS), "rmse": series(FRONTIER_B_RMSE),
                 "random": [], "probe": [],
                 "excluded": dropped(FRONTIER_B_PHYS + FRONTIER_B_RMSE)}],
            "ceiling": (series(("ceiling",)) or [(None, None)])[0][1]}


def draw_allocation_frontier(data, outdir, name, title, caption="", legend_loc="upper right"):
    from matplotlib.lines import Line2D
    panels = [p for p in data.get("panels", []) if p["physics"] or p["rmse"]]
    if not panels:
        print(f"  skipped {name} (no frontier tags in harness_results.csv)")
        return None
    fig, axes = plt.subplots(1, len(panels), figsize=(6.4 * len(panels), 5.6), squeeze=False)
    for ax, p in zip(axes[0], panels):
        for key, colour, marker, ls, lab in (
                ("physics", C["physics"], "o", "-", "physics-guided"),
                ("rmse", C["rmse"], "s", "--", "rmse-guided")):
            pts = p[key]
            if pts:
                ax.plot([x for x, _ in pts], [y for _, y in pts], ls, color=colour, lw=2,
                        marker=marker, ms=8, mec="white", mew=1, label=lab, zorder=5)
        for x, y in p.get("excluded", []):
            ax.scatter(x, y, facecolors="none", edgecolors=C["physics"], marker="o", s=95,
                       lw=1.6, ls=":", zorder=5)
        for x, y in p["random"]:
            ax.scatter(x, y, color=C["accent"], marker="^", s=70, ec="white", lw=1, zorder=4)
        for _, y in p["probe"]:
            ax.scatter(0.03, y, color=C["warn"], marker="X", s=80, ec="white", lw=0.8, zorder=6)
        if p["logx"]:
            ax.set_xscale("log")
        ax.set_yscale("log")
        if data.get("ceiling"):
            ax.axhline(data["ceiling"], color="#888", ls=":", lw=1.3, zorder=1)
        ax.set_title(p["title"], fontsize=12, fontweight="bold")
        ax.set_xlabel(p["xlabel"], fontsize=9)
        ax.set_ylabel(f"measured {data['axis']} distortion @120 h  (SVR, log; lower = better)")
        style(ax)
        ax.grid(True, which="both", ls="-", lw=0.4, color="#e8e8e8")
    handles = [Line2D([], [], color=C["physics"], marker="o", lw=2, mec="white",
                      label="physics-guided"),
               Line2D([], [], color=C["rmse"], marker="s", ls="--", lw=2, mec="white",
                      label="rmse-guided"),
               Line2D([], [], color=C["accent"], marker="^", ls="", mec="white",
                      label="random (control)"),
               Line2D([], [], color=C["warn"], marker="X", ls="", label="probe (control)")]
    if any(p.get("excluded") for p in panels):
        handles.append(Line2D([], [], color=C["physics"], marker="o", ls="", mfc="none",
                              mew=1.6, label="off-scalarisation (excluded)"))
    pad = 1.6 if legend_loc.startswith("lower") else 0.5
    axes[0][0].legend(handles=handles, frameon=False, fontsize=9, loc=legend_loc,
                      borderaxespad=pad)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    if caption:
        frac = min(0.34, 0.055 * (1 + len(caption) // 150))
        fig.subplots_adjust(bottom=frac)
        fig.text(0.5, frac * 0.5, caption, ha="center", va="top", fontsize=8.5, wrap=True)
    return save(fig, outdir, name)


# ------------------------------------------------------------------ per-variable RMSE

def prepare_per_variable_rho(path, jackknife_path=None):
    rows = [r for r in read_csv(path) if r.get("var")]
    jack = {}
    for j in (read_csv(jackknife_path) if jackknife_path else []):
        try:
            jack.setdefault(j["var"], []).append(float(j["rho"]))
        except (KeyError, ValueError, TypeError):
            continue
    out = []
    for r in rows:
        try:
            out.append({"var": r["var"], "rho": float(r["rho"]),
                        "jack": jack.get(r["var"], [])})
        except (KeyError, ValueError, TypeError):
            continue
    return sorted(out, key=lambda d: d["rho"])


def draw_per_variable_rho(rows, outdir, name, title, caption=""):
    if not rows:
        print(f"  skipped {name} (no per-variable rho CSV)")
        return None
    fig, ax = plt.subplots(figsize=(8.4, 0.46 * len(rows) + 2.6))
    y = np.arange(len(rows))
    jrng = np.random.default_rng(0)     # seeded: unseeded jitter diffs on every rebuild
    colour = C["physics"]
    for i, r in enumerate(rows):
        _draw_row_distribution(ax, r.get("jack") or [], i, colour, "o", 0.45,
                               width=0.34, rng=jrng, dim=True)
        ax.scatter([r["rho"]], [i], s=150, marker="o", zorder=6, color="white",
                   edgecolor="none")
        ax.scatter([r["rho"]], [i], s=64, marker="o", zorder=7, linewidth=1.6,
                   edgecolor=colour, facecolor=colour)
    ax.axvline(0, color="#444", lw=1, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([r["var"] for r in rows], fontsize=10)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel(r"Spearman $\rho$(RMSE$_v$, balance) across configs")
    ax.set_title(title, fontsize=11.5, fontweight="bold")
    style(ax)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=colour, marker="o", ls="",
                              mec=colour, label=r"$\rho$ over all configs"),
                       Line2D([], [], color=colour, marker="o", ls="", ms=4.5,
                              mec=colour, alpha=0.6, markerfacecolor="none",
                              label="leave-one-config-out")],
              frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    if caption:
        frac = min(0.30, 0.042 * (1 + len(caption) // 150))
        fig.subplots_adjust(bottom=frac)
        fig.text(0.5, frac * 0.72, caption, ha="center", va="top", fontsize=8.5, wrap=True)
    return save(fig, outdir, name)


# ------------------------------------------------- the reproducibility floor

def prepare_reproducibility_floor(floor_csv, scheme_csvs, lead=120,
                                  axes=("balance", "conservation")):
    floor = {r["axis"]: r for r in read_csv(floor_csv)}
    if not floor:
        return {}
    from allocator import load_distortion_table
    present = {k: v for k, v in scheme_csvs.items() if os.path.exists(v)}
    if not present:
        return {}
    d = load_distortion_table(present, lead=lead, axes=tuple(axes))
    out = {"lead": lead, "axes": {}}
    for a in axes:
        if a not in floor:
            continue
        vals = []
        for g in d:
            for prec in d[g]:
                if prec == "bf16":
                    continue
                v = d[g][prec].get(a)
                if v is not None and np.isfinite(v) and v > 0:
                    vals.append({"group": g, "precision": prec, "value": float(v)})
        if vals:
            out["axes"][a] = {"cells": sorted(vals, key=lambda r: r["value"]),
                              "p95": float(floor[a]["p95"]),
                              "median": float(floor[a]["median"]),
                              "n_members": floor[a].get("n_members", "?")}
    return out


def draw_reproducibility_floor(data, outdir, name, title):
    """Strip plot per axis: every group x precision cell against the measured floor band."""
    axes_d = (data or {}).get("axes") or {}
    if not axes_d:
        print(f"  skipped {name} (no axis_null_p95.csv -- run derive_axis_null.py --out)")
        return None
    keys = [a for a in ("balance", "conservation") if a in axes_d]
    fig, axs = plt.subplots(1, len(keys), figsize=(5.9 * len(keys), 3.9), squeeze=False)
    precisions = sorted({c["precision"] for a in keys for c in axes_d[a]["cells"]})
    palette = {p: col for p, col in zip(
        precisions, (C["physics"], C["rmse"], C["accent"], C["warn"], C["neutral"]))}
    for ax, a in zip(axs[0], keys):
        info = axes_d[a]
        rng = np.random.default_rng(0)          # fixed jitter: the figure must be stable
        for c in info["cells"]:
            ax.scatter(c["value"], rng.uniform(-0.28, 0.28),
                       color=palette.get(c["precision"], C["neutral"]), s=34, alpha=0.85,
                       ec="white", lw=0.5, zorder=5)
        ax.axvspan(1e-6, info["p95"], color="#bbbbbb", alpha=0.35, zorder=0)
        ax.axvline(info["p95"], color="#666", ls="--", lw=1.2, zorder=2)
        n_below = sum(1 for c in info["cells"] if c["value"] <= info["p95"])
        ax.set_xscale("log")
        ax.set_ylim(-0.55, 0.75)
        ax.set_yticks([])
        ax.set_xlabel(f"{a} distortion (SVR, log)")
        ax.set_title(f"{a}   -   {n_below}/{len(info['cells'])} cells below the floor",
                     fontsize=11, fontweight="bold")
        # .4f prints 0.0000 for Stormer's 4.6e-05 floor; .3g keeps both models legible.
        ax.annotate(f"reproducibility floor p95 = {info['p95']:.3g}", xy=(info["p95"], 0.60),
                    xytext=(4, 0), textcoords="offset points", fontsize=8.5, color="#444")
        style(ax)
        ax.grid(True, axis="x", which="both", ls="-", lw=0.4, color="#ececec")
    from matplotlib.lines import Line2D
    axs[0][0].legend(handles=[Line2D([], [], color=palette[p], marker="o", ls="", mec="white",
                                     label=p) for p in precisions],
                     frameon=False, fontsize=9, loc="upper left")
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return save(fig, outdir, name)


# ------------------------------------------------------------------ anticontrol

def prepare_anticontrol(balance_csv, composite_csv, balance_labels_csv=None,
                        composite_labels_csv=None, labels_family=None, labels_lead=None):
    labels_by_endpoint = {
        "wind_balance": read_label_samples(balance_labels_csv, "pair", "contribution",
                                           family=labels_family, lead=labels_lead),
        # No family filter: the composite file's binding family varies BY PAIR by design.
        "max(balance, conservation)": read_label_samples(
            composite_labels_csv, "pair", "contribution", lead=labels_lead),
    }
    out = []
    for path, endpoint in ((balance_csv, "wind_balance"),
                           (composite_csv, "max(balance, conservation)")):
        labels = labels_by_endpoint[endpoint]
        for r in read_csv(path):
            try:
                out.append({"pair": r["pair"], "endpoint": endpoint,
                            "ratio": float(r["ratio"]),
                            "samples": labels.get(r["pair"], []),
                            "excess": float(r["cost_excess_pct"])
                            if r.get("cost_excess_pct") else None})
            except (KeyError, ValueError, TypeError):
                continue
    return out


def draw_anticontrol(rows, outdir, name, title, caption=""):
    if not rows:
        print(f"  skipped {name} (no anticontrol CSVs)")
        return None
    endpoints = []
    for r in rows:
        if r["endpoint"] not in endpoints:
            endpoints.append(r["endpoint"])
    fig, axs = plt.subplots(1, len(endpoints), figsize=(6.3 * len(endpoints), 4.6),
                            squeeze=False, sharey=True)
    pairs = []
    for r in rows:
        short = r["pair"].split("|")[0]
        if short not in pairs:
            pairs.append(short)
    for ax, ep in zip(axs[0], endpoints):
        sel = [r for r in rows if r["endpoint"] == ep]
        rng = np.random.default_rng(0)
        for r in sel:
            i = pairs.index(r["pair"].split("|")[0])
            win = r["ratio"] > 1
            colour = C["physics"] if win else C["rmse"]
            _draw_strips(ax, [r.get("samples")], [i], colour, width=0.34, rng=rng)
            ax.scatter([r["ratio"]], [i], s=64, marker="o", color=colour,
                       edgecolor="white", linewidth=1.0, zorder=5)
            if r["excess"] is not None:
                ax.annotate(f"+{r['excess']:.0f}% budget", xy=(r["ratio"], i),
                            xytext=(0, -14), textcoords="offset points", ha="center",
                            fontsize=7.5, color="#666")
        ax.axvline(1.0, color="#444", lw=1.1, zorder=1)
        ax.set_xscale("log")
        ax.set_yticks(range(len(pairs)))
        ax.set_yticklabels(pairs, fontsize=9)
        ax.set_ylim(-0.7, len(pairs) - 0.3)
        ax.set_xlabel("physics-guided advantage  (>1 = physics less distorted, log)")
        ax.set_title(ep, fontsize=11, fontweight="bold")
        ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0),
                                                      numticks=12))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _pos: f"{v:g}" if v >= 0.01 else f"{v:.3g}"))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        style(ax)
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if caption:
        frac = min(0.4, 0.062 * (1 + len(caption) // 140))
        fig.subplots_adjust(bottom=frac)
        fig.text(0.5, frac * 0.5, caption, ha="center", va="top", fontsize=8.5, wrap=True)
    return save(fig, outdir, name)


# ------------------------------------------------------------------ across quantisers

def prepare_cross_quantiser(path, lead=120, axis="balance_share", top_n=8):
    rows = [r for r in read_csv(path) if r.get("lead") and int(float(r["lead"])) == lead]
    if not rows:
        return {}
    by_scheme = {}
    for r in rows:
        try:
            by_scheme.setdefault(r["scheme"], {})[r["group"]] = float(r[axis])
        except (KeyError, ValueError, TypeError):
            continue
    if not by_scheme:
        return {}
    schemes = [s for s in SCHEME_ORDER if s in by_scheme] + \
              sorted(set(by_scheme) - set(SCHEME_ORDER))
    mean_share = {}
    for g in {g for v in by_scheme.values() for g in v}:
        vals = [v[g] for v in by_scheme.values() if g in v]
        mean_share[g] = sum(vals) / len(vals)
    groups = [g for g, _ in sorted(mean_share.items(), key=lambda kv: -kv[1])[:top_n]]
    return {"schemes": schemes, "groups": groups, "by_scheme": by_scheme,
            "axis": axis, "lead": lead}


def draw_cross_quantiser(data, outdir, name, title, caption=""):
    if not data or not data.get("groups"):
        print(f"  skipped {name} (no cross_scheme.csv)")
        return None
    schemes, groups = data["schemes"], data["groups"]
    fig, ax = plt.subplots(figsize=(9.2, 0.52 * len(groups) + 2.8))
    h = 0.8 / len(schemes)
    palette = (C["physics"], C["rmse"], C["accent"], C["warn"], C["neutral"])
    for si, s in enumerate(schemes):
        vals = [data["by_scheme"][s].get(g, 0.0) for g in groups]
        y = [i + (si - (len(schemes) - 1) / 2) * h for i in range(len(groups))]
        ax.barh(y, vals, height=h * 0.92, color=palette[si % len(palette)], alpha=0.9,
                label=s, zorder=4)
    ax.axvline(0, color="#444", lw=0.9, zorder=1)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(f"share of physics damage carried by the group  "
                  f"({data['axis'].replace('_', ' ')}, {data['lead']} h)")
    ax.set_title(title, fontsize=11.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=9, ncol=len(schemes), loc="lower right")
    style(ax)
    fig.tight_layout()
    if caption:
        frac = min(0.4, 0.06 * (1 + len(caption) // 140))
        fig.subplots_adjust(bottom=frac)
        fig.text(0.5, frac * 0.55, caption, ha="center", va="top", fontsize=8.5, wrap=True)
    return save(fig, outdir, name)


# ------------------------------------------------------- uniform-scheme scorecard

SCORECARD_VMAX = 50.0
"""Colour clip for the scorecard, in percent. Matches the retired compare_runs.py"""


def scorecard_pct_text(v):
    """Cell text for a percent change, narrow enough not to overrun the cell."""
    if not np.isfinite(v):
        return ""
    if abs(v) < 0.5:
        return "0"
    if abs(v) < 1000:
        return f"{v:+.0f}"
    if abs(v) < 10000:
        return f"{v / 1000:+.1f}k"
    return f"{v / 1000:+.0f}k"


def prepare_scorecard(csv_path, card):
    """One card of a scorecard.csv, grouped for draw_scorecard."""
    rows = [r for r in read_csv(csv_path)
            if r.get("block") and r.get("card") == card]
    if not rows:
        return None
    order = {}
    for r in rows:
        order[int(r["order"])] = (r["metric"], r["block"])
    labels = [order[k][0] for k in sorted(order)]
    blocks = []
    for k in sorted(order):
        name = order[k][1]
        if blocks and blocks[-1][0] == name:
            blocks[-1][2] = k + 1
        else:
            blocks.append([name, k, k + 1])

    leads = sorted({int(r["lead"]) for r in rows})
    schemes = list(dict.fromkeys(r["scheme"] for r in rows))
    grid, floor = {}, {}
    for s in schemes:
        grid[s] = np.full((len(labels), len(leads)), np.nan)
        floor[s] = np.zeros_like(grid[s], dtype=bool)
    for r in rows:
        i, j = int(r["order"]), leads.index(int(r["lead"]))
        try:
            grid[r["scheme"]][i, j] = float(r["pct_diff"])
        except ValueError:
            continue
        floor[r["scheme"]][i, j] = _truthy(r["at_floor"])
    n_measured = sum(1 for r in rows if _truthy(r.get("floor_measured")))
    return {"labels": labels, "blocks": [tuple(b) for b in blocks], "leads": leads,
            "schemes": schemes, "grid": grid, "floor": floor,
            "n_measured": n_measured, "n_cells": len(rows)}


def draw_scorecard(data, outdir, name, title, caption=""):
    """Percent change versus FP32: rows = metrics, columns = leads, one panel per scheme."""
    caption = _cap(caption)
    if data is None:
        print(f"  skipped {name} (no scorecard rows)")
        return None
    labels, leads, schemes = data["labels"], data["leads"], data["schemes"]
    n = len(schemes)
    fig, axes = plt.subplots(1, n, figsize=(2.35 * n + 2.2, 0.30 * len(labels) + 2.6),
                             sharey=True, squeeze=False)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#f2f2f2")                    # nan: FP32 baseline is exactly 0

    im = None
    for ax, s in zip(axes.flat, schemes):
        g, fl = data["grid"][s], data["floor"][s]
        im = ax.imshow(np.clip(g, -SCORECARD_VMAX, SCORECARD_VMAX), cmap=cmap,
                       vmin=-SCORECARD_VMAX, vmax=SCORECARD_VMAX, aspect="auto")
        for i in range(len(labels)):
            for j in range(len(leads)):
                if np.isnan(g[i, j]):
                    continue
                strong = abs(g[i, j]) > SCORECARD_VMAX * 0.6
                ax.text(j, i - 0.10, scorecard_pct_text(g[i, j]), ha="center",
                        va="center", fontsize=6.4,
                        color="white" if strong else "#111")
                if fl[i, j]:
                    ax.plot(j, i + 0.29, marker="o", ms=1.9,
                            color="white" if strong else "#333", lw=0)
        for _, start, stop in data["blocks"][:-1]:
            ax.axhline(stop - 0.5, color="#333", lw=0.9)
        for j in range(len(leads) - 1):
            ax.axvline(j + 0.5, color="#ffffff", lw=0.6)
        ax.set_xticks(range(len(leads)))
        ax.set_xticklabels([f"+{lt}h" for lt in leads], fontsize=8)
        ax.set_title(s, fontsize=9.5, fontweight="bold")
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)

    axes.flat[0].set_yticks(range(len(labels)))
    axes.flat[0].set_yticklabels(labels, fontsize=7.2)
    last = axes.flat[-1]
    for bname, start, stop in data["blocks"]:
        last.text(len(leads) - 0.34, (start + stop - 1) / 2.0, bname, rotation=270,
                  ha="left", va="center", fontsize=7.4, fontweight="bold",
                  color="#444", clip_on=False)

    cb = fig.colorbar(im, ax=axes, shrink=0.55, pad=0.06,
                      label=f"% change vs FP32  (red = larger, clipped at "
                            f"\u00b1{SCORECARD_VMAX:.0f}%)")
    cb.outline.set_visible(False)
    fig.suptitle(title, fontsize=11.5, fontweight="bold")
    if data.get("n_measured"):
        note = ("dot = mean shift within the measured numerical noise floor for that "
                "metric and lead")
        if data["n_measured"] < data.get("n_cells", 0):
            note += ("   (floors exist for the registry metrics only; "
                     f"{data['n_measured']} of {data['n_cells']} cells)")
        fig.text(0.5, 0.012, note, ha="center", va="bottom", fontsize=7.6, color="#555")
    if caption:
        fig.text(0.5, -0.02, caption, ha="center", va="top", fontsize=8.5, wrap=True)
    return save(fig, outdir, name)
