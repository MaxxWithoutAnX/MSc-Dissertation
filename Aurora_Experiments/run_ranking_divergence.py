# run_ranking_divergence.py
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ranking_divergence import (sensitivities, spearman_rho, permutation_pvalue,
                                rank_table, rho_matrix)

SCHEME_CSV = {s: {s: f"ablation_analysis_ablations_{s}/sensitivity.csv"}
              for s in ("W8", "W4", "W8A8", "W8A8_sq")}


def _log_floor(*dicts):
    """Positive floor for log axes: half the smallest positive value across the data
    (zeros/exact-0 sensitivities are drawn at the floor so log scaling is well-defined)."""
    vals = [v for d in dicts for v in d.values() if v > 0]
    return 0.5 * min(vals) if vals else 1e-6


def _scatter(ax, phys, rmse, rho, pval):
    groups = list(phys)
    floor = _log_floor(phys, rmse)
    x = np.array([max(rmse[g], floor) for g in groups])
    y = np.array([max(phys[g], floor) for g in groups])
    mx, my = np.median(x), np.median(y)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.scatter(x, y, s=28, color="tab:blue", zorder=3)
    for g, xi, yi in zip(groups, x, y):
        ax.annotate(g, (xi, yi), fontsize=6.5, xytext=(3, 3),
                    textcoords="offset points")
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()               # after autoscale on the data
    ax.fill_between([x0, mx], my, y1, color="tab:red", alpha=0.08, zorder=0)
    ax.axvline(mx, color="0.7", lw=0.7, ls=":"); ax.axhline(my, color="0.7", lw=0.7, ls=":")
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.text(0.03, 0.97, "RMSE-blind\ndanger zone", transform=ax.transAxes,
            fontsize=7, color="tab:red", va="top", ha="left")
    ax.set_xlabel("RMSE sensitivity (|SVR|)")
    ax.set_ylabel("physics sensitivity (Σ|SVR| balance+conservation)")
    ax.set_title(f"per-group sensitivity  (Spearman ρ={rho:+.2f}, p={pval:.3f})")
    ax.grid(alpha=0.25, which="both")


def _slopegraph(ax, phys, rmse, swap_thresh=5.0):
    rows = rank_table(phys, rmse)
    n = len(rows)
    for r in rows:
        big = abs(r["rank_delta"]) >= swap_thresh
        ax.plot([0, 1], [r["rank_physics"], r["rank_rmse"]],
                color=("tab:red" if big else "0.75"),
                lw=(1.6 if big else 0.8), zorder=(3 if big else 1))
        ax.annotate(r["group"], (0, r["rank_physics"]), fontsize=6.5,
                    ha="right", va="center", xytext=(-4, 0), textcoords="offset points")
        ax.annotate(r["group"], (1, r["rank_rmse"]), fontsize=6.5,
                    ha="left", va="center", xytext=(4, 0), textcoords="offset points")
    ax.set_xlim(-0.55, 1.55); ax.set_ylim(n + 0.5, 0.5)          # rank 1 at top
    ax.set_xticks([0, 1]); ax.set_xticklabels(["physics rank", "RMSE rank"])
    ax.set_yticks([1, n]); ax.set_yticklabels(["most\nsensitive", "least"])
    ax.set_title("protection ordering (red = swaps >= %g ranks)" % swap_thresh)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "physics", "rmse", "rank_physics", "rank_rmse", "rank_delta"])
        for r in sorted(rows, key=lambda r: r["rank_delta"]):
            w.writerow([r["group"], f"{r['physics']:.6g}", f"{r['rmse']:.6g}",
                        f"{r['rank_physics']:.1f}", f"{r['rank_rmse']:.1f}",
                        f"{r['rank_delta']:+.1f}"])


def run(scheme="W8", scheme_csvs=None, lead=120, outdir="."):
    """Two-panel figure + per-group CSV for one scheme."""
    os.makedirs(outdir, exist_ok=True)
    scheme_csvs = scheme_csvs or SCHEME_CSV[scheme]
    phys, rmse = sensitivities(scheme_csvs, scheme, lead=lead)
    rho = spearman_rho(phys, rmse)
    pval = permutation_pvalue(phys, rmse)

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 5.6))
    _scatter(a0, phys, rmse, rho, pval)
    _slopegraph(a1, phys, rmse)
    fig.suptitle(f"Physics vs RMSE layer sensitivity - {scheme} @ {lead}h")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"ranking_divergence_{scheme}.png"), dpi=140)
    plt.close(fig)

    _write_csv(os.path.join(outdir, f"ranking_sensitivities_{scheme}.csv"),
               rank_table(phys, rmse))
    return {"rho": rho, "pval": pval, "phys": phys, "rmse": rmse}


def rho_heatmap(schemes=("W8", "W4", "W8A8", "W8A8_sq"), lead=120, outdir="."):
    """Scheme x reducer-variant Spearman-rho heatmap (robustness of the divergence)."""
    os.makedirs(outdir, exist_ok=True)
    m = rho_matrix({s: SCHEME_CSV[s] for s in schemes}, lead=lead)
    variants = list(next(iter(m.values())))
    rows = list(m)
    grid = np.array([[m[s][v] for v in variants] for s in rows])
    fig, ax = plt.subplots(figsize=(1.6 + 1.2 * len(variants), 1.2 + 0.5 * len(rows)))
    im = ax.imshow(grid, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(variants))); ax.set_xticklabels(variants, rotation=35, ha="right", fontsize=7)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows)
    for i in range(len(rows)):
        for j in range(len(variants)):
            ax.text(j, i, f"{grid[i, j]:+.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(grid[i, j]) > 0.55 else "black")
    fig.colorbar(im, ax=ax, label="Spearman ρ (physics vs RMSE)")
    ax.set_title("Ranking divergence robustness across schemes × reducers")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "ranking_rho_matrix.png"), dpi=140)
    plt.close(fig)
    return m


def main():
    res = run("W8", outdir=".")
    print(f"W8: rho={res['rho']:+.3f}  p={res['pval']:.3f}")
    m = rho_heatmap(outdir=".")
    for s, row in m.items():
        print(f"  {s:8s} " + "  ".join(f"{v}={rho:+.2f}" for v, rho in row.items()))


if __name__ == "__main__":
    main()
