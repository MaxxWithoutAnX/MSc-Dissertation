import os, re, csv
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

key = "_28"

HERE = os.path.dirname(os.path.abspath(__file__))
STATS = os.path.join(HERE, f"activation_stats{key}.pt")
TOP_N = 30

stats = torch.load(STATS, weights_only=False)
rows = [(name, s) for name, s in stats.items()]


def group_of(name):
    """Coarse module family from the fully-qualified name, for colouring."""
    n = name.lower()
    if "encoder" in n:
        base = "encoder"
    elif "decoder" in n:
        base = "decoder"
    elif "backbone" in n or "layers" in n or "blocks" in n:
        base = "backbone"
    else:
        base = "other"
    if re.search(r"qkv|\.q\b|\.k\b|\.v\b|attn|attention", n):
        kind = "attn"
    elif re.search(r"mlp|fc|ffn|feed", n):
        kind = "mlp"
    elif "proj" in n:
        kind = "proj"
    else:
        kind = "misc"
    return f"{base}/{kind}"


for name, s in rows:
    s["group"] = group_of(name)

# --- CSV dump ------------------------------------------------------------------
cols = ["layer", "group", "C", "tensor_max", "outlier_ratio",
        "kurtosis", "frac_outlier_ch"]
with open(os.path.join(HERE, f"activation_stats_table{key}.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(cols)
    for name, s in sorted(rows, key=lambda r: r[1]["outlier_ratio"], reverse=True):
        w.writerow([name, s["group"], s["C"], f"{s['tensor_max']:.4g}",
                    f"{s['outlier_ratio']:.4g}", f"{s['kurtosis']:.4g}",
                    f"{s['frac_outlier_ch']:.4g}"])
print(f"wrote activation_stats_table{key}.csv")

# --- 1. top-N ranking bar ------------------------------------------------------
top = sorted(rows, key=lambda r: r[1]["outlier_ratio"], reverse=True)[:TOP_N]
groups = sorted({s["group"] for _, s in rows})
cmap = {g: plt.cm.tab10(i % 10) for i, g in enumerate(groups)}

fig, ax = plt.subplots(figsize=(10, max(6, 0.32 * len(top))))
y = np.arange(len(top))[::-1]
ax.barh(y, [s["outlier_ratio"] for _, s in top],
        color=[cmap[s["group"]] for _, s in top])
ax.set_yticks(y)
ax.set_yticklabels([n[-48:] for n, _ in top], fontsize=7)
ax.set_xlabel("outlier_ratio  (tensor_max / median channel-max)")
ax.set_title(f"Top {TOP_N} activation-outlier layers (higher = harder to int8-quantise)")
ax.axvline(20, color="grey", ls="--", lw=1)  # rough SmoothQuant 'needs smoothing' line
handles = [plt.Rectangle((0, 0), 1, 1, color=cmap[g]) for g in groups]
ax.legend(handles, groups, fontsize=7, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(HERE, f"activation_outlier_ranking{key}.png"), dpi=150)
print(f"wrote activation_outlier_ranking{key}.png")

# --- 2. ratio vs kurtosis scatter ---------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
for g in groups:
    pts = [(s["kurtosis"], s["outlier_ratio"]) for _, s in rows if s["group"] == g]
    if pts:
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, s=18, color=cmap[g], label=g, alpha=0.7)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("kurtosis (Gaussian = 3)")
ax.set_ylabel("outlier_ratio")
ax.set_title("Outlier severity vs tail heaviness, per layer")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(HERE, f"activation_ratio_vs_kurtosis{key}.png"), dpi=150)
print(f"wrote activation_ratio_vs_kurtosis{key}.png")

# --- 3. per-group box plot -----------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
data = [[s["outlier_ratio"] for _, s in rows if s["group"] == g] for g in groups]
ax.boxplot(data, labels=groups, showfliers=True)
ax.set_yscale("log")
ax.set_ylabel("outlier_ratio")
ax.set_title("Outlier_ratio distribution by module family")
plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(HERE, f"activation_group_summary{key}.png"), dpi=150)
print(f"wrote activation_group_summary{key}.png")

# --- console summary -----------------------------------------------------------
ratios = np.array([s["outlier_ratio"] for _, s in rows])
print(f"\n{len(rows)} layers profiled")
print(f"outlier_ratio: median={np.median(ratios):.1f}  "
      f"max={ratios.max():.1f}  p90={np.percentile(ratios, 90):.1f}")
n_bad = int((ratios > 20).sum())
print(f"{n_bad} layers with outlier_ratio > 20 (candidates to keep in higher precision / smooth)")
