import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

C = {"physics":"#0072B2", "rmse":"#D55E00", "random":"#009E73", "probe":"#CC79A7"}
M = {"physics":"o", "rmse":"s", "random":"^", "probe":"X"}
CEIL = 0.0392656  # bf16 measurement noise floor

rows=[]
with open("harness_results/harness_results.csv") as f:
    for r in csv.DictReader(f):
        for k in ("predicted_cost","balance","conservation"):
            r[k]=float(r[k]) if r[k] not in ("",None) else None
        rows.append(r)
by={r["tag"]:r for r in rows}

fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 5.4))

# ---- Panel A: W8A8 family, cost already normalised 0..1 ----
phys=["W8A8_floor","W8A8_knee","W8A8_span1","W8A8_span2","W8A8_span3"]
rmse=["W8A8_floor","W8A8_rmse_knee","W8A8_rmse_span1","W8A8_rmse_span2","W8A8_rmse_span3"]
axA.plot([by[t]["predicted_cost"] for t in phys],[by[t]["balance"] for t in phys],
         "-", color=C["physics"], lw=2, marker=M["physics"], ms=8, mec="white", mew=1,
         label="physics-guided", zorder=5)
axA.plot([by[t]["predicted_cost"] for t in rmse],[by[t]["balance"] for t in rmse],
         "--", color=C["rmse"], lw=2, marker=M["rmse"], ms=8, mec="white", mew=1,
         label="rmse-guided", zorder=5)
for t in ("W8A8_rand_0","W8A8_rand_1","W8A8_rand_2","W8A8_rand_3"):
    axA.scatter(by[t]["predicted_cost"],by[t]["balance"],color=C["random"],marker=M["random"],
                s=70,ec="white",lw=1,zorder=4,label="random")
for t in ("W8A8_probe_divergent","W8A8_probe_rmse_favoured"):
    axA.scatter(0.03,by[t]["balance"],color=C["probe"],marker=M["probe"],s=80,ec="white",lw=0.8,
                zorder=6,label="probe")
axA.set_title("W8A8 family  (8-bit backbone)", fontsize=12, fontweight="bold")
axA.set_xlabel("allocation cost  (share of quantisable Linear FLOPs at bf16;  0 = all-W8A8,  1 = all-bf16)")

# ---- Panel B: W4W8 family, cost in FLOP-ish units -> log x ----
phys4=["W4W8_floor","W4W8_knee","W4W8_span1","W4W8_span2","W4W8_span3"]
rmse4=["W4W8_span1","W4W8_rmse_span1","W4W8_rmse_span2","W4W8_rmse_span3"]  # rmse spans + shared span1 pt not ideal; plot rmse alone
rmse4=["W4W8_rmse_span1","W4W8_rmse_span2","W4W8_rmse_span3"]
axB.plot([by[t]["predicted_cost"] for t in phys4],[by[t]["balance"] for t in phys4],
         "-", color=C["physics"], lw=2, marker=M["physics"], ms=8, mec="white", mew=1, zorder=5)
axB.plot([by[t]["predicted_cost"] for t in rmse4],[by[t]["balance"] for t in rmse4],
         "--", color=C["rmse"], lw=2, marker=M["rmse"], ms=8, mec="white", mew=1, zorder=5)
axB.set_xscale("log")
axB.set_title("W4W8 family  (4-bit backbone)", fontsize=12, fontweight="bold")
axB.set_xlabel("allocation cost  (model weight bytes, log;  6.3e8 = all-W4)")

for ax in (axA, axB):
    ax.set_yscale("log")
    ax.axhline(CEIL, color="#888", ls=":", lw=1.3, zorder=1)
    ax.set_ylabel("measured balance distortion @ 120 h  (log, lower = better)")
    ax.grid(True, which="both", ls="-", lw=0.4, color="#e8e8e8", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top","right"): ax.spines[s].set_visible(False)
axA.text(0.02, CEIL*1.15, "bf16 ceiling", fontsize=8, color="#666")

handles=[Line2D([],[],color=C["physics"],marker=M["physics"],lw=2,label="physics-guided",mec="white"),
         Line2D([],[],color=C["rmse"],marker=M["rmse"],ls="--",lw=2,label="rmse-guided",mec="white"),
         Line2D([],[],color=C["random"],marker=M["random"],ls="",label="random (control)",mec="white"),
         Line2D([],[],color=C["probe"],marker=M["probe"],ls="",label="probe (control)")]
axA.legend(handles=handles, frameon=False, fontsize=9, loc="upper right")

fig.suptitle("Measured physics-distortion frontier",
             fontsize=13, fontweight="bold", y=0.99)
fig.tight_layout(rect=[0,0,1,0.96])
os.makedirs("plots_harness_wpd", exist_ok=True)
out="plots_harness_wpd/pareto_cost_vs_distortion.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
print("wrote", out)
