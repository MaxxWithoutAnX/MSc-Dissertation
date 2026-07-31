import csv, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

C={"physics":"#0072B2","rmse":"#D55E00","random":"#009E73","probe":"#CC79A7"}
M={"physics":"o","rmse":"s","random":"^","probe":"X"}
BAL_FLOOR=0.00284          # normalised round-off floor @120h (raw - floored)

def load(p):
    with open(p) as f: return {r["tag"]:r for r in csv.DictReader(f)}
man=load("harness_results/harness_results.csv")          # predicted_cost
flo=load("harness_results/harness_results_floored.csv")  # official points
ci =load("harness_results/harness_results_ci.csv")       # raw-IQR pt + CI

def pt(tag):  return float(flo[tag]["balance"])
def cost(tag):return float(man[tag]["predicted_cost"])
def bars(tag):
    """official point with asymmetric bars via relative CI from the raw bootstrap."""
    r=ci[tag]; rp=float(r["bal_pt"]); lo=float(r["bal_lo"]); hi=float(r["bal_hi"])
    p=pt(tag); rl=lo/rp if rp>0 else 1; rh=hi/rp if rp>0 else 1
    return p, max(p-p*rl,1e-4), p*rh-p     # (y, lower_err, upper_err)

fig,(axA,axB)=plt.subplots(1,2,figsize=(12.8,5.6))

def frontier(ax, tags, color, marker, ls, label):
    xs=[cost(t) for t in tags]; ys=[]; el=[]; eh=[]
    for t in tags:
        y,lo,hi=bars(t); ys.append(y); el.append(lo); eh.append(hi)
    ax.errorbar(xs,ys,yerr=[el,eh],fmt=marker,ls=ls,color=color,lw=2,ms=8,mec="white",
                mew=1,capsize=3,elinewidth=1.2,ecolor=color,label=label,zorder=5)

# Panel A: W8A8
frontier(axA,["W8A8_floor","W8A8_knee","W8A8_span1","W8A8_span2","W8A8_span3"],
         C["physics"],M["physics"],"-","physics-guided")
frontier(axA,["W8A8_floor","W8A8_rmse_knee","W8A8_rmse_span1","W8A8_rmse_span2","W8A8_rmse_span3"],
         C["rmse"],M["rmse"],"--","rmse-guided")
for t in ("W8A8_rand_0","W8A8_rand_1","W8A8_rand_2","W8A8_rand_3"):
    y,lo,hi=bars(t); axA.errorbar(cost(t),y,yerr=[[lo],[hi]],fmt=M["random"],color=C["random"],
                                  ms=7,mec="white",mew=1,capsize=2,elinewidth=1,ecolor=C["random"],zorder=4)
for t in ("W8A8_probe_divergent","W8A8_probe_rmse_favoured"):
    y,lo,hi=bars(t); axA.errorbar(0.03,y,yerr=[[lo],[hi]],fmt=M["probe"],color=C["probe"],
                                  ms=9,mec="white",mew=0.8,capsize=2,elinewidth=1,ecolor=C["probe"],zorder=6)
axA.set_title("W8A8 family  (8-bit backbone)",fontsize=12,fontweight="bold")
axA.set_xlabel("precision budget  (share of quantisable Linear FLOPs at bf16;  0 = all-W8A8)")

# Panel B: W4W8
frontier(axB,["W4W8_floor","W4W8_knee","W4W8_span1","W4W8_span2","W4W8_span3"],
         C["physics"],M["physics"],"-","physics-guided")
frontier(axB,["W4W8_floor","W4W8_rmse_span1","W4W8_rmse_span2","W4W8_rmse_span3"],
         C["rmse"],M["rmse"],"--","rmse-guided")
axB.set_xscale("log")
axB.set_title("W4W8 family  (4-bit backbone)",fontsize=12,fontweight="bold")
axB.set_xlabel("precision budget  (model weight bytes, log;  6.3e8 = all-W4)")

CEIL=pt("ceiling")
for ax in (axA,axB):
    ax.set_yscale("log"); ax.set_ylim(1e-3,120)
    ax.axhspan(1e-3,BAL_FLOOR,color="#bbbbbb",alpha=0.35,zorder=0)     # round-off floor band
    ax.axhline(CEIL,color="#888",ls=":",lw=1.3,zorder=1)               # bf16 ceiling (separate!)
    ax.axhline(1.0,color="#444",ls="-",lw=0.6,alpha=0.4,zorder=1)      # SVR=1 acceptability line
    ax.set_ylabel("measured balance distortion @120h  (SVR, log; lower = better)")
    ax.grid(True,which="both",ls="-",lw=0.4,color="#ececec",zorder=0); ax.set_axisbelow(True)
    for s in ("top","right"): ax.spines[s].set_visible(False)
axA.text(0.02,BAL_FLOOR*1.25,"round-off noise floor",fontsize=7.5,color="#666")
axA.text(0.02,CEIL*1.15,"bf16 ceiling",fontsize=7.5,color="#666")
axA.text(0.02,1.12,"SVR=1",fontsize=7.5,color="#444")

handles=[Line2D([],[],color=C["physics"],marker="o",lw=2,label="physics-guided",mec="white"),
         Line2D([],[],color=C["rmse"],marker="s",ls="--",lw=2,label="rmse-guided",mec="white"),
         Line2D([],[],color=C["random"],marker="^",ls="",label="random (control)",mec="white"),
         Line2D([],[],color=C["probe"],marker="X",ls="",label="probe (control)")]
axA.legend(handles=handles,frameon=False,fontsize=9,loc="upper right")
fig.suptitle("Measured physics-distortion frontier",fontsize=12.5,fontweight="bold",y=0.99)
fig.tight_layout(rect=[0,0,1,0.95])
os.makedirs("plots_harness_wpd",exist_ok=True)
out="plots_harness_wpd/frontier_ci.png"; fig.savefig(out,dpi=160,bbox_inches="tight")
print("wrote",out)
