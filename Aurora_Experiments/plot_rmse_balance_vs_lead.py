import os, sys, csv
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.getcwd())
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

pc=torch.load("harness_results/harness_results.pt", weights_only=False)
fp=pc["FP32"]; inits=sorted(fp.keys()); LEADS=[24,72,120,168]
HEAD={"Z500":"geopotential_500","T850":"temperature_850","T2M":"2m_temperature",
      "MSLP":"mean_sea_level_pressure","Q700":"specific_humidity_700",
      "U500":"u_component_of_wind_500","V500":"v_component_of_wind_500"}
def rmse_degr(tag,L):
    VK=["w_rmse_%s_%d"%(v,L) for v in HEAD.values()]
    ds=[]
    for vk in VK:
        f=np.array([float(fp[i][L]["RMSE"][vk]) for i in inits])
        c=np.array([float(pc[tag][i][L]["RMSE"][vk]) for i in inits])
        ds.append(np.mean((c-f)/f))
    return 100*np.mean(ds)
flo={(r["tag"],int(r["lead"])):float(r["balance"])
     for r in csv.DictReader(open("harness_results/harness_results_by_lead_floored.csv"))}

series={"uniform W8A8 (floor)":("#D55E00","s","W8A8_floor"),
        "physics-protected (span1)":("#0072B2","o","W8A8_span1"),
        "bf16 ceiling":("#333333","^","ceiling")}
fig,(axR,axB)=plt.subplots(1,2,figsize=(12.6,5.4))

for lab,(c,mk,tag) in series.items():
    axR.plot(LEADS,[rmse_degr(tag,L) for L in LEADS],"-",color=c,marker=mk,ms=7,lw=2,
             mec="white",mew=1,label=lab,zorder=5)
    axB.plot(LEADS,[flo[(tag,L)] for L in LEADS],"-",color=c,marker=mk,ms=7,lw=2,
             mec="white",mew=1,label=lab,zorder=5)

axR.axhline(0,color="#888",lw=1,ls=":")
axR.set_title("RMSE degradation vs fp32  (headline variables)",fontsize=11.5,fontweight="bold")
axR.set_ylabel("RMSE degradation (%)")

axB.set_yscale("log")
axB.set_title("balance distortion  (SVR)",fontsize=11.5,fontweight="bold")
axB.set_ylabel("balance distortion (SVR, log)")
axB.axhline(1.0,color="#444",lw=0.6,alpha=0.5)
axB.text(24,1.12,"SVR=1",fontsize=8,color="#444")

for ax in (axR,axB):
    ax.set_xlabel("lead time (h)"); ax.set_xticks(LEADS)
    ax.grid(True,which="both",ls="-",lw=0.4,color="#ececec"); ax.set_axisbelow(True)
    for s in ("top","right"): ax.spines[s].set_visible(False)
axR.legend(frameon=False,fontsize=9,loc="upper right")
fig.suptitle("Same three models, two axes: uniform-W8A8 looks fine-to-better on RMSE as lead grows, "
             "yet its physics stays broken at every lead",fontsize=11.5,fontweight="bold",y=1.0)
fig.tight_layout(rect=[0,0,1,0.95])
out="plots_harness_wpd/rmse_balance_vs_lead.png"; fig.savefig(out,dpi=160,bbox_inches="tight")
print("wrote",out)
for L in LEADS:
    print(f"  @{L}h  floor: RMSE {rmse_degr('W8A8_floor',L):+.2f}% bal {flo[('W8A8_floor',L)]:.2f}"
          f"   span1: RMSE {rmse_degr('W8A8_span1',L):+.2f}% bal {flo[('W8A8_span1',L)]:.2f}")
