import os, sys, csv
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.getcwd())
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LogNorm

pc=torch.load("harness_results/harness_results.pt", weights_only=False)
fp=pc["FP32"]; inits=sorted(fp.keys()); LEAD=120
HEAD={"Z500":"geopotential_500","T850":"temperature_850","T2M":"2m_temperature",
      "MSLP":"mean_sea_level_pressure","Q700":"specific_humidity_700",
      "U500":"u_component_of_wind_500","V500":"v_component_of_wind_500",
      "U10":"10m_u_component_of_wind","V10":"10m_v_component_of_wind"}
VK={k:"w_rmse_%s_%d"%(v,LEAD) for k,v in HEAD.items()}
def degr(tag,vk):
    return 100*float(np.mean([(float(pc[tag][i][LEAD]["RMSE"][vk])-float(fp[i][LEAD]["RMSE"][vk]))
                              /float(fp[i][LEAD]["RMSE"][vk]) for i in inits]))
flo={r["tag"]:r for r in csv.DictReader(open("harness_results/harness_results_floored.csv"))}
CONF=["W8A8_floor","W8A8_knee","W8A8_rmse_knee","W8A8_probe_divergent","W8A8_probe_rmse_favoured",
      "W8A8_rand_0","W8A8_rand_2","W8A8_rand_3","W8A8_rmse_span1","W8A8_rand_1","W8A8_span1",
      "W8A8_span2","W8A8_rmse_span2","W8A8_span3","W8A8_rmse_span3","ceiling"]
CONF=sorted(CONF,key=lambda t:-float(flo[t]["balance"]))
labels=list(HEAD.keys()); nV=len(labels); nC=len(CONF)
Mtx=np.array([[degr(t,VK[v]) for t in CONF] for v in labels])
bal=np.array([float(flo[t]["balance"]) for t in CONF])
rng=bal.max()/bal.min()

rho={}
if os.path.exists("harness_results/rho_ci.csv"):
    for r in csv.DictReader(open("harness_results/rho_ci.csv")):
        rho[r["var"]]=(float(r["rho_med"]),float(r["ci_lo"]),float(r["ci_hi"]))

fig=plt.figure(figsize=(14.5,6.8))
gs=fig.add_gridspec(2,2,width_ratios=[nC,3.2],height_ratios=[nV,1.5],hspace=0.06,wspace=0.03)
ax=fig.add_subplot(gs[0,0]); axf=fig.add_subplot(gs[0,1]); axb=fig.add_subplot(gs[1,0])

vmax=float(np.nanpercentile(np.abs(Mtx),98))
im=ax.imshow(Mtx,aspect="auto",cmap="RdBu_r",norm=TwoSlopeNorm(0,-vmax,vmax))
ax.set_yticks(range(nV)); ax.set_yticklabels(labels,fontsize=10)
ax.set_xticks([]); ax.set_xlim(-0.5,nC-0.5)
ax.set_title("Per-variable RMSE degradation vs fp32 (%)  -  W8A8 family, lead 120 h",
             fontsize=12,fontweight="bold",pad=8)
for i in range(nV):
    for j in range(nC):
        ax.text(j,i,f"{Mtx[i,j]:.1f}",ha="center",va="center",fontsize=6.4,
                color="#222" if abs(Mtx[i,j])<vmax*0.6 else "white")
cb=fig.colorbar(im,ax=ax,location="left",pad=0.055,fraction=0.03); cb.set_label("RMSE degr %",fontsize=8)

axf.axvline(0,color="#888",lw=1,zorder=1)
for i,v in enumerate(labels):
    if v in rho:
        m,lo,hi=rho[v]; sig=(lo>0)or(hi<0); c="#c0392b" if sig else "#999"
        axf.errorbar(m,i,xerr=[[m-lo],[hi-m]],fmt="o",ms=6,color=c,ecolor=c,capsize=3,
                     elinewidth=1.4,zorder=5)
    else:
        axf.text(0,i,"(pending)",fontsize=7,color="#bbb",va="center",ha="center")
axf.set_ylim(nV-0.5,-0.5); axf.set_yticks([]); axf.set_xlim(-1.05,1.05)
axf.set_xlabel(r"Spearman $\rho$(RMSE$_v$, balance),  95% CI",fontsize=9)
axf.set_title("does this variable's\nRMSE track balance?",fontsize=9.5)
for s in ("top","right","left"): axf.spines[s].set_visible(False)
axf.grid(True,axis="x",ls="-",lw=0.4,color="#eee")

imb=axb.imshow(bal[None,:],aspect="auto",cmap="magma_r",norm=LogNorm(vmin=0.02,vmax=100))
axb.set_yticks([0]); axb.set_yticklabels(["balance (SVR)"],fontsize=10,fontweight="bold")
axb.set_xticks(range(nC)); axb.set_xlim(-0.5,nC-0.5)
axb.set_xticklabels([t.replace("W8A8_","").replace("ceiling","bf16") for t in CONF],
                    rotation=40,ha="right",fontsize=8)
for j in range(nC):
    axb.text(j,0,f"{bal[j]:.2f}",ha="center",va="center",fontsize=6.6,
             color="white" if bal[j]>0.3 else "#222")
cbb=fig.colorbar(imb,ax=axb,location="left",pad=0.055,fraction=0.11); cbb.set_label("SVR",fontsize=8)

fig.suptitle(f"RMSE stays within a few % across EVERY variable while balance swings {rng:.0f}× "
             f"-> no standard-variable RMSE flags the physics damage",
             fontsize=12,fontweight="bold",y=1.0)
out="plots_harness_wpd/rmse_blindness_pervar.png"; fig.savefig(out,dpi=160,bbox_inches="tight")
print(f"balance range {bal.min():.3f}..{bal.max():.2f} ({rng:.0f}x); rho loaded: {bool(rho)}")
print("wrote",out)
