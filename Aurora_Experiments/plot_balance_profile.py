import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.getcwd())
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

pc=torch.load("harness_results/harness_results.pt", weights_only=False)
fp=pc["FP32"]; inits=sorted(fp.keys()); LEAD=120; FAM="wind_balance"
levs=[50,100,150,200,250,300,400,500,600,700,850,925,1000]
def prof(tag):
    m=[]; e=[]
    for L in levs:
        s=np.array([float(pc[tag][i][LEAD][FAM][f"wbal_ageo_geo_pred_{L}_{LEAD}"]) for i in inits])
        m.append(s.mean()); e.append(s.std()/np.sqrt(len(s)))
    return np.array(m),np.array(e)
series={"fp32 (reference)":("#000000","-","o","FP32"),
        "uniform W8A8 (all-int8)":("#D55E00","--","s","W8A8_floor"),
        "physics-protected (span1)":("#0072B2","-","o","W8A8_span1")}
fig,ax=plt.subplots(figsize=(6.6,7.2))
for lab,(c,ls,mk,tag) in series.items():
    m,e=prof(tag)
    ax.plot(m,levs,ls,color=c,marker=mk,ms=5,lw=2,mec="white",mew=0.8,label=lab,zorder=5)
    ax.fill_betweenx(levs,m-e,m+e,color=c,alpha=0.15,zorder=2)
ax.set_yscale("log"); ax.invert_yaxis()
ax.set_yticks(levs); ax.set_yticklabels(levs,fontsize=8)
ax.set_ylabel("pressure level (hPa)"); ax.set_xlabel("ageostrophic / geostrophic wind ratio  @120h")
ax.set_title("Physics-guided protection (span1) restores the geostrophic\n"
             "balance that uniform W8A8 destroys (median +34%)",
             fontsize=11.5,fontweight="bold")
ax.legend(frameon=False,fontsize=9.5,loc="lower right")
ax.grid(True,which="both",ls="-",lw=0.4,color="#ececec"); ax.set_axisbelow(True)
for s in ("top","right"): ax.spines[s].set_visible(False)
# annotate the 500hPa gap
m0,_=prof("FP32"); m1,_=prof("W8A8_floor"); i5=levs.index(500)
ax.annotate(f"+{100*(m1[i5]-m0[i5])/m0[i5]:.0f}%",xy=(m1[i5],500),xytext=(m1[i5]+0.03,430),
            fontsize=9,color="#D55E00",arrowprops=dict(arrowstyle="->",color="#D55E00",lw=1))
out="plots_harness_wpd/balance_profile.png"; fig.tight_layout(); fig.savefig(out,dpi=160,bbox_inches="tight")
print("wrote",out)
