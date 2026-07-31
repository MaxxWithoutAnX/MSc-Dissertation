import os, sys, csv
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.getcwd())
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

pc=torch.load("harness_results/harness_results.pt", weights_only=False)
fp=pc["FP32"]; inits=sorted(fp.keys()); LEAD=120
VARS=list(fp[inits[0]][LEAD]["RMSE"].keys())
def rmse_degr_pct(tag):
    """per-var (config-fp32)/fp32, averaged over vars & inits -> dimensionless %."""
    per=[]
    for i in inits:
        c=pc[tag][i][LEAD]["RMSE"]; f=fp[i][LEAD]["RMSE"]
        per.append(np.mean([(float(c[v])-float(f[v]))/float(f[v]) for v in VARS if float(f[v])>0]))
    return 100*float(np.mean(per))
flo={r["tag"]:r for r in csv.DictReader(open("harness_results/harness_results_floored.csv"))}

groups={  # tag -> (family, guide)
 "ceiling":("anchor","ceiling"),"W8A8_floor":("W8A8","floor"),
 "W8A8_knee":("W8A8","physics"),"W8A8_span1":("W8A8","physics"),"W8A8_span2":("W8A8","physics"),"W8A8_span3":("W8A8","physics"),
 "W8A8_rmse_knee":("W8A8","rmse"),"W8A8_rmse_span1":("W8A8","rmse"),"W8A8_rmse_span2":("W8A8","rmse"),"W8A8_rmse_span3":("W8A8","rmse"),
 "W8A8_rand_0":("W8A8","random"),"W8A8_rand_1":("W8A8","random"),"W8A8_rand_2":("W8A8","random"),"W8A8_rand_3":("W8A8","random"),
 "W8A8_probe_divergent":("W8A8","probe"),"W8A8_probe_rmse_favoured":("W8A8","probe"),
 "W4W8_floor":("W4W8","floor"),"W4W8_knee":("W4W8","physics"),"W4W8_span1":("W4W8","physics"),
 "W4W8_span2":("W4W8","physics"),"W4W8_span3":("W4W8","physics"),
 "W4W8_rmse_span1":("W4W8","rmse"),"W4W8_rmse_span2":("W4W8","rmse"),"W4W8_rmse_span3":("W4W8","rmse")}
C={"physics":"#0072B2","rmse":"#D55E00","random":"#009E73","probe":"#CC79A7","floor":"#333333","ceiling":"#888888"}
Mk={"W8A8":"o","W4W8":"D","anchor":"*"}

xs=[]; ys=[]; data=[]
for t,(fam,g) in groups.items():
    x=rmse_degr_pct(t); y=float(flo[t]["balance"]); xs.append(x); ys.append(y)
    data.append((t,fam,g,x,max(y,1e-3)))
# Spearman across all configs
def spearman(a,b):
    ra=np.argsort(np.argsort(a)); rb=np.argsort(np.argsort(b))
    return np.corrcoef(ra,rb)[0,1]
rho=spearman(np.array(xs),np.array(ys))

fig,ax=plt.subplots(figsize=(9.5,6.6))
for t,fam,g,x,y in data:
    ax.scatter(x,y,c=C[g],marker=Mk[fam],s=95,ec="white",lw=1,zorder=5)
# vertical connector showing same-RMSE / different-physics
d={t:(x,y) for t,_,_,x,y in data}
x0,y0=d["W8A8_floor"]; x1,y1=d["W8A8_span1"]
ax.annotate("",xy=(x1,y1),xytext=(x0,y0),arrowprops=dict(arrowstyle="<->",color="#c0392b",lw=1.4,ls=":"))
ax.text((x0+x1)/2-0.3,(y0*y1)**0.5,"16× balance",fontsize=8.5,color="#c0392b",ha="right")

ax.axhline(1.0,color="#444",ls="-",lw=0.6,alpha=0.5)
ax.text(ax.get_xlim()[0],1.1,"SVR=1",fontsize=8,color="#444")
ax.set_yscale("log"); ax.set_ylim(1e-2,120)
ax.set_xlabel("RMSE degradation vs fp32  @120h  (%, mean over variables)")
ax.set_ylabel("measured balance distortion  @120h  (SVR, log)")
ax.set_title(f"RMSE is blind to physics damage  (Spearman ρ = {rho:.2f} across configs)",
             fontsize=13,fontweight="bold")
ax.grid(True,which="both",ls="-",lw=0.4,color="#ececec"); ax.set_axisbelow(True)
for s in ("top","right"): ax.spines[s].set_visible(False)
handles=[Line2D([],[],color=C["physics"],marker="o",ls="",label="physics-guided",mec="white"),
         Line2D([],[],color=C["rmse"],marker="o",ls="",label="rmse-guided",mec="white"),
         Line2D([],[],color=C["random"],marker="o",ls="",label="random",mec="white"),
         Line2D([],[],color=C["probe"],marker="o",ls="",label="probe",mec="white"),
         Line2D([],[],color=C["floor"],marker="o",ls="",label="uniform floor",mec="white"),
         Line2D([],[],color="#555",marker="o",ls="",label="W8A8 family",mec="white"),
         Line2D([],[],color="#555",marker="D",ls="",label="W4W8 family",mec="white")]
ax.legend(handles=handles,frameon=False,fontsize=8.5,loc="lower right",ncol=2)
fig.tight_layout()
out="plots_harness_wpd/rmse_blindness.png"; fig.savefig(out,dpi=160,bbox_inches="tight")
print(f"Spearman rho(RMSE_degr, balance) = {rho:.3f}")
print("RMSE degradation range: %.2f%% .. %.2f%%"%(min(xs),max(xs)))
print("balance SVR range: %.3f .. %.3f (%.0fx)"%(min(ys),max(ys),max(ys)/max(min(ys),1e-6)))
print("wrote",out)
