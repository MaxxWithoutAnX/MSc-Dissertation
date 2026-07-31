import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.getcwd())
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

pc=torch.load("harness_results/harness_results.pt", weights_only=False)
fp=pc["FP32"]; inits=sorted(fp.keys()); L=120; FAM="power spectrum"
VARS={"Z500":"geopotential_500","U500":"u_component_of_wind_500"}
def psd(tag,var):   # mean PSD over inits, and wavenumbers
    P=[]; 
    for i in inits:
        d=pc[tag][i][L][FAM]
        P.append(np.asarray(d[f"psd_preds_{var}_{L}"],dtype=float))
    k=np.asarray(fp[inits[0]][L][FAM][f"wavenumbers_{var}_{L}"],dtype=float)
    return k, np.mean(P,axis=0)
cfgs={"uniform W8A8 (floor)":("#D55E00","W8A8_floor"),
      "physics-protected (span1)":("#0072B2","W8A8_span1")}

print("k^2-weighted (gradient/balance) vs plain (RMSE) change in field power, floor vs fp32:")
fig,axes=plt.subplots(1,2,figsize=(12.6,5.4))
for ax,(vlab,var) in zip(axes,VARS.items()):
    k,pf=psd("FP32",var); m=k>0
    for clab,(c,tag) in cfgs.items():
        _,pc_=psd(tag,var)
        ax.plot(k[m],(pc_/pf)[m],"-",color=c,lw=2,label=clab)
    ax.axhline(1,color="#888",ls=":",lw=1)
    ax.set_xscale("log"); ax.set_xlabel("wavenumber k  (large scale -> small scale)")
    ax.set_ylabel(f"PSD(config) / PSD(fp32)   [{vlab}]")
    ax.set_title(vlab,fontsize=11,fontweight="bold")
    ax.grid(True,which="both",ls="-",lw=0.4,color="#ececec"); ax.set_axisbelow(True)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    # quantitative test (floor)
    _,pflr=psd("W8A8_floor",var)
    var_plain=(pflr[m].sum()-pf[m].sum())/pf[m].sum()
    kk=k[m]**2
    var_grad=((kk*pflr[m]).sum()-(kk*pf[m]).sum())/(kk*pf[m]).sum()
    print(f"  {vlab:6s}:  plain d(sum PSD)={100*var_plain:+6.1f}%   k^2-weighted d(sum k^2 PSD)={100*var_grad:+7.1f}%   "
          f"amplification x{var_grad/var_plain if var_plain!=0 else float('nan'):.0f}")
axes[0].legend(frameon=False,fontsize=9,loc="upper left")
fig.suptitle("Quantisation error is concentrated at small scales (high k): "
             "invisible to bulk RMSE, amplified by the gradients that define balance",
             fontsize=11.5,fontweight="bold",y=1.0)
fig.tight_layout(rect=[0,0,1,0.95])
out="plots_harness_wpd/spectral_test.png"; fig.savefig(out,dpi=160,bbox_inches="tight")
print("wrote",out)
