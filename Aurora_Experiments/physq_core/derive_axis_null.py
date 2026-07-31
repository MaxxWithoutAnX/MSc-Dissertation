import collections
import os

import numpy as np
import torch

import ablation_comp as ac
import paired_stats as ps
from plot_common import AGG_CLASS, MetricsRun

LEAD = 120
DIR = "noise_floor_ens"
AXES = ["balance", "conservation", "standard"]

inv = collections.defaultdict(list)
for fam, cls in AGG_CLASS.items():
    inv[cls].append(fam)

paths = sorted(p for p in os.listdir(DIR) if p.endswith(".pt"))
print(f"{len(paths)} members; reference = {paths[0]}", flush=True)

ref = MetricsRun("ref", torch.load(os.path.join(DIR, paths[0]),
                                   map_location="cpu", weights_only=False)["metrics"])

per_axis = {a: [] for a in AXES}
for pth in paths[1:]:
    mem = MetricsRun(pth, torch.load(os.path.join(DIR, pth),
                                     map_location="cpu", weights_only=False)["metrics"])
    for a in AXES:
        fam_vals = {}
        for fam in inv.get(a, []):
            specs = ps.specs_for_family(ref, mem, fam, LEAD)
            if not specs:
                continue
            deltas, denoms = ps.family_deltas(ref, mem, specs, LEAD)
            if not deltas:
                continue
            idx = np.arange(len(next(iter(deltas.values()))))
            v = ps.axis_value(deltas, denoms, idx)
            if np.isfinite(v):
                fam_vals[fam] = v
        if fam_vals:
            per_axis[a].append(ac._reduce_axis(fam_vals, a, None))
    print(f"  {pth}: " + "  ".join(
        f"{a}={per_axis[a][-1]:.5g}" if per_axis[a] else f"{a}=--" for a in AXES), flush=True)

print("\n=== AXIS-LEVEL NULL (round-off ensemble, n=%d perturbed members) ===" % (len(paths) - 1))
print(f"{'axis':14}{'median':>11}{'p90':>11}{'p95':>11}{'max':>11}")
for a in AXES:
    v = np.asarray([x for x in per_axis[a] if np.isfinite(x)])
    if not v.size:
        print(f"{a:14}{'no data':>11}")
        continue
    print(f"{a:14}{np.median(v):11.5g}{np.percentile(v,90):11.5g}"
          f"{np.percentile(v,95):11.5g}{v.max():11.5g}")

print("\n=== GATE COMPARISON ===")
print("G1 lower bound (film balance @W8, must be erased) = 0.03066")
print("DEFAULT_FLOOR (practical-significance, in use)     = 0.05")
print("G2 upper bound (decoder_heads standard @W8)        = 0.09295")
for a in AXES:
    v = np.asarray([x for x in per_axis[a] if np.isfinite(x)])
    if not v.size:
        continue
    p95 = np.percentile(v, 95)
    verdict = ("WOULD PASS G1 (>0.03066)" if a == "balance" and p95 > 0.03066
               else "BELOW G1 bound" if a == "balance" else "")
    print(f"  {a:14} p95 = {p95:.5g}   {verdict}")
