"""Aggregate noise_floor_ens/member_*.pt into validation/noise_floor_detailed.pt."""
import physq_path

import argparse
import glob
import os
from collections import defaultdict

import numpy as np
import torch
import xarray as xr

import ablation_comp as ac
from distortion import NoiseFloor, floor_family_of
from plot_common import MetricsRun
from run_precision_harness import compute_scored_dates


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ens", default="noise_floor_ens")
    ap.add_argument("--data", default="data/era5_sampled_2021_4pm_aurora_0p25.nc")
    ap.add_argument("--out", default=os.path.join("validation", "noise_floor_detailed.pt"))
    ap.add_argument("--quantile", type=float, default=0.9)
    ap.add_argument("--leads", default="24,72,120,168")
    a = ap.parse_args()
    leads = [int(x) for x in a.leads.split(",")]

    members = {}
    for p in sorted(glob.glob(os.path.join(a.ens, "member_*.pt"))):
        d = torch.load(p, map_location="cpu", weights_only=False)
        members[d["seed"]] = d["metrics"]
    if 0 not in members:
        raise SystemExit("need member_000.pt (the unperturbed reference, seed 0)")
    ref = MetricsRun("ref", members[0])
    pert = {f"ens{s}": MetricsRun(f"ens{s}", m) for s, m in members.items() if s != 0}
    if not pert:
        raise SystemExit("no perturbed members (seeds >= 1) found in " + a.ens)
    print(f"reference + {len(pert)} perturbed members; inits={len(ref.dates)}", flush=True)

    ds = xr.open_dataset(a.data)
    dates = compute_scored_dates(ds, ref.dates)
    block = ac.bootstrap_block(len(ref.dates))
    registry = ac.metric_registry(ref)

    pooled = defaultdict(list)                       # (family, lead) -> [|mean_delta|, ...]
    for lead in leads:
        if lead not in ref.leads:
            print(f"  skip lead {lead} (not in members)", flush=True)
            continue
        specs = registry
        for run in pert.values():                    # keep only specs present everywhere
            specs = ac.available_specs(ref, run, specs, lead)
        spec_by_label = {s.label: s for s in specs}
        recs = ac.sensitivity_records(ref, pert, specs, list(pert), [lead], dates, block,
                                      full_table=None, norm_mode="colmax",
                                      noise_floor=NoiseFloor.null())
        for r in recs:
            fam = floor_family_of(spec_by_label[r["metric"]])
            if fam is not None:
                pooled[(fam, lead)].append(abs(r["mean_delta"]))
        print(f"  lead {lead}: {len(specs)} specs x {len(pert)} members", flush=True)

    per_family = defaultdict(dict)
    for (fam, lead), vals in pooled.items():
        per_family[fam][lead] = float(np.quantile(vals, a.quantile))
    detailed = {"ensemble": {"per_family": {k: dict(v) for k, v in per_family.items()},
                             "n_members": len(pert), "quantile": a.quantile}}

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save(detailed, a.out)
    print(f"\nwrote {a.out} (q={a.quantile}, {len(pert)} members):")
    for fam, ld in sorted(detailed["ensemble"]["per_family"].items()):
        print(f"  {fam:16} " + "  ".join(f"{lt}h={v:.4g}" for lt, v in sorted(ld.items())))


if __name__ == "__main__":
    main()
