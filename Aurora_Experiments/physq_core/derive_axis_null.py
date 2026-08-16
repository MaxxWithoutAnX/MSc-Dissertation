"""Derive the axis-level null from the round-off ensemble (CPU only, no inference)."""
import argparse
import collections
import contextlib
import csv as _csv
import os

import numpy as np
import torch

import ablation_comp as ac
import paired_stats as ps
from plot_common import AGG_CLASS, MetricsRun

@contextlib.contextmanager
def _csv_writer(path):
    """Small helper so the summary is written the same way everywhere."""
    with open(path, "w", newline="") as fh:
        yield _csv.writer(fh)


LEAD = 120
DIR = "noise_floor_ens"
AXES = ["balance", "conservation", "standard"]

AURORA_G1 = 0.03066     # film balance @W8 -- the floor must erase this
AURORA_G2 = 0.09295     # decoder_heads standard @W8 -- the floor must NOT erase this
DEFAULT_FLOOR = 0.05    # the practical-significance floor actually in use


def derive(ens_dir, lead=LEAD, verbose=True):
    inv = collections.defaultdict(list)
    for fam, cls in AGG_CLASS.items():
        inv[cls].append(fam)

    paths = sorted(p for p in os.listdir(ens_dir) if p.endswith(".pt"))
    if len(paths) < 2:
        raise SystemExit(f"{ens_dir}: need >=2 members, found {len(paths)}")
    if verbose:
        print(f"{len(paths)} members; reference = {paths[0]}", flush=True)

    ref = MetricsRun("ref", torch.load(os.path.join(ens_dir, paths[0]),
                                       map_location="cpu", weights_only=False)["metrics"])

    per_axis = {a: [] for a in AXES}
    for pth in paths[1:]:
        mem = MetricsRun(pth, torch.load(os.path.join(ens_dir, pth),
                                         map_location="cpu", weights_only=False)["metrics"])
        for a in AXES:
            fam_vals = {}
            for fam in inv.get(a, []):
                specs = ps.specs_for_family(ref, mem, fam, lead)
                if not specs:
                    continue
                deltas, denoms = ps.family_deltas(ref, mem, specs, lead)
                if not deltas:
                    continue
                idx = np.arange(len(next(iter(deltas.values()))))
                v = ps.axis_value(deltas, denoms, idx)
                if np.isfinite(v):
                    fam_vals[fam] = v
            if fam_vals:
                per_axis[a].append(ac._reduce_axis(fam_vals, a, None))
        if verbose:
            print(f"  {pth}: " + "  ".join(
                f"{a}={per_axis[a][-1]:.5g}" if per_axis[a] else f"{a}=--" for a in AXES),
                flush=True)
    return per_axis, len(paths) - 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ens", default=DIR, help=f"round-off ensemble dir (default {DIR})")
    ap.add_argument("--lead", type=int, default=LEAD)
    ap.add_argument("--gates", default="aurora", choices=("aurora", "none"),
                    help="print the comparison against Aurora's W8 gate bounds. Use 'none' "
                         "for any other model -- those constants are Aurora measurements.")
    ap.add_argument("--out", default=None,
                    help="write the per-axis null summary here as CSV "
                         "(default: <ens>/axis_null_p95.csv)")
    a = ap.parse_args(argv)

    per_axis, n = derive(a.ens, lead=a.lead)

    print(f"\n=== AXIS-LEVEL NULL ({a.ens}, round-off ensemble, "
          f"n={n} perturbed members, {a.lead}h) ===")
    print(f"{'axis':14}{'median':>11}{'p90':>11}{'p95':>11}{'max':>11}")
    for ax in AXES:
        v = np.asarray([x for x in per_axis[ax] if np.isfinite(x)])
        if not v.size:
            print(f"{ax:14}{'no data':>11}")
            continue
        print(f"{ax:14}{np.median(v):11.5g}{np.percentile(v, 90):11.5g}"
              f"{np.percentile(v, 95):11.5g}{v.max():11.5g}")

    if a.gates == "aurora":
        print("\n=== GATE COMPARISON (Aurora W8 reference points) ===")
        print(f"G1 lower bound (film balance @W8, must be erased) = {AURORA_G1}")
        print(f"DEFAULT_FLOOR (practical-significance, in use)     = {DEFAULT_FLOOR}")
        print(f"G2 upper bound (decoder_heads standard @W8)        = {AURORA_G2}")
        for ax in AXES:
            v = np.asarray([x for x in per_axis[ax] if np.isfinite(x)])
            if not v.size:
                continue
            p95 = np.percentile(v, 95)
            verdict = ("WOULD PASS G1 (>%s)" % AURORA_G1 if ax == "balance" and p95 > AURORA_G1
                       else "BELOW G1 bound" if ax == "balance" else "")
            print(f"  {ax:14} p95 = {p95:.5g}   {verdict}")

    out = a.out or os.path.join(a.ens, "axis_null_p95.csv")
    with _csv_writer(out) as w:
        w.writerow(["axis", "lead", "n_members", "median", "p90", "p95", "max"])
        for ax in AXES:
            v = np.asarray([x for x in per_axis[ax] if np.isfinite(x)])
            if not v.size:
                continue
            w.writerow([ax, a.lead, n, f"{np.median(v):.10g}",
                        f"{np.percentile(v, 90):.10g}", f"{np.percentile(v, 95):.10g}",
                        f"{v.max():.10g}"])
    print(f"\nwrote {out}")
    return per_axis


if __name__ == "__main__":
    main()
