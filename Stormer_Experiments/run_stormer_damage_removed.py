"""The damage-removed table for Stormer."""
import physq_path

import argparse
import os

import torch

import run_paired_stats as rps
from run_stormer_stats import AXIS_FAMILIES, unwrap_store

FLOOR_TAG = "W8A8_floor"
REMOVED_TAGS = ["W8A8_knee", "W8A8_span1", "W8A8_rmse_span1",
                "W8A8_probe_divergent", "W8A8_probe_rmse_favoured"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pt", default=os.path.join("harness_results_n47",
                                                 "harness_results.pt"))
    ap.add_argument("--outdir", default="harness_results_n47")
    ap.add_argument("--lead", type=int, default=120)
    ap.add_argument("--n-boot", type=int, default=4000, dest="n_boot")
    a = ap.parse_args(argv)

    pt = unwrap_store(torch.load(a.pt, map_location="cpu", weights_only=False))
    missing = [t for t in REMOVED_TAGS + [FLOOR_TAG] if t not in pt]
    if missing:
        raise SystemExit(f"tags absent from {a.pt}: {missing}")
    os.makedirs(a.outdir, exist_ok=True)

    for axis, family in AXIS_FAMILIES.items():
        rows = rps.damage_removed(pt, lead=a.lead, family=family, tags=REMOVED_TAGS,
                                  floor_tag=FLOOR_TAG, blocks=(1, 2, 3), n_boot=a.n_boot)
        if not rows:
            raise SystemExit(f"no rows for {family} -- the floor tag did not resolve")
        sfx = "" if family == "wind_balance" else f"_{family}"
        out = os.path.join(a.outdir, f"harness_damage_removed{sfx}.csv")
        rps._write(out, rps.REMOVED_FIELDS, rows)

        print(f"\n=== {axis} ({family}) removed vs {FLOOR_TAG} @ {a.lead}h, block=2 ===")
        print(f"{'config':30}{'removed %':>11}{'95% CI':>22}  sig")
        for r in rows:
            if r["block"] == 2:
                print(f"  {r['tag']:28}{r['removed_pct']:11.1f}  "
                      f"[{r['ci_lo']:8.1f},{r['ci_hi']:9.1f}]  "
                      f"{'YES' if r['significant'] else '--'}")


if __name__ == "__main__":
    main()
