"""The uniform-floor vs physics-protected RMSE gap at each lead."""
import physq_path

import argparse
import csv
import os

import torch

import rmse_compression as rc

FIELDS = ["tag_a", "tag_b", "lead", "block", "diff_pp", "ci_lo", "ci_hi", "excludes_zero",
          "deg_a_pct", "deg_b_pct"]

PAIR = ("W8A8_floor", "W8A8_span1")
LEADS = (24, 72, 120, 168)
BLOCKS = (1, 2, 3)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pt", default=os.path.join("harness_results_n48", "harness_results.pt"))
    ap.add_argument("--outdir", default="harness_results_n48")
    ap.add_argument("--tag-a", default=PAIR[0], dest="tag_a")
    ap.add_argument("--tag-b", default=PAIR[1], dest="tag_b")
    ap.add_argument("--n-boot", type=int, default=4000, dest="n_boot")
    a = ap.parse_args(argv)

    pt = torch.load(a.pt, map_location="cpu", mmap=True, weights_only=False)
    for t in (a.tag_a, a.tag_b, "FP32"):
        if t not in pt:
            raise SystemExit(f"{t} not in {a.pt}")
    pt = {tag: (v["metrics"] if isinstance(v, dict) and "metrics" in v else v)
          for tag, v in pt.items()}
    os.makedirs(a.outdir, exist_ok=True)

    rows = []
    for lead in LEADS:
        deg_a = rc.rmse_degradation(pt, a.tag_a, lead)
        deg_b = rc.rmse_degradation(pt, a.tag_b, lead)
        for block in BLOCKS:
            r = rc.rmse_difference_ci(pt, a.tag_a, a.tag_b, lead,
                                      block=block, n_boot=a.n_boot)
            rows.append({"tag_a": a.tag_a, "tag_b": a.tag_b, "lead": lead, "block": block,
                         "diff_pp": f"{r['diff']:.4f}",
                         "ci_lo": f"{r['ci_lo']:.4f}", "ci_hi": f"{r['ci_hi']:.4f}",
                         "excludes_zero": r["excludes_zero"],
                         "deg_a_pct": f"{deg_a:.4f}", "deg_b_pct": f"{deg_b:.4f}"})

    out = os.path.join(a.outdir, "harness_rmse_crossing.csv")
    with open(out, "w", newline="") as f:
        f.write(f"# {a.tag_a} minus {a.tag_b} RMSE degradation, percentage points, "
                f"paired on initialisation date. DESCRIPTIVE: point estimates only, no "
                f"interval and no test; only the 168h comparison was pre-registered.\n")
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")

    print(f"\n=== {a.tag_a} - {a.tag_b} RMSE gap (block=2) ===")
    print(f"{'lead':>6}{'diff pp':>11}{'95% CI':>22}  excl 0   sign robust across blocks")
    for lead in LEADS:
        at = [r for r in rows if r["lead"] == lead]
        b2 = next(r for r in at if r["block"] == 2)
        signs = {float(r["diff_pp"]) > 0 for r in at}
        excl = {r["excludes_zero"] for r in at}
        print(f"{lead:>6}{float(b2['diff_pp']):>11.2f}"
              f"  [{float(b2['ci_lo']):>7.2f},{float(b2['ci_hi']):>8.2f}]"
              f"   {'YES' if b2['excludes_zero'] else '--':<6}"
              f"   {'yes' if len(signs) == 1 else 'NO -- SIGN FLIPS'}"
              f"{'' if excl == {True} else '  (not all blocks exclude 0)'}")
    return rows


if __name__ == "__main__":
    main()
