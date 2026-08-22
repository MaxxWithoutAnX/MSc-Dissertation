"""The uniform-floor vs physics-protected RMSE gap at each lead."""
import physq_path

import argparse
import csv
import os

import torch

import rmse_compression as rc

FIELDS = ["tag_a", "tag_b", "lead", "diff_pp", "deg_a_pct", "deg_b_pct"]

PAIR = ("W8A8_floor", "W8A8_span1")
LEADS = (24, 72, 120, 168)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pt", default=os.path.join("harness_results_n48", "harness_results.pt"))
    ap.add_argument("--outdir", default="harness_results_n48")
    ap.add_argument("--tag-a", default=PAIR[0], dest="tag_a")
    ap.add_argument("--tag-b", default=PAIR[1], dest="tag_b")
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
        r = rc.rmse_difference(pt, a.tag_a, a.tag_b, lead)
        rows.append({"tag_a": a.tag_a, "tag_b": a.tag_b, "lead": lead,
                     "diff_pp": f"{r['diff']:.4f}",
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

    print(f"\n=== {a.tag_a} - {a.tag_b} RMSE gap ===")
    print(f"{'lead':>6}{'diff pp':>11}{'deg a %':>11}{'deg b %':>11}")
    for r in rows:
        print(f"{r['lead']:>6}{float(r['diff_pp']):>11.2f}"
              f"{float(r['deg_a_pct']):>11.2f}{float(r['deg_b_pct']):>11.2f}")
    return rows


if __name__ == "__main__":
    main()
