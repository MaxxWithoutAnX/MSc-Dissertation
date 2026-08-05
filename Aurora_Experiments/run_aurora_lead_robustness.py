"""Does Aurora's physics-vs-RMSE composite advantage hold at every lead?"""
import physq_path

import argparse
import csv
import os

import torch

import lead_robustness as lr
import run_composite_stats as rcs

PAIR = ("W8A8_span1", "W8A8_rmse_span1")
LEADS = (24, 72, 120, 168)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pt", default=os.path.join("harness_results_merged",
                                                 "harness_results.pt"))
    ap.add_argument("--outdir", default="harness_results_merged")
    a = ap.parse_args(argv)

    pt = torch.load(a.pt, map_location="cpu", weights_only=False)
    missing = [t for t in PAIR if t not in pt]
    if missing:
        raise SystemExit(f"pair names tags absent from {a.pt}: {missing}")

    rows, trend = lr.run(pt, PAIR, LEADS, rcs.AXIS_FAMILIES)
    if len(rows) != len(LEADS):
        raise SystemExit(f"{len(rows)} tests but {len(LEADS)} leads were requested")

    os.makedirs(a.outdir, exist_ok=True)
    out = os.path.join(a.outdir, "aurora_lead_robustness.csv")
    with open(out, "w", newline="") as f:
        f.write(f"# pair={'|'.join(PAIR)} | "
                f"POST-HOC trend: log-log slope {trend['slope']:+.3f}\n")
        w = csv.DictWriter(f, fieldnames=rcs.FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}\n")

    print(f"pair {PAIR[0]} vs {PAIR[1]}\n")
    print(f"{'lead':>6}{'phys':>9}{'rmse':>9}{'ratio':>8}  binding")
    for r in rows:
        print(f"{r['lead']:>5}h{r['physics']:9.4f}{r['rmse']:9.4f}{r['ratio']:8.2f}"
              f"  {r['physics_argmax']}/{r['rmse_argmax']}")
    print(f"\nphysics leads {sum(r['ratio'] > 1 for r in rows)}/{len(rows)} leads")
    print(f"\n--- POST HOC trend --- log(ratio) ~ log(lead) slope {trend['slope']:+.3f}")


if __name__ == "__main__":
    main()
