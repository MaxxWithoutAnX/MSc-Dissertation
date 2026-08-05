"""Does Stormer's physics-vs-RMSE composite advantage hold at every lead?"""
import physq_path

import argparse
import csv
import os

import torch

import lead_robustness as lr
import run_composite_stats as rcs
from run_stormer_stats import AXIS_FAMILIES, PAIRS, unwrap_store

PAIR = ("W8A8_span1", "W8A8_rmse_span1")
LEADS = (24, 72, 120, 168)
FIELDS = rcs.FIELDS


def _write(path, fields, rows, note=None):
    with open(path, "w", newline="") as f:
        if note:
            f.write(f"# {note}\n")
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pt", default=os.path.join("harness_results_2021_stormer",
                                                 "harness_results.pt"))
    ap.add_argument("--outdir", default="harness_results_2021_stormer")
    a = ap.parse_args(argv)

    pt = unwrap_store(torch.load(a.pt, map_location="cpu", weights_only=False))
    missing = [t for t in PAIR if t not in pt]
    if missing:
        raise SystemExit(f"declared pair names tags absent from {a.pt}: {missing}")
    os.makedirs(a.outdir, exist_ok=True)

    # --- 1. the headline pair, per lead -------------------------------------------------
    rows, trend = lr.run(pt, PAIR, LEADS, AXIS_FAMILIES)
    if len(rows) != len(LEADS):
        raise SystemExit(f"{len(rows)} tests but {len(LEADS)} leads were requested")

    note = (f"pair={'|'.join(PAIR)} | POST-HOC trend: log-log slope "
            f"{trend['slope']:+.3f} -- hypothesis-generating only")
    _write(os.path.join(a.outdir, "stormer_lead_robustness.csv"), FIELDS, rows, note)

    print(f"\n=== headline pair, per lead ===")
    print(f"pair {PAIR[0]} vs {PAIR[1]}  (inherited from Aurora's lead_robustness, "
          f"not picked from the table below)\n")
    print(f"{'lead':>6}{'phys':>9}{'rmse':>9}{'ratio':>8}  binding")
    for r in rows:
        print(f"{r['lead']:>5}h{r['physics']:9.4f}{r['rmse']:9.4f}{r['ratio']:8.2f}"
              f"  {r['physics_argmax']}/{r['rmse_argmax']}")
    print(f"\nphysics leads {n_win}/{len(rows)} leads")

    print(f"\n--- POST HOC trend (not pre-registered) ---")
    print(f"log(ratio) ~ log(lead) slope {trend['slope']:+.3f}")

    # --- 2. descriptive all-pairs table ------------------------------------------------
    desc = []
    for lead in LEADS:
        tags = sorted({t for p in PAIRS for t in p})
        tables = lr.build_tables(pt, lead, tags, AXIS_FAMILIES)
        for r in rcs.compare(tables, PAIRS):
            r["lead"] = lead
            desc.append(r)
    _write(os.path.join(a.outdir, "stormer_lead_by_pair.csv"), FIELDS, desc,
           "DESCRIPTIVE ONLY -- all 6 pairs x 4 leads. The headline pair is "
           f"{'|'.join(PAIR)} alone.")

    print(f"\n=== descriptive: all pairs x leads ===")
    print(f"{'pair':26}" + "".join(f"{l:>10}h" for l in LEADS))
    for a_tag, b_tag in PAIRS:
        key = f"{a_tag}|{b_tag}"
        cells = {r["lead"]: r["ratio"] for r in desc if r["pair"] == key}
        print(f"  {a_tag:24}" + "".join(f"{cells.get(l, float('nan')):10.2f} " for l in LEADS))


if __name__ == "__main__":
    main()
