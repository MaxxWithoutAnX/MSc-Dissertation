"""Compare Stormer harness results: composite endpoint and axis decomposition."""
import physq_path

import argparse
import csv
import os

import torch

import run_composite_stats as rcs
import run_paired_stats as rps

PAIRS = [("W8A8_span1", "W8A8_rmse_span1"),
         ("W8A8_span2", "W8A8_rmse_span2"),
         ("W8A8_span3", "W8A8_rmse_span3"),
         ("W4W8_span1", "W4W8_rmse_span1"),
         ("W4W8_span2", "W4W8_rmse_span2"),
         ("W4W8_span3", "W4W8_rmse_span3")]

AXIS_FAMILIES = {"balance": "wind_balance", "conservation": "dry_air_mass"}


def unwrap_store(store):
    out = {}
    for tag, entry in store.items():
        out[tag] = entry["metrics"] if isinstance(entry, dict) and "metrics" in entry \
            else entry
    return out


def _write(path, fields, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pt", default=os.path.join("harness_results_n47",
                                                 "harness_results.pt"))
    ap.add_argument("--outdir", default="harness_results_n47")
    ap.add_argument("--lead", type=int, default=120)
    a = ap.parse_args(argv)

    pt = unwrap_store(torch.load(a.pt, map_location="cpu", weights_only=False))
    tags = sorted({t for pair in PAIRS for t in pair})
    missing = [t for t in tags if t not in pt]
    if missing:
        raise SystemExit(f"missing tags in {a.pt}: {missing}")

    tables = {}
    for axis, fam in AXIS_FAMILIES.items():
        deltas, denoms = rps._prepared(pt, a.lead, fam, tags)
        if deltas:
            tables[axis] = (deltas, denoms)
    if not tables:
        raise SystemExit("no axis families resolved -- nothing to test")
    first_axis_deltas = next(iter(tables.values()))[0]
    n_inits = len(next(iter(next(iter(first_axis_deltas.values())).values())))
    print(f"axes resolved: {sorted(tables)}   n_inits={n_inits}", flush=True)

    os.makedirs(a.outdir, exist_ok=True)

    # --- primary endpoint: the objective the allocator actually minimises --------------
    rows = rcs.compare(tables, PAIRS)
    for r in rows:
        r["lead"] = a.lead
    _write(os.path.join(a.outdir, "stormer_composite.csv"), rcs.FIELDS, rows)

    lrows = rcs.label_rows(tables, PAIRS, a.lead)
    rcs.check_contributions(rows, lrows)
    rcs.write_label_rows(os.path.join(a.outdir, "stormer_composite_labels.csv"), lrows)

    print(f"\n=== PRIMARY: max(balance, conservation), lead {a.lead}h ===")
    print(f"{'pair':34}{'phys':>8}{'rmse':>8}{'ratio':>7}  binding")
    for r in rows:
        print(f"  {r['pair']:32}{r['physics']:8.3f}{r['rmse']:8.3f}{r['ratio']:7.2f}"
              f"  {r['physics_argmax']}/{r['rmse_argmax']}")

    # --- decomposition: each axis on its own -------------------------------------------
    for axis, fam in AXIS_FAMILIES.items():
        arows = rps.compare_pairs(lead=a.lead, family=fam, pairs=PAIRS,
                                  prepared=tables[axis])
        _write(os.path.join(a.outdir, f"stormer_paired_{fam}.csv"), rps.FIELDS, arows)
        print(f"\n--- decomposition: {axis} ({fam}) ---")
        for r in arows:
            print(f"  {r['pair']:32}{r['physics']:9.4f}{r['rmse']:9.4f}{r['ratio']:7.2f}")


if __name__ == "__main__":
    main()
