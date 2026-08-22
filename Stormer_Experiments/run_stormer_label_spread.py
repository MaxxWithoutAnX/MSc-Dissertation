"""Stormer twin of Aurora_Experiments/run_label_spread.py."""
# run_stormer_label_spread.py
import physq_path

import argparse
import os

from run_label_spread import PAIR_FIELDS, REMOVED_FIELDS, pair_rows, removed_rows, write_csv

__all__ = ["PAIR_FIELDS", "REMOVED_FIELDS", "pair_rows", "removed_rows", "main"]


def parse_pairs(text):
    """'a|b,c|d' -> [('a', 'b'), ('c', 'd')], or None for 'use Stormer's PAIRS'."""
    if not text:
        return None
    return [tuple(p.split("|")) for p in text.split(",") if p]


def main(argv=None):
    from run_stormer_damage_removed import FLOOR_TAG, REMOVED_TAGS
    from run_stormer_stats import PAIRS, unwrap_store

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pt", default="harness_results_n47/harness_results.pt")
    ap.add_argument("--outdir", default="harness_results_n47")
    ap.add_argument("--lead", type=int, default=120)
    ap.add_argument("--family", default="wind_balance")
    ap.add_argument("--pairs", default=None,
                    help="comma-separated 'physics|rmse' pairs, overriding Stormer's PAIRS")
    ap.add_argument("--out", default=None,
                    help="path for the per-pair CSV (default "
                         "<outdir>/stormer_results_labels.csv)")
    a = ap.parse_args(argv)

    pairs = parse_pairs(a.pairs) or PAIRS

    import torch
    pt = unwrap_store(torch.load(a.pt, map_location="cpu", weights_only=False))
    write_csv(a.out or os.path.join(a.outdir, "stormer_results_labels.csv"), PAIR_FIELDS,
              pair_rows(pt, lead=a.lead, family=a.family, pairs=pairs))
    if a.pairs is None:
        write_csv(os.path.join(a.outdir, "stormer_damage_removed_labels.csv"),
                  REMOVED_FIELDS, removed_rows(pt, lead=a.lead, family=a.family,
                                               tags=REMOVED_TAGS, floor_tag=FLOOR_TAG))


if __name__ == "__main__":
    main()
