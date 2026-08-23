"""Re-score the measured consistency axes under alternative family definitions."""
# rescore_axes.py
import argparse
import csv
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import torch

import paired_stats as ps
from plot_common import MetricsRun, lag1_autocorr

AXIS_FAMILIES = {
    "dry_only": ["dry_air_mass"],
    "dry_and_negq": ["dry_air_mass", "neg_humidity"],
    "multi_family": ["dry_air_mass", "neg_humidity", "hypsometric", "lapse_rate",
                     "div_vort", "dke", "spec_div", "spec_res"],
}

FIELDS = ["tag", "variant", "lead", "value", "n_eff_min", "n_labels"]

DEFAULT_TAGS = ["W8A8_floor", "W8A8_knee", "W8A8_rmse_knee", "W8A8_span1",
                "W8A8_rmse_span1", "W8A8_span2", "W8A8_rmse_span2",
                "W8A8_span3", "W8A8_rmse_span3", "ceiling"]


def _family_tables(pt, lead, families, tags, dates, spread_method):
    """{family: ({tag: {label: delta array}}, {label: denom})} from a real results .pt."""
    fp32 = MetricsRun("FP32", pt["FP32"])
    out = {}
    for fam in families:
        per_tag, denoms = {}, {}
        for tag in tags:
            if tag not in pt:
                continue
            run = MetricsRun(tag, pt[tag])
            specs = ps.specs_for_family(fp32, run, fam, lead)
            if not specs:
                continue
            per_tag[tag], denoms = ps.family_deltas(fp32, run, specs, lead,
                                                    dates, spread_method)
        if per_tag:
            out[fam] = (per_tag, denoms)
    return out


def score_variant(pt=None, lead=120, families=None, tags=None, variant="", dates=None,
                  spread_method="iqr_raw", prepared=None):
    families = families or AXIS_FAMILIES["dry_only"]
    tags = tags or DEFAULT_TAGS
    tables = prepared if prepared is not None else _family_tables(
        pt, lead, families, tags, dates, spread_method)
    rows = []
    for tag in tags:
        per_family, n_effs, n_labels = [], [], 0
        for fam in families:
            if fam not in tables:
                continue
            per_tag, denoms = tables[fam]
            if tag not in per_tag:
                continue
            deltas = per_tag[tag]
            idx = np.arange(len(next(iter(deltas.values()))))
            per_family.append(ps.axis_value(deltas, denoms, idx))
            n_labels += len(deltas)
            for series in deltas.values():
                r1 = lag1_autocorr(series)
                n = len(series)
                n_effs.append(n * (1 - r1) / (1 + r1) if r1 > -1 else float(n))
        if not per_family:
            continue
        rows.append({"tag": tag, "variant": variant, "lead": lead,
                     "value": float(np.nanmean(per_family)),
                     "n_eff_min": float(min(n_effs)) if n_effs else float("nan"),
                     "n_labels": n_labels})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pt", default=os.path.join("harness_results", "harness_results.pt"))
    ap.add_argument("--lead", type=int, default=120)
    ap.add_argument("--outdir", default="harness_results")
    ap.add_argument("--dates", default=None,
                    help="JSON list of the scored init timestamps (produced on the cluster). "
                         "Switches the SVR denominator from raw IQR to the deseasonalised basis.")
    a = ap.parse_args(argv)

    dates, spread_method = None, "iqr_raw"
    if a.dates:
        import datetime as _dt
        import json
        raw = json.load(open(a.dates))
        dates = [_dt.datetime.fromisoformat(str(s).replace("Z", "").split(".")[0])
                 for s in raw]
        import ablation_comp as _ac
        spread_method = _ac.pick_svr_method(dates)
        print(f"deseasonalised denominator from {len(dates)} scored dates "
              f"-> {spread_method}", flush=True)

    pt = torch.load(a.pt, map_location="cpu", weights_only=False)
    rows = []
    for variant, families in AXIS_FAMILIES.items():
        rows += score_variant(pt, a.lead, families, DEFAULT_TAGS, variant=variant,
                              dates=dates, spread_method=spread_method)
    os.makedirs(a.outdir, exist_ok=True)
    path = os.path.join(a.outdir, "harness_axis_variants.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)", flush=True)

    by_tag = {}
    for r in rows:
        by_tag.setdefault(r["tag"], {})[r["variant"]] = r
    variants = list(AXIS_FAMILIES)
    print(f"\n=== conservation axis definition, measured @ {a.lead}h ===")
    hdr = "".join(f"{v[:12]:>13}" for v in variants)
    print(f"{'tag':24}{hdr}{'min n_eff':>11}")
    for tag, v in by_tag.items():
        if not all(name in v for name in variants):
            continue
        vals = "".join(f"{v[name]['value']:13.4f}" for name in variants)
        worst = min(v[name]["n_eff_min"] for name in variants)
        print(f"{tag:24}{vals}{worst:11.2f}")


if __name__ == "__main__":
    main()
