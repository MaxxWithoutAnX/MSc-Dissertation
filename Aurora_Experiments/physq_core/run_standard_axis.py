"""Re-score the measured RMSE (standard) axis from harness_results.pt. No inference."""
import argparse
import csv
import importlib
import os
import sys

_CWD = os.path.abspath(os.getcwd())
if _CWD not in sys.path:
    sys.path.insert(0, _CWD)

import numpy as np
import torch

from distortion import NoiseFloor
from plot_common import MetricsRun
from rmse_compression import tag_data

DEFAULT_HARNESS = "precision_harness"

AXIS = "standard"
LEAD = 120
REFERENCE_TAG = "FP32"

RESCORED_SUFFIX = "_rescored"
CONSISTENCY = ("balance", "conservation")
FIELDS = ["tag", "lead", AXIS] + [a + RESCORED_SUFFIX for a in CONSISTENCY]


def dates_from_pt(pt):
    keys = sorted(tag_data(pt, REFERENCE_TAG).keys())
    if not keys or isinstance(keys[0], (int, np.integer)):
        return None
    import ablation_comp as ac
    got = ac.recover_dates(keys)
    if not got or not all(hasattr(d, "month") for d in got):
        return None
    return list(got)


def score_axes(pt, lead=LEAD, tags=None, dates=None, noise_floor=None,
               harness=DEFAULT_HARNESS):
    if noise_floor is None:
        noise_floor = NoiseFloor.null()
    if dates is None:
        dates = dates_from_pt(pt)
    ph = importlib.import_module(harness)
    fp32 = MetricsRun(REFERENCE_TAG, tag_data(pt, REFERENCE_TAG))
    tags = tags if tags is not None else [t for t in pt if t != REFERENCE_TAG]
    axes = (AXIS,) + CONSISTENCY
    rows = []
    for tag in tags:
        if tag not in pt:
            continue
        got = ph.measured_consistency_axes(fp32, MetricsRun(tag, tag_data(pt, tag)), tag,
                                           lead, dates, noise_floor=noise_floor, axes=axes)
        row = {"tag": tag, "lead": lead, AXIS: float(got[AXIS])}
        row.update({a + RESCORED_SUFFIX: float(got[a]) for a in CONSISTENCY})
        rows.append(row)
    return rows


def _frozen(results_csv):
    """{tag: row} from harness_results.csv, or {} -- the frozen measured values to compare
    the re-scored consistency axes against."""
    if not os.path.exists(results_csv):
        return {}
    with open(results_csv, newline="") as f:
        return {r["tag"]: r for r in csv.DictReader(l for l in f if not l.startswith("#"))}


def _report_basis_gap(rows, frozen):
    """Print re-scored vs frozen balance/conservation. This IS the caveat, quantified."""
    if not frozen:
        print("  no harness_results.csv to compare against -- basis gap NOT measured",
              flush=True)
        return
    print(f"\n=== re-scored vs frozen consistency axes (the denominator-basis gap) ===")
    print(f"{'tag':26}{'balance re':>12}{'balance frz':>13}"
          f"{'consv re':>11}{'consv frz':>11}")
    for r in rows:
        f = frozen.get(r["tag"])
        if not f:
            continue
        try:
            fb, fc = float(f["balance"]), float(f["conservation"])
        except (KeyError, ValueError):
            continue
        print(f"{r['tag']:26}{r['balance' + RESCORED_SUFFIX]:12.4f}{fb:13.4f}"
              f"{r['conservation' + RESCORED_SUFFIX]:11.4f}{fc:11.4f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pt", default=os.path.join("harness_results", "harness_results.pt"))
    ap.add_argument("--lead", type=int, default=LEAD)
    ap.add_argument("--outdir", default="harness_results")
    ap.add_argument("--dates", default=None,
                    help="JSON list of the scored init timestamps (cluster only). Switches "
                         "the SVR denominator from raw IQR to the deseasonalised basis the "
                         "harness itself used.")
    ap.add_argument("--harness", default=DEFAULT_HARNESS,
                    help="harness module supplying measured_consistency_axes. Aurora: "
                         "precision_harness (default). Stormer: stormer_precision_harness.")
    a = ap.parse_args(argv)

    import ablation_comp as _ac
    dates, source = None, ""
    if a.dates:
        import datetime as _dt
        import json
        raw = json.load(open(a.dates))
        dates = [_dt.datetime.fromisoformat(str(s).replace("Z", "").split(".")[0])
                 for s in raw]
        source = "--dates"

    pt = torch.load(a.pt, map_location="cpu", weights_only=False, mmap=True)

    if dates is None:
        dates = dates_from_pt(pt)
        source = "recovered from the .pt's own init keys" if dates else ""
    basis = _ac.pick_svr_method(dates)
    if dates:
        print(f"deseasonalised denominator from {len(dates)} init dates "
              f"({source}) -> {basis}", flush=True)
    else:
        print("SVR denominator falls back to iqr_raw: the init keys are integer indices, "
              "which do NOT determine a date, and resolving them against ablation_comp's "
              "hard-coded 2020 NC_PATH would date-stamp the wrong year. This is NOT the "
              "basis harness_results.csv's frozen balance/conservation were scored on -- "
              "pass --dates to fix it. The comparison below is the size of that gap.",
              flush=True)

    rows = score_axes(pt, lead=a.lead, dates=dates, harness=a.harness)

    os.makedirs(a.outdir, exist_ok=True)
    path = os.path.join(a.outdir, "harness_axis_standard.csv")
    with open(path, "w", newline="") as f:
        f.write(f"# measured RMSE (standard) axis re-scored from {os.path.basename(a.pt)}. "
                f"lead={a.lead} svr_basis={basis} harness={a.harness} n_tags={len(rows)}. "
                f"balance_rescored/conservation_rescored are the SAME basis as `standard` "
                f"and are for comparison against the frozen harness_results.csv columns -- "
                f"they are not a replacement for them.\n")
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows, basis={basis})", flush=True)

    _report_basis_gap(rows, _frozen(os.path.join(a.outdir, "harness_results.csv")))
    return rows


if __name__ == "__main__":
    main()
