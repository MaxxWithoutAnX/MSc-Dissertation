"""Re-gate the harness's balance and conservation with a now-available noise floor."""
import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import xarray as xr

import precision_harness as ph
from distortion import NoiseFloor
from plot_common import MetricsRun
from run_precision_harness import compute_scored_dates, _output_paths


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--data", default="data/era5_sampled_2021_4pm_aurora_0p25.nc")
    ap.add_argument("--leads", default="24,72,120,168")
    ap.add_argument("--score-lead", type=int, default=120, dest="score_lead")
    a = ap.parse_args(argv)
    leads = [int(x) for x in a.leads.split(",")]
    if a.score_lead not in leads:
        ap.error(f"--score-lead {a.score_lead} not in --leads {leads}")

    paths = _output_paths(a.outdir)
    per_config = torch.load(paths["pt"], weights_only=False)
    if "FP32" not in per_config:
        raise SystemExit(f"{paths['pt']} has no FP32 baseline -- cannot deseasonalise")
    csv_rows = ph.read_manifest(paths["csv"])                 # ordered list of dict rows

    nf = NoiseFloor.from_detailed()
    if nf.is_null:
        raise SystemExit("noise floor is NULL (validation/noise_floor_detailed.pt absent) -- "
                         "nothing to re-gate; run build_ensemble_floor.py first")
    print(f"NOISE FLOOR: LOADED ({', '.join(sorted(nf._detailed))})", flush=True)

    fp32_run = MetricsRun("FP32", per_config["FP32"])
    ds = xr.open_dataset(a.data)
    dates = compute_scored_dates(ds, fp32_run.dates)

    by_lead_rows = []
    for row in csv_rows:
        tag = row["tag"]
        if tag not in per_config:
            continue
        run = MetricsRun(tag, per_config[tag])
        axes = ph.measured_consistency_axes_all_leads(fp32_run, run, tag, leads, dates,
                                                      noise_floor=nf)
        for lead in leads:
            by_lead_rows.append({
                "tag": tag, "floor": row["floor"], "guide": row.get("guide", ""),
                "lead": lead, "balance": f"{axes[lead]['balance']:.6g}",
                "conservation": f"{axes[lead]['conservation']:.6g}"})
        h = axes[a.score_lead]
        row["balance"] = f"{h['balance']:.6g}"                # overwrite ONLY these two
        row["conservation"] = f"{h['conservation']:.6g}"
        print(f"  re-gated {tag}", flush=True)

    out_csv = paths["csv"].replace(".csv", "_floored.csv")
    out_by = paths["by_lead_csv"].replace(".csv", "_floored.csv")
    ph.write_results_csv(out_csv, csv_rows)
    ph.write_results_by_lead_csv(out_by, by_lead_rows)
    print(f"done: {len(csv_rows)} configs re-gated -> {out_csv}, {out_by}", flush=True)


if __name__ == "__main__":
    main()
