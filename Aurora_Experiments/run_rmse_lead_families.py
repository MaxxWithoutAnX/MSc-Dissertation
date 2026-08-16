"""Extract figA16's per-lead RMSE degradation series to a small CSV, one model at a time."""
import physq_path

import argparse
import csv
import os

import torch

import figure_core as fc
import figure_specs as fs

FIELDS = ["scheme", "role", "tag", "lead", "deg_pct"]
OUT_NAME = "harness_rmse_lead_families.csv"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pt", default=os.path.join("harness_results_n48", "harness_results.pt"))
    ap.add_argument("--outdir", default="harness_results_n48")
    a = ap.parse_args(argv)

    leads = list(fs.RMSE_LEAD_LEADS)
    pt = torch.load(a.pt, map_location="cpu", mmap=True, weights_only=False)
    os.makedirs(a.outdir, exist_ok=True)

    # Every tag the figure can draw: the ceiling reference plus each scheme's roles.
    wanted = ["ceiling"] + [f"{scheme}_{role}"
                            for scheme, _ in fs.RMSE_LEAD_SCHEMES
                            for role, _, _, _ in fs.RMSE_LEAD_ROLES]
    got = fc.prepare_rmse_lead_families(pt, wanted, leads, fs.RMSE_LEAD_VARS)

    rows = []
    for tag, series in got.items():
        if tag == "ceiling":
            scheme, role = "", "ceiling"
        else:
            scheme, role = tag.rsplit("_", 1)
        for lead, val in zip(leads, series):
            rows.append({"scheme": scheme, "role": role, "tag": tag,
                         "lead": lead, "deg_pct": f"{val:.6g}"})

    out = os.path.join(a.outdir, OUT_NAME)
    with open(out, "w", newline="") as f:
        f.write(f"# mean signed RMSE degradation vs FP32, percent, per lead. "
                f"source={os.path.basename(a.pt)} vars={len(fs.RMSE_LEAD_VARS)}\n")
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    missing = [t for t in wanted if t not in got]
    print(f"wrote {out} ({len(rows)} rows, {len(got)} tags)")
    if missing:
        print(f"  absent from this run: {', '.join(missing)}")
    return rows


if __name__ == "__main__":
    main()
