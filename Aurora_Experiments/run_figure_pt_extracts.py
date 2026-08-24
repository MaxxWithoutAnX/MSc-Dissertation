"""Reduce harness_results.pt to the compact CSVs that figA19 and figA21 plot."""
import physq_path

import argparse
import csv
import os

import numpy as np
import torch

from plot_common import block_bootstrap
from rmse_compression import tag_data

LEAD = 120
SPECTRAL_FAMILY = "power spectrum"
BALANCE_FAMILY = "wind_balance"
SPECTRAL_VARS = (("Z500", "geopotential_500"), ("U500", "u_component_of_wind_500"))
# Tags the two figures draw. FP32 is the reference for both and is not optional.
SERIES = ("FP32", "W8A8_floor", "W8A8_span1")
LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)


def _mean_psd(pt, tag, var, inits, lead):
    rows = []
    for i in inits:
        try:
            rows.append(np.asarray(
                tag_data(pt, tag)[i][lead][SPECTRAL_FAMILY][
                    f"psd_preds_{var}_{lead}"], dtype=float))
        except (KeyError, TypeError, ValueError):
            return None
    return np.mean(rows, axis=0) if rows else None


def extract_spectral(pt, inits, lead=LEAD):
    """(psd_rows, amp_rows). Empty lists when the spectral family is absent -- that family
    is not in the harness's default metric set, so absence is a normal state."""
    psd_rows, amp_rows = [], []
    for var_label, var in SPECTRAL_VARS:
        pf = _mean_psd(pt, "FP32", var, inits, lead)
        if pf is None:
            continue
        try:
            k = np.asarray(tag_data(pt, "FP32")[inits[0]][lead][SPECTRAL_FAMILY][
                f"wavenumbers_{var}_{lead}"], dtype=float)
        except (KeyError, TypeError, ValueError):
            continue
        m = k > 0                    # k=0 is the domain mean; a ratio there is not a scale
        for tag in SERIES:
            if tag == "FP32":
                continue
            pc = _mean_psd(pt, tag, var, inits, lead)
            if pc is None:
                continue
            for kk, rr in zip(k[m], (pc / pf)[m]):
                psd_rows.append({"var": var_label, "tag": tag, "lead": lead,
                                 "k": f"{kk:.10g}", "ratio": f"{rr:.10g}"})
        floor = _mean_psd(pt, "W8A8_floor", var, inits, lead)
        if floor is None:
            continue
        w = k[m] ** 2
        dp, dg = pf[m].sum(), (w * pf[m]).sum()
        plain = (floor[m].sum() - dp) / dp if dp else None
        grad = ((w * floor[m]).sum() - dg) / dg if dg else None
        if plain and grad is not None:
            amp_rows.append({"var": var_label, "lead": lead,
                             "plain_frac": f"{plain:.10g}", "grad_frac": f"{grad:.10g}",
                             "amplification": f"{grad / plain:.10g}"})
    return psd_rows, amp_rows


def extract_balance_profile(pt, inits, lead=LEAD):
    """Mean +/- standard error of the ageostrophic/geostrophic ratio per pressure level."""
    rows = []
    for tag in SERIES:
        for lev in LEVELS:
            try:
                s = np.array([float(tag_data(pt, tag)[i][lead][BALANCE_FAMILY][
                    f"wbal_ageo_geo_pred_{lev}_{lead}"]) for i in inits])
            except (KeyError, TypeError, ValueError):
                return []            # partial profiles would plot as silent gaps
            block = max(1, round(len(s) ** (1.0 / 3.0)))
            m, lo, hi = block_bootstrap(s, block=block)
            rows.append({"tag": tag, "lead": lead, "level": lev,
                         "mean": f"{float(m):.10g}",
                         "lo": f"{float(lo):.10g}", "hi": f"{float(hi):.10g}"})
    return rows


def _write(path, rows, fields):
    if not rows:
        print(f"  SKIPPED {os.path.basename(path)} (family absent from the .pt)")
        return None
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path} ({len(rows)} rows)")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default="harness_results_n48")
    ap.add_argument("--pt", default=None, help="default: <results>/harness_results.pt")
    ap.add_argument("--outdir", default=None, help="default: <results>")
    ap.add_argument("--lead", type=int, default=LEAD)
    a = ap.parse_args(argv)

    pt_path = a.pt or os.path.join(a.results, "harness_results.pt")
    outdir = a.outdir or a.results
    if not os.path.exists(pt_path):
        raise SystemExit(f"{pt_path} not found")
    print(f"loading {pt_path} ({os.path.getsize(pt_path) / 1e9:.1f} GB, mmap) ...", flush=True)
    pt = torch.load(pt_path, map_location="cpu", weights_only=False, mmap=True)
    if "FP32" not in pt:
        raise SystemExit("no FP32 baseline in the .pt; both extracts are relative to it")
    inits = sorted(tag_data(pt, "FP32").keys())
    print(f"  loaded; {len(pt)} tags, {len(inits)} inits", flush=True)

    psd_rows, amp_rows = extract_spectral(pt, inits, a.lead)
    _write(os.path.join(outdir, "harness_spectral_psd.csv"), psd_rows,
           ["var", "tag", "lead", "k", "ratio"])
    _write(os.path.join(outdir, "harness_spectral_amplification.csv"), amp_rows,
           ["var", "lead", "plain_frac", "grad_frac", "amplification"])
    _write(os.path.join(outdir, "harness_balance_profile.csv"),
           extract_balance_profile(pt, inits, a.lead),
           ["tag", "lead", "level", "mean", "lo", "hi"])
    return {"psd": len(psd_rows), "amp": len(amp_rows)}


if __name__ == "__main__":
    main()
