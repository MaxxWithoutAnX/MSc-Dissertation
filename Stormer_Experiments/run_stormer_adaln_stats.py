"""adaln's marginal contribution, as a share of the W8A8 floor's damage."""
import physq_path

import argparse
import csv
import os

import numpy as np
import torch

import paired_stats as ps
import run_composite_stats as rcs
from lead_robustness import build_tables
from run_stormer_stats import AXIS_FAMILIES, unwrap_store

KNEE = "W8A8_knee"                 # bf16:embed          -- Phase 3
PROBE = "W8A8_adaln_probe"         # bf16:adaln|embed    -- the probe run
FLOOR = "W8A8_floor"               # the uniform floor both are measured against
LEADS = (24, 72, 120, 168)
MARGIN_PCT = 5.0                   # declared above, before looking
FIELDS = ["lead", "endpoint", "knee", "probe", "floor", "delta", "pct_of_knee",
          "pct_of_floor", "within_margin"]


def merged_store(phase3_dir, probe_dir):
    p3 = unwrap_store(torch.load(os.path.join(phase3_dir, "harness_results.pt"),
                                 map_location="cpu", weights_only=False))
    pr = unwrap_store(torch.load(os.path.join(probe_dir, "harness_results.pt"),
                                 map_location="cpu", weights_only=False))
    for tag, store, where in ((KNEE, p3, phase3_dir), (FLOOR, p3, phase3_dir),
                              (PROBE, pr, probe_dir)):
        if tag not in store:
            raise SystemExit(f"{tag} not in {where}/harness_results.pt")
    a, b = p3["FP32"], pr["FP32"]
    if sorted(a) != sorted(b):
        raise SystemExit(f"the two stores scored different inits: {len(a)} vs {len(b)}")
    for d in a:
        for lead in LEADS:
            va, vb = a[d][lead]["wind_balance"], b[d][lead]["wind_balance"]
            for k in va:
                if not isinstance(va[k], (int, float, torch.Tensor, np.ndarray)):
                    continue
                x, y = float(np.asarray(va[k])), float(np.asarray(vb[k]))
                if not np.isclose(x, y, rtol=1e-9, atol=0.0):
                    raise SystemExit(f"FP32 baselines differ at {d}/{lead}h/{k}: {x} vs {y} "
                                     f"-- the difference would not isolate adaln")
    return {"FP32": p3["FP32"], FLOOR: p3[FLOOR], KNEE: p3[KNEE], PROBE: pr[PROBE]}, len(a)


def _values(tables, tag, idx, endpoint):
    """Axis or composite value for `tag` on resample `idx`."""
    if endpoint == "composite":
        return rcs.composite(tables, tag, idx)
    deltas, denoms = tables[endpoint]
    return ps.axis_value(deltas[tag], denoms, idx)


def analyse(store, leads=LEADS, margin_pct=MARGIN_PCT):
    rows = []
    for lead in leads:
        tables = build_tables(store, lead, [KNEE, PROBE, FLOOR], AXIS_FAMILIES)
        n = len(next(iter(next(iter(tables["balance"][0].values())).values())))
        full = np.arange(n)
        for endpoint in ("composite", "balance", "conservation"):
            k = _values(tables, KNEE, full, endpoint)
            p = _values(tables, PROBE, full, endpoint)
            f = _values(tables, FLOOR, full, endpoint)
            pct_of_floor = 100.0 * (k - p) / f if f else float("nan")
            rows.append({
                "lead": lead, "endpoint": endpoint, "knee": k, "probe": p, "floor": f,
                "delta": p - k, "pct_of_knee": 100.0 * (k - p) / k if k else float("nan"),
                "pct_of_floor": pct_of_floor,
                "within_margin": bool(abs(pct_of_floor) <= margin_pct)})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase3-dir", default="harness_results_2021_stormer",
                    dest="phase3_dir")
    ap.add_argument("--probe-dir", default="harness_results_adaln", dest="probe_dir")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--margin", type=float, default=MARGIN_PCT)
    ap.add_argument("--score-lead", type=int, default=120, dest="score_lead")
    a = ap.parse_args(argv)

    store, n_inits = merged_store(a.phase3_dir, a.probe_dir)
    print(f"[OK] FP32 baselines identical across stores ({n_inits} inits)", flush=True)
    rows = analyse(store, margin_pct=a.margin)

    outdir = a.outdir or a.probe_dir
    out = os.path.join(outdir, "adaln_marginal.csv")
    with open(out, "w", newline="") as f:
        f.write(f"# adaln marginal contribution, {KNEE} (bf16:embed) -> {PROBE} "
                f"(bf16:adaln|embed).\n")
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)\n", flush=True)

    print(f"=== adaln's marginal contribution, as a share of the W8A8 floor's damage ===")
    print(f"(margin +/-{a.margin}%)\n")
    print(f"{'lead':>5} {'endpoint':>13}{'knee':>9}{'+adaln':>9}"
          f"{'% of floor':>12}  within margin")
    for r in rows:
        print(f"{r['lead']:>4}h {r['endpoint']:>13}{r['knee']:9.4f}{r['probe']:9.4f}"
              f"{r['pct_of_floor']:11.2f}%  {'YES' if r['within_margin'] else 'no':>13}")


if __name__ == "__main__":
    main()
