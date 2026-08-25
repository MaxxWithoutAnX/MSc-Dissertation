"""Extract figA16's per-lead RMSE degradation series to a small CSV, one model at a time."""
import physq_path

import argparse
import csv
import os

import torch

import figure_core as fc
import figure_specs as fs

FIELDS = ["scheme", "role", "tag", "lead", "deg_pct"]
PER_VAR_NAME = "harness_rmse_lead_per_init_var.csv"
PER_VAR_FIELDS = ["tag", "lead", "init", "var", "r_cfg", "r_ref"]
OUT_NAME = "harness_rmse_lead_families.csv"

PER_INIT_FIELDS = ["tag", "lead", "init", "deg_pct"]
PER_INIT_NAME = "harness_rmse_lead_per_init.csv"


def split_tag(tag, schemes):
    """('W8A8', 'rmse_span1') from 'W8A8_rmse_span1'. The ceiling has no scheme."""
    for scheme in sorted(schemes, key=len, reverse=True):
        if tag.startswith(scheme + "_"):
            return scheme, tag[len(scheme) + 1:]
    return "", tag


def wanted_tags():
    """Every tag figA16 and figA16b can draw: the ceiling, each scheme's physics roles, and"""
    schemes = [s for s, _ in fs.RMSE_LEAD_SCHEMES]
    roles = [r for r, _, _, _ in fs.RMSE_LEAD_ROLES]
    roles += [r for r, _, _, _ in fs.RMSE_LEAD_GUIDED_ROLES]
    out = ["ceiling"] + [f"{s}_{r}" for s in schemes for r in roles]
    return list(dict.fromkeys(out))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pt", default=os.path.join("harness_results_n48", "harness_results.pt"))
    ap.add_argument("--outdir", default="harness_results_n48")
    ap.add_argument("--leads", default=None,
                    help="comma-separated override; defaults to figA16's four leads")
    ap.add_argument("--vars", default=None,
                    help="comma-separated override; defaults to figA16's seven variables")
    a = ap.parse_args(argv)

    leads = ([int(s) for s in a.leads.split(",")] if a.leads
             else list(fs.RMSE_LEAD_LEADS))
    variables = a.vars.split(",") if a.vars else fs.RMSE_LEAD_VARS
    pt = torch.load(a.pt, map_location="cpu", mmap=True, weights_only=False)
    os.makedirs(a.outdir, exist_ok=True)

    wanted = wanted_tags()
    got = fc.prepare_rmse_lead_families(pt, wanted, leads, variables)

    schemes = [s for s, _ in fs.RMSE_LEAD_SCHEMES]
    rows = []
    for tag, series in got.items():
        scheme, role = split_tag(tag, schemes)
        for lead, val in zip(leads, series):
            rows.append({"scheme": scheme, "role": role, "tag": tag,
                         "lead": lead, "deg_pct": f"{val:.6g}"})

    out = os.path.join(a.outdir, OUT_NAME)
    with open(out, "w", newline="") as f:
        f.write(f"# mean signed RMSE degradation vs FP32, percent, per lead. "
                f"source={os.path.basename(a.pt)} vars={len(variables)}\n")
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    per = fc.prepare_rmse_lead_families(pt, wanted, leads, variables, per_init=True)
    pi_rows = [{"tag": tag, "lead": lead, "init": i, "deg_pct": f"{v:.6g}"}
               for tag, series in per.items()
               for lead, vals in zip(leads, series)
               for i, v in enumerate(vals)]
    pi_out = os.path.join(a.outdir, PER_INIT_NAME)
    with open(pi_out, "w", newline="") as f:
        f.write(f"# per-initialisation RMSE degradation vs FP32, percent. "
                f"source={os.path.basename(a.pt)} vars={len(variables)}. "
                f"Spread over inits is WEATHER VARIABILITY, not an interval on the mean -- "
                f"inits are autocorrelated and n_eff is well below n.\n")
        w = csv.DictWriter(f, fieldnames=PER_INIT_FIELDS)
        w.writeheader()
        w.writerows(pi_rows)
    print(f"wrote {pi_out} ({len(pi_rows)} rows)")

    import rmse_compression as rc
    pv_rows = []
    for tag in wanted:
        if tag not in pt:
            continue
        for lead in leads:
            keys = [f"w_rmse_{v}_{lead}" for v in variables]
            try:
                q = rc._per_init_rmse(pt, tag, lead, keys)
                f = rc._per_init_rmse(pt, "FP32", lead, keys)
            except (KeyError, TypeError, ValueError):
                continue
            for di in range(q.shape[0]):
                for vi, v in enumerate(variables):
                    pv_rows.append({"tag": tag, "lead": lead, "init": di, "var": v,
                                    "r_cfg": f"{q[di, vi]:.10g}",
                                    "r_ref": f"{f[di, vi]:.10g}"})
    pv_out = os.path.join(a.outdir, PER_VAR_NAME)
    with open(pv_out, "w", newline="") as f:
        print("# per (initialisation, variable) RMSE for the config and FP32. "
              f"source={os.path.basename(a.pt)}. Raw material for the POOLED "
              "bootstrap: pooling happens inside each resample, so a pre-collapsed "
              "degradation series cannot produce it.", file=f)
        w = csv.DictWriter(f, fieldnames=PER_VAR_FIELDS)
        w.writeheader()
        w.writerows(pv_rows)
    print(f"wrote {pv_out} ({len(pv_rows)} rows)")

    missing = [t for t in wanted if t not in got]
    print(f"wrote {out} ({len(rows)} rows, {len(got)} tags)")
    if missing:
        print(f"  absent from this run: {', '.join(missing)}")
    return rows


if __name__ == "__main__":
    main()
