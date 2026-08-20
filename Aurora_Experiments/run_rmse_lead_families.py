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

PER_INIT_FIELDS = ["tag", "lead", "init", "deg_pct"]
PER_INIT_NAME = "harness_rmse_lead_per_init.csv"


def split_tag(tag, schemes):
    for scheme in sorted(schemes, key=len, reverse=True):
        if tag.startswith(scheme + "_"):
            return scheme, tag[len(scheme) + 1:]
    return "", tag


def wanted_tags():
    """Every tag figA16 and figA16b can draw: the ceiling, each scheme's physics roles, and
    each scheme's RMSE-GUIDED twins. One extraction serves both figures -- the .pt pass is
    the expensive part and neither figure is worth a second one."""
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
    # Second pass over the SAME loaded object: the per-init sample behind each mean above.
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

    missing = [t for t in wanted if t not in got]
    print(f"wrote {out} ({len(rows)} rows, {len(got)} tags)")
    if missing:
        print(f"  absent from this run: {', '.join(missing)}")
    return rows


if __name__ == "__main__":
    main()
