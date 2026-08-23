"""Per-axis rank correlations between layer-group orderings."""
# run_axis_rho.py
import physq_path

import argparse
import csv
import glob
import os

from allocator import load_distortion_table
from ranking_divergence import _pearson, _rankdata, permutation_pvalue

SCHEMES = ("W8", "W4", "W8A8", "W8A8_sq")

ROOTS = {
    "aurora": r"C:\Users\maxsh\Aurora_Experiments",
    "stormer": r"C:\Users\maxsh\Stormer",
}
PAIRS = (("balance", "standard"), ("conservation", "standard"),
         ("balance", "conservation"))


def find_sensitivity_csv(root, scheme):
    hits = []
    for pat in ("ablation_analysis_ablation_%s", "ablation_analysis_ablations_%s"):
        hits += glob.glob(os.path.join(root, pat % scheme, "sensitivity.csv"))
    hits = sorted(set(hits))
    return hits[0] if hits else None


def axis_values(csv_path, scheme, lead):
    scheme_csvs = {scheme: csv_path}
    d_phys = load_distortion_table(scheme_csvs, lead=lead,
                                   axes=("balance", "conservation"))
    d_std = load_distortion_table(scheme_csvs, lead=lead, axes=("standard",))
    groups = sorted(set(d_phys) & set(d_std))
    return {
        "balance": {g: d_phys[g][scheme]["balance"] for g in groups},
        "conservation": {g: d_phys[g][scheme]["conservation"] for g in groups},
        "standard": {g: d_std[g][scheme]["standard"] for g in groups},
    }


def rho_of(x, y):
    groups = sorted(set(x) & set(y))
    if len(groups) < 3:
        return None
    rx, ry = _rankdata([x[g] for g in groups]), _rankdata([y[g] for g in groups])
    if len(set(rx)) < 2 or len(set(ry)) < 2:
        return None
    return _pearson(rx, ry)


def rows_for(model, root, leads, perms):
    out = []
    for scheme in SCHEMES:
        path = find_sensitivity_csv(root, scheme)
        if path is None:
            print(f"  skip {model}/{scheme}: no sensitivity.csv under {root}")
            continue
        for lead in leads:
            try:
                ax = axis_values(path, scheme, lead)
            except Exception as exc:
                print(f"  skip {model}/{scheme}@{lead}h: {type(exc).__name__}: {exc}")
                continue
            if not ax["balance"]:
                print(f"  skip {model}/{scheme}@{lead}h: no groups at this lead")
                continue
            for a, b in PAIRS:
                x, y = ax[a], ax[b]
                rho = rho_of(x, y)
                groups = sorted(set(x) & set(y))
                p = (permutation_pvalue({g: x[g] for g in groups},
                                        {g: y[g] for g in groups}, n=perms)
                     if rho is not None else None)
                out.append({
                    "model": model, "scheme": scheme, "lead": lead,
                    "pair": f"{a}_vs_{b}",
                    "rho": rho, "p_perm": p, "n_groups": len(groups),
                    "n_nonzero_x": sum(1 for g in groups if x[g] > 0),
                    "n_nonzero_y": sum(1 for g in groups if y[g] > 0),
                    "source": os.path.relpath(path, root).replace("\\", "/"),
                })
    return out


def write_csv(path, rows):
    cols = ["model", "scheme", "lead", "pair", "rho", "p_perm", "n_groups",
            "n_nonzero_x", "n_nonzero_y", "source"]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            r = dict(r)
            r["rho"] = "" if r["rho"] is None else f"{r['rho']:.4f}"
            r["p_perm"] = "" if r["p_perm"] is None else f"{r['p_perm']:.4f}"
            w.writerow(r)


def report(rows, lead):
    """Human-readable table at one lead -- the form the numbers get quoted in."""
    print(f"\n=== Spearman rho over layer groups @ {lead}h (uncensored) ===")
    print(f"{'model':9s}{'scheme':10s}"
          + "".join(f"{a[:4] + '~' + b[:4]:>17}" for a, b in PAIRS))
    for model in ROOTS:
        for scheme in SCHEMES:
            sel = {r["pair"]: r for r in rows
                   if r["model"] == model and r["scheme"] == scheme and r["lead"] == lead}
            if not sel:
                continue
            cells = []
            for a, b in PAIRS:
                r = sel.get(f"{a}_vs_{b}")
                if r is None or r["rho"] is None:
                    cells.append(f"{'n/a':>17}")
                else:
                    star = "*" if r["p_perm"] is not None and r["p_perm"] < 0.05 else " "
                    cells.append(f"{r['rho']:+16.3f}{star}")
            print(f"{model:9s}{scheme:10s}" + "".join(cells))
    print("  * permutation p < 0.05 (uncorrected)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", default="aurora,stormer")
    ap.add_argument("--leads", default="24,72,120,168")
    ap.add_argument("--perms", type=int, default=10000)
    ap.add_argument("--outdir", default="harness_results_n48",
                    help="written relative to each model's own root")
    ap.add_argument("--name", default="harness_axis_rho.csv")
    a = ap.parse_args(argv)

    models = [m.strip() for m in a.models.split(",") if m.strip()]
    leads = [int(x) for x in a.leads.split(",") if x.strip()]

    allrows = []
    for model in models:
        root = ROOTS[model]
        print(f"[{model}] {root}")
        rows = rows_for(model, root, leads, a.perms)
        out = os.path.join(root, a.outdir if model == "aurora" else
                           a.outdir.replace("n48", "n47"), a.name)
        write_csv(out, rows)
        print(f"  wrote {out}  ({len(rows)} rows)")
        allrows += rows

    for lead in leads:
        report(allrows, lead)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
