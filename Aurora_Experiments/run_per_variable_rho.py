"""Per-variable Spearman rho between RMSE degradation and physics balance distortion."""
# run_per_variable_rho.py
import physq_path

import argparse
import csv
import os

import numpy as np
import torch

HEADLINE = {"Z500": "geopotential_500", "T850": "temperature_850", "T2M": "2m_temperature",
            "MSLP": "mean_sea_level_pressure", "Q700": "specific_humidity_700",
            "U500": "u_component_of_wind_500", "V500": "v_component_of_wind_500",
            "U10": "10m_u_component_of_wind", "V10": "10m_v_component_of_wind"}

CONFIGS = ["W8A8_floor", "W8A8_knee", "W8A8_rmse_knee", "W8A8_probe_divergent",
           "W8A8_probe_rmse_favoured", "W8A8_rand_0", "W8A8_rand_2", "W8A8_rand_3",
           "W8A8_rmse_span1", "W8A8_rand_1", "W8A8_span1", "W8A8_span2",
           "W8A8_rmse_span2", "W8A8_span3", "W8A8_rmse_span3", "ceiling"]

FIELDS = ["var", "family", "lead", "rho"]


def _rank(a):
    """Average-tie ranks, so Spearman is correct when two configs tie exactly."""
    a = np.asarray(a, dtype=np.float64)
    order = a.argsort()
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(len(a), dtype=np.float64)
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    if (counts > 1).any():                      # average the tied blocks
        sums = np.zeros(len(counts))
        np.add.at(sums, inv, ranks)
        ranks = (sums / counts)[inv]
    return ranks


def _spearman(x, y):
    rx, ry = _rank(x), _rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom else float("nan")


def per_init_fracs(pt, tag, lead, var_key):
    """(config - fp32)/fp32 per init for one variable. Shape (n_dates,)."""
    fp = pt["FP32"]
    inits = sorted(fp.keys())
    out = []
    for i in inits:
        f = float(fp[i][lead]["RMSE"][var_key])
        c = float(pt[tag][i][lead]["RMSE"][var_key])
        out.append((c - f) / f)
    return np.asarray(out)


def compute_rho(pt, balance, configs, var_key, lead=120):
    """Spearman rho across configs, on the full sample only."""
    fr = np.vstack([per_init_fracs(pt, t, lead, var_key) for t in configs])   # (C, D)
    bal = np.asarray([balance[t] for t in configs], dtype=np.float64)
    return {"rho": _spearman(fr.mean(axis=1), bal)}


def rho_jackknife(pt, balance, configs, var_key, lead=120):
    fr = np.vstack([per_init_fracs(pt, t, lead, var_key) for t in configs])   # (C, D)
    means = fr.mean(axis=1)
    bal = np.asarray([balance[t] for t in configs], dtype=np.float64)
    out = []
    for i, tag in enumerate(configs):
        keep = [j for j in range(len(configs)) if j != i]
        out.append({"dropped": tag, "rho": _spearman(means[keep], bal[keep])})
    return out


JACKKNIFE_FIELDS = ["var", "lead", "dropped", "rho"]
JACKKNIFE_NAME = "harness_per_variable_rho_jackknife.csv"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="harness_results_n48")
    ap.add_argument("--lead", type=int, default=120)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    out = a.out or os.path.join(a.results, "harness_per_variable_rmse.csv")
    pt = torch.load(os.path.join(a.results, "harness_results.pt"), weights_only=False)
    floored = os.path.join(a.results, "harness_results_floored.csv")
    gated = os.path.exists(floored)
    src = floored if gated else os.path.join(a.results, "harness_results.csv")
    with open(src) as f:
        balance = {r["tag"]: float(r["balance"]) for r in csv.DictReader(f)}

    configs = [t for t in CONFIGS if t in pt and t in balance]
    rows, jack = [], []
    for short, long in HEADLINE.items():
        var_key = f"w_rmse_{long}_{a.lead}"
        r = compute_rho(pt, balance, configs, var_key, lead=a.lead)
        rows.append({"var": short, "family": "per_variable_rmse", "lead": a.lead, **r})
        jack += [{"var": short, "lead": a.lead, **j}
                 for j in rho_jackknife(pt, balance, configs, var_key, lead=a.lead)]

    jpath = os.path.join(a.results, JACKKNIFE_NAME)
    with open(jpath, "w", newline="") as f:
        f.write(f"# leave-one-CONFIG-out Spearman rho, lead={a.lead}, "
                f"n_configs={len(configs)}, balance_source={os.path.basename(src)}. "
                f"INFLUENCE, not sampling uncertainty: the config set varies and the dates "
                f"are held fixed -- this is unrelated to sampling uncertainty, which is not "
                f"estimated anywhere in this driver any more.\n")
        w = csv.DictWriter(f, fieldnames=JACKKNIFE_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(jack)
    print(f"wrote {jpath} ({len(jack)} rows)")

    with open(out, "w", newline="") as f:
        f.write(f"# family=per_variable_rmse lead={a.lead} n_configs={len(configs)} "
                f"balance_source={os.path.basename(src)} "
                f"balance_gated={'yes' if gated else 'NO'}\n")
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows, {len(configs)} configs)")

    print(f"\n=== per-variable rho, lead={a.lead}h ===")
    print(f"{'var':6}{'rho':>8}")
    for r in rows:
        print(f"{r['var']:6}{r['rho']:8.3f}")


if __name__ == "__main__":
    main()
