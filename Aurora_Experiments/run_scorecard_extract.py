"""Reduce the all_metrics_*.pt runs to the compact CSV the scorecard figures plot."""
import physq_path

import argparse
import csv
import gc
import os
from collections import namedtuple

import numpy as np
import torch

import ablation_comp as ac
from distortion import NoiseFloor
from plot_common import MetricsRun

FP32_KEY = "FP32"
LEADS = (24, 72, 120, 168)

SCHEMES = {"aurora": ("W4", "W8", "W8A8", "W8A8_sq", "W8_g64"),
           "stormer": ("W4", "W8", "W8A8")}

HEADLINE_LEVELS = (50, 250, 500, 700, 850)
BALANCE_LEVELS = (50, 250, 500, 850)          # no moisture level needed for wind balance

UPPER = (("Z", "geopotential"), ("T", "temperature"), ("Q", "specific_humidity"),
         ("U", "u_component_of_wind"), ("V", "v_component_of_wind"))
SURFACE = (("T2M", "2m_temperature"), ("U10", "10m_u_component_of_wind"),
           ("V10", "10m_v_component_of_wind"), ("MSLP", "mean_sea_level_pressure"))
UPPER_BLOCK = {"Z": "GEOPOTENTIAL", "T": "TEMPERATURE", "Q": "HUMIDITY",
               "U": "U-WIND", "V": "V-WIND"}

#
Row = namedtuple("Row", "label block kind group key agg")


def _raw(label, block, group, key, agg="mean"):
    return Row(label, block, "raw", group, key, agg)


def _spec(label, block, agg="mean"):
    return Row(label, block, "spec", None, None, agg)


def build_cards():
    """{card name: [Row, ...]} in draw order."""
    rmse = []
    for short, var in UPPER:
        for L in HEADLINE_LEVELS:
            rmse.append(_raw(f"{short} {L}", UPPER_BLOCK[short], "RMSE",
                             f"w_rmse_{var}_{L}_{{lt}}", agg="rms"))
    for short, var in SURFACE:
        rmse.append(_raw(short, "SURFACE", "RMSE", f"w_rmse_{var}_{{lt}}", agg="rms"))

    phys = [_spec(f"RMSE {v}", "RMSE", agg="rms") for v in ("Z500", "T850", "Q700")]
    for lbl, group, fmt in (("|Vag|", "wind_balance", "wbal_vag_pred_{L}_{{lt}}"),
                            ("Vag/Vg", "wind_balance", "wbal_ageo_geo_pred_{L}_{{lt}}")):
        for L in BALANCE_LEVELS:
            phys.append(_raw(f"{lbl} {L}hPa", "BALANCE", group, fmt.format(L=L)))
    phys.append(_spec("HypsRel 600-500", "BALANCE"))
    for m in ("|DryAir Md err|", "neg-q fraction", "|neg-q mass|"):
        phys.append(_spec(m, "CONSERVATION"))
    for fam in ("SpecDivW1", "SpecResLog"):
        for v in ("Z500", "T850", "Q700"):
            phys.append(_spec(f"{fam} {v}", "SPECTRAL"))
    for v in ("Z500", "T850", "Q700"):
        phys.append(_spec(f"|RQE| {v}", "EXTREMES"))
    return {"rmse": rmse, "physics": phys}


CARDS = build_cards()

FIELDS = ("card", "scheme", "metric", "block", "order", "lead", "fp32_mean",
          "quant_mean", "pct_diff", "sigma", "floor_measured", "at_floor")


class _Label:
    """Minimal stand-in for a MetricSpec: NoiseFloor.sigma_for reads only `.label`."""
    __slots__ = ("label",)

    def __init__(self, label):
        self.label = label


def reduce_series(x, agg):
    """Collapse a per-init series to the single number the card compares."""
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x ** 2))) if agg == "rms" else float(np.mean(x))


def scorecard_cell(quant_series, fp32_series, sigma, agg="mean"):
    """One cell: the signed percent change versus FP32, under `agg`."""
    f = reduce_series(fp32_series, agg)
    q = reduce_series(quant_series, agg)
    delta = q - f
    mean_delta = float(np.mean(quant_series)) - float(np.mean(fp32_series))
    return {"fp32_mean": f,
            "quant_mean": q,
            "pct_diff": 100.0 * delta / f if f != 0.0 else float("nan"),
            "sigma": float(sigma),
            "floor_measured": bool(sigma > 0.0),
            "at_floor": bool(abs(mean_delta) <= sigma) if sigma > 0.0 else False}


def _series(run, row, spec_by_label, lead):
    """[D] per-init values for one Row, or None when this run cannot supply it."""
    try:
        if row.kind == "raw":
            return run.scalar_at(row.group, row.key, lead)
        spec = spec_by_label.get(row.label)
        return None if spec is None else ac.spec_series(run, spec, lead)
    except (KeyError, ValueError, TypeError, IndexError):
        return None


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def extract(root, model, floor_path, leads=LEADS):
    """One dict per (card, scheme, metric, lead). Loads one quantised run at a time."""
    fp32_path = os.path.join(root, f"all_metrics_{FP32_KEY}.pt")
    if not os.path.exists(fp32_path):
        raise SystemExit(f"missing baseline {fp32_path}")
    fp32 = MetricsRun(FP32_KEY, _load(fp32_path))
    print(f"baseline {FP32_KEY}: {len(fp32.dates)} inits, leads {fp32.leads}", flush=True)

    floor = NoiseFloor.from_detailed(os.path.join(root, floor_path))
    spec_by_label = {s.label: s for s in ac.metric_registry(fp32)}

    rows = []
    for scheme in SCHEMES[model]:
        path = os.path.join(root, f"all_metrics_{scheme}.pt")
        if not os.path.exists(path):
            print(f"  skip {scheme} (no {path})", flush=True)
            continue
        quant = MetricsRun(scheme, _load(path))
        n = miss = 0
        for card, defs in CARDS.items():
            for lead in leads:
                if lead not in quant.leads or lead not in fp32.leads:
                    continue
                for i, row in enumerate(defs):
                    q = _series(quant, row, spec_by_label, lead)
                    f = _series(fp32, row, spec_by_label, lead)
                    if q is None or f is None:
                        miss += 1
                        continue
                    cell = scorecard_cell(q, f, floor.sigma_for(_Label(row.label), lead),
                                          agg=row.agg)
                    rows.append({"card": card, "scheme": scheme, "metric": row.label,
                                 "block": row.block, "order": i, "lead": lead, **cell})
                    n += 1
        print(f"  {scheme}: {n} cells" + (f" ({miss} unavailable)" if miss else ""),
              flush=True)
        del quant
        gc.collect()
    return rows


def write_csv(path, rows):
    outdir = os.path.dirname(path)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="aurora", choices=tuple(SCHEMES))
    ap.add_argument("--root", default=".", help="project root holding all_metrics_*.pt")
    ap.add_argument("--out", default=None,
                    help="results dir for scorecard.csv (default: <root>/harness_results_n48 "
                         "for aurora, <root>/harness_results_n47 for stormer)")
    ap.add_argument("--noise-floor", dest="floor", default="noise_floor_detailed.pt")
    a = ap.parse_args(argv)

    outdir = a.out or os.path.join(
        a.root, "harness_results_n48" if a.model == "aurora" else "harness_results_n47")
    rows = extract(a.root, a.model, a.floor)
    if not rows:
        raise SystemExit("no cells extracted")
    path = write_csv(os.path.join(outdir, "scorecard.csv"), rows)
    by_card = {c: sum(1 for r in rows if r["card"] == c) for c in CARDS}
    floored = sum(1 for r in rows if r["at_floor"])
    nofloor = sum(1 for r in rows if not r["floor_measured"])
    print(f"\nwrote {path}: {len(rows)} cells {by_card}; "
          f"{floored} at the noise floor, {nofloor} with no measured floor")


if __name__ == "__main__":
    main()
