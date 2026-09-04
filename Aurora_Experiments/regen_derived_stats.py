"""One-shot regeneration of every derived statistics CSV for one harness results dir."""
import physq_path

import argparse
import os
import time
import traceback

import torch

DEFAULT_RESULTS = "harness_results_n48"

ANTICONTROL_PAIRS = ("W8A8_knee|W8A8_rmseup_knee,"
                     "W8A8_span1|W8A8_rmseup_span1,"
                     "W4W8_span1|W4W8_rmseup_span1")

STAGES = [
    ("paired", "run_paired_stats",
     lambda pt, out: ["--pt", pt, "--outdir", out, "--family", "wind_balance"]),
    ("paired_dry", "run_paired_stats",
     lambda pt, out: ["--pt", pt, "--outdir", out, "--family", "dry_air_mass"]),
    ("label_spread", "run_label_spread",
     lambda pt, out: ["--pt", pt, "--outdir", out, "--family", "wind_balance"]),
    ("composite", "run_composite_stats",
     lambda pt, out: ["--pt", pt, "--outdir", out]),
    ("anticontrols", "run_anticontrols",
     lambda pt, out: ["--pt", pt, "--outdir", out]),
    ("anticontrols_composite", "run_composite_stats",
     lambda pt, out: ["--pt", pt, "--outdir", out,
                      "--pairs", ANTICONTROL_PAIRS,
                      "--out", os.path.join(
                          out, "harness_results_anticontrol_composite.csv"),
                      "--labels-out", os.path.join(
                          out, "harness_results_anticontrol_composite_labels.csv")]),
    ("anticontrol_labels", "run_label_spread",
     lambda pt, out: ["--pt", pt, "--outdir", out, "--family", "wind_balance",
                      "--pairs", ANTICONTROL_PAIRS,
                      "--out", os.path.join(
                          out, "harness_results_anticontrol_labels.csv")]),
    ("per_variable", "run_per_variable_rho",
     lambda pt, out: ["--results", out]),
    ("lead_robustness", "run_aurora_lead_robustness",
     lambda pt, out: ["--pt", pt, "--outdir", out]),
    ("rmse_crossing", "run_rmse_crossing",
     lambda pt, out: ["--pt", pt, "--outdir", out]),
    ("figure_pt_extracts", "run_figure_pt_extracts",
     lambda pt, out: ["--pt", pt, "--outdir", out, "--results", out]),
    ("standard_axis", "run_standard_axis",
     lambda pt, out: ["--pt", pt, "--outdir", out]),
]


def install_shared_load(pt_path):
    real = torch.load
    target = os.path.abspath(pt_path)
    cache = {}

    def shared(f, *args, **kwargs):
        try:
            same = isinstance(f, (str, os.PathLike)) and os.path.abspath(f) == target
        except (TypeError, ValueError):
            same = False
        if not same:
            return real(f, *args, **kwargs)
        if "obj" not in cache:
            t0 = time.time()
            print(f"  loading {target} (mmap, once for all stages) ...", flush=True)
            cache["obj"] = real(target, map_location="cpu", weights_only=False, mmap=True)
            print(f"  loaded in {time.time() - t0:.0f}s "
                  f"({len(cache['obj'])} top-level keys)", flush=True)
        return cache["obj"]

    torch.load = shared
    return shared


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=DEFAULT_RESULTS,
                    help="results dir holding harness_results.pt; CSVs are written here")
    ap.add_argument("--only", default=None,
                    help="comma-separated stage names to run (default: all). "
                         f"Choices: {','.join(s for s, _, _ in STAGES)}")
    ap.add_argument("--dry-run", action="store_true", dest="dry_run",
                    help="print the argv each stage would get, load nothing")
    a = ap.parse_args(argv)

    pt = os.path.join(a.results, "harness_results.pt")
    stages = STAGES
    if a.only:
        want = {s.strip() for s in a.only.split(",")}
        unknown = want - {s for s, _, _ in STAGES}
        if unknown:
            ap.error(f"unknown stage(s): {sorted(unknown)}")
        stages = [s for s in STAGES if s[0] in want]

    if a.dry_run:
        for name, mod, build in stages:
            print(f"{name:16} {mod}.main({build(pt, a.results)})")
        return {"ok": [], "failed": []}

    if not os.path.exists(pt):
        raise SystemExit(f"{pt} not found")
    os.makedirs(a.results, exist_ok=True)
    install_shared_load(pt)

    import importlib
    ok, failed = [], []
    for name, modname, build in stages:
        argv_i = build(pt, a.results)
        print(f"\n{'=' * 78}\n=== {name}  ({modname})\n{'=' * 78}", flush=True)
        t0 = time.time()
        try:
            mod = importlib.import_module(modname)
            mod.main(argv_i)
        except Exception:
            traceback.print_exc()
            print(f"--- {name} FAILED after {time.time() - t0:.0f}s", flush=True)
            failed.append(name)
        else:
            print(f"--- {name} ok in {time.time() - t0:.0f}s", flush=True)
            ok.append(name)

    print(f"\n{'=' * 78}\n{len(ok)} ok, {len(failed)} failed")
    if failed:
        print("  FAILED: " + ", ".join(failed))
    return {"ok": ok, "failed": failed}


if __name__ == "__main__":
    main()
