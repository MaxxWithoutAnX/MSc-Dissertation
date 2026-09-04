"""Measure adaln's marginal contribution at the W8A8 floor."""
import physq_path

import argparse
import csv
import os

import numpy as np
import torch

PROBE_TAG = "W8A8_adaln_probe"
PROBE_CONFIG = "bf16:adaln|embed"
BASELINE_TAG = "W8A8_knee"            # bf16:embed, measured in Phase 3
FLOOR = "W8A8"
MANIFEST = "stormer_adaln_probe.csv"
PHASE3_DIR = "harness_results_2021_stormer"
PROBE_DIR = "harness_results_2021_adaln"
LEADS = [24, 72, 120, 168]


def write_manifest(path=MANIFEST):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "floor", "guide", "cost", "balance", "conservation", "config"])
        w.writerow([PROBE_TAG, FLOOR, "probe", "", "", "", PROBE_CONFIG])
    print(f"wrote {path}: {PROBE_TAG} | {FLOOR} | {PROBE_CONFIG}", flush=True)
    return path


def verify_baseline_config():
    man = {r["tag"]: r for r in csv.DictReader(open("stormer_harness_configs.csv"))}
    if BASELINE_TAG not in man:
        raise SystemExit(f"{BASELINE_TAG} not in stormer_harness_configs.csv")
    base = man[BASELINE_TAG]
    if base["floor"] != FLOOR:
        raise SystemExit(f"{BASELINE_TAG} floor is {base['floor']!r}, expected {FLOOR!r} -- "
                         f"a floor mismatch makes the two configs incomparable")

    from select_harness_configs import decode_config
    groups = ["embed", "adaln", "head", "b0-5_attn", "b0-5_mlp", "b6-11_attn", "b6-11_mlp",
              "b12-17_attn", "b12-17_mlp", "b18-23_attn", "b18-23_mlp"]
    cb = decode_config(base["config"], FLOOR, groups)
    cp = decode_config(PROBE_CONFIG, FLOOR, groups)
    diff = {g for g in groups if cb[g] != cp[g]}
    if diff != {"adaln"}:
        raise SystemExit(f"configs differ in {sorted(diff)}, expected exactly {{'adaln'}} -- "
                         f"the difference would not isolate adaln")
    print(f"verified: {BASELINE_TAG} ({base['config']}) vs {PROBE_TAG} ({PROBE_CONFIG}) "
          f"differ in exactly one group: adaln ({cb['adaln']} -> {cp['adaln']})", flush=True)


def measure(args):
    import run_stormer_precision_harness as D
    write_manifest()
    verify_baseline_config()
    D.main(["--manifest", MANIFEST,
            "--root-dir", args.root_dir,
            "--data-split", args.data_split,
            "--checkpoint", args.checkpoint,
            "--outdir", PROBE_DIR,
            "--n-inits", str(args.n_inits),
            "--leads", ",".join(str(x) for x in LEADS),
            "--score-lead", str(args.score_lead)])


def _axes(store_dir, tag, lead):
    """(balance, conservation) at `lead` from a harness store's cached axes."""
    pt = torch.load(os.path.join(store_dir, "harness_results.pt"),
                    map_location="cpu", weights_only=False)
    if tag not in pt:
        raise SystemExit(f"{tag} not in {store_dir}/harness_results.pt")
    axes = pt[tag]["axes"]
    key = lead if lead in axes else str(lead)
    return float(axes[key]["balance"]), float(axes[key]["conservation"]), pt


def _fp32_agrees(lead=120):
    """The two stores hold independently computed FP32 baselines. Every measured axis is a
    difference against that baseline, so if they disagree the comparison is invalid. FP32 is
    deterministic given the same inits, so this should be exact."""
    a = torch.load(os.path.join(PHASE3_DIR, "harness_results.pt"),
                   map_location="cpu", weights_only=False)["FP32"]["metrics"]
    b = torch.load(os.path.join(PROBE_DIR, "harness_results.pt"),
                   map_location="cpu", weights_only=False)["FP32"]["metrics"]
    if sorted(a) != sorted(b):
        return False, f"different inits: {len(a)} vs {len(b)}"
    for d in a:
        va = a[d][lead]["wind_balance"]
        vb = b[d][lead]["wind_balance"]
        ka = sorted(k for k in va if isinstance(va[k], (int, float, torch.Tensor)))
        for k in ka:
            x, y = float(np.asarray(va[k])), float(np.asarray(vb[k]))
            if not np.isclose(x, y, rtol=1e-9, atol=0.0):
                return False, f"{d} {k}: {x!r} vs {y!r}"
    return True, f"{len(a)} inits, wind_balance identical"


def analyse(score_lead=120):
    for d, what in ((PHASE3_DIR, "Phase 3 results"), (PROBE_DIR, "the probe run")):
        if not os.path.exists(os.path.join(d, "harness_results.pt")):
            raise SystemExit(
                f"{d}/harness_results.pt not found -- {what} has not been produced here.\n"
                f"Run this on the cluster (sbatch run_stormer_adaln_probe.sh), or copy "
                f"both result directories back before using --analyse-only.")
    ok, detail = _fp32_agrees(score_lead)
    print(f"[{'OK' if ok else 'FAIL'}] FP32 baselines agree across stores   {detail}",
          flush=True)
    if not ok:
        raise SystemExit("the two stores have different FP32 baselines -- the difference "
                         "between them would not isolate adaln")

    rows = []
    for lead in LEADS:
        bb, bc, _ = _axes(PHASE3_DIR, BASELINE_TAG, lead)
        pb, pc, _ = _axes(PROBE_DIR, PROBE_TAG, lead)
        rows.append({"lead": lead,
                     "knee_balance": bb, "probe_balance": pb,
                     "delta_balance": pb - bb,
                     "pct_balance": 100.0 * (bb - pb) / bb if bb else float("nan"),
                     "knee_conservation": bc, "probe_conservation": pc,
                     "delta_conservation": pc - bc,
                     "pct_conservation": 100.0 * (bc - pc) / bc if bc else float("nan")})

    out = os.path.join(PROBE_DIR, "adaln_marginal.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}\n", flush=True)

    print(f"=== adaln MARGINAL contribution: {BASELINE_TAG} ({BASELINE_TAG and 'bf16:embed'}) "
          f"-> {PROBE_TAG} ({PROBE_CONFIG}) ===")
    print(f"{'lead':>6} {'balance knee':>13} {'+adaln':>10} {'removed':>9}"
          f"{'cons knee':>12} {'+adaln':>10} {'removed':>9}")
    for r in rows:
        print(f"{r['lead']:>5}h {r['knee_balance']:13.5f} {r['probe_balance']:10.5f} "
              f"{r['pct_balance']:8.2f}%{r['knee_conservation']:12.5f} "
              f"{r['probe_conservation']:10.5f} {r['pct_conservation']:8.2f}%")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root-dir", default="/vol/bitbucket/mes25/wb2_h5df", dest="root_dir")
    ap.add_argument("--data-split", default="era5_2021", dest="data_split")
    ap.add_argument("--checkpoint",
                    default="/vol/bitbucket/mes25/stormer_checkpoints/"
                            "stormer_1.40625_patch_size_2.ckpt")
    ap.add_argument("--n-inits", type=int, default=12, dest="n_inits")
    ap.add_argument("--score-lead", type=int, default=120, dest="score_lead")
    ap.add_argument("--analyse-only", action="store_true", dest="analyse_only",
                    help="skip the GPU measurement and re-report from existing stores")
    a = ap.parse_args(argv)

    if not a.analyse_only:
        measure(a)
    analyse(a.score_lead)


if __name__ == "__main__":
    main()
