"""Score the headline pair on metric families the allocator never optimised."""
import physq_path

import argparse
import csv
import os

import torch

import run_paired_stats as rps

PAIR = ("W8A8_span1", "W8A8_rmse_span1")

TIERS = {
    "held_out": ["spec_div", "spec_res", "RQE"],
    "variant": ["spec_div_w1", "spec_res_log"],
    "partial": ["div_vort", "hypsometric", "lapse_rate", "neg_humidity"],
    "objective": ["wind_balance", "dry_air_mass"],
}

FIELDS = ["tier", "family", "pair", "lead", "physics", "rmse", "ratio"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pt", default=os.path.join("harness_results_n48", "harness_results.pt"))
    ap.add_argument("--outdir", default="harness_results_n48")
    ap.add_argument("--lead", type=int, default=120)
    a = ap.parse_args(argv)

    pt = torch.load(a.pt, map_location="cpu", weights_only=False, mmap=True)
    for t in PAIR:
        if t not in pt:
            raise SystemExit(f"{t} not in {a.pt}")
    os.makedirs(a.outdir, exist_ok=True)

    import ablation_comp as ac
    from plot_common import MetricsRun
    _probe = MetricsRun(PAIR[0], pt[PAIR[0]])
    _fp32 = MetricsRun("FP32", pt["FP32"])
    available = sorted({s.group for s in
                        ac.available_specs(_fp32, _probe, ac.metric_registry(_fp32), a.lead)
                        if s.group})
    print(f"registry groups available at {a.lead}h ({len(available)}):\n  "
          + ", ".join(available), flush=True)
    unknown = [f for fams in TIERS.values() for f in fams if f not in available]
    if unknown:
        print(f"  NOT AVAILABLE: {unknown}", flush=True)

    rows, missing = [], []
    for tier, families in TIERS.items():
        for fam in families:
            got = rps.compare_pairs(pt, lead=a.lead, family=fam, pairs=[PAIR])
            if not got:
                missing.append(fam)
                print(f"  MISSING: family {fam!r} produced no comparable specs", flush=True)
                continue
            for r in got:
                r["tier"] = tier
                rows.append(r)

    held = [r for r in rows if r["tier"] == "held_out"]
    if held and len(held) != len(TIERS["held_out"]):
        raise SystemExit(
            f"tested {len(held)} held-out comparisons but TIERS['held_out'] has "
            f"{len(TIERS['held_out'])} entries (missing: {missing})")

    out = os.path.join(a.outdir, "harness_heldout_families.csv")
    with open(out, "w", newline="") as f:
        f.write(f"# pair={PAIR[0]}|{PAIR[1]} lead={a.lead}. tier=held_out is the primary "
                f"evidence; tier=variant, tier=partial and tier=objective are "
                f"robustness/descriptive.\n")
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} rows)")

    print(f"\n=== {PAIR[0]} vs {PAIR[1]} @ {a.lead}h ===")
    print(f"{'tier':11}{'family':15}{'ratio':>8}")
    for tier in TIERS:
        for r in [x for x in rows if x["tier"] == tier]:
            print(f"{tier:11}{r['family']:15}{r['ratio']:8.2f}")
    print("\nratio > 1 = physics-guided is LESS distorted on that family.")
    return rows


if __name__ == "__main__":
    main()
