"""Emit one cluster manifest: what the corrected chain needs that is not already measured."""
# build_merged_manifest.py
import argparse
import csv
import os

TIER_A = {
    "W8A8_knee",
    "W8A8_rmseup_knee",
    "W8A8_rmseup_span1",
    "W4W8_span1",
    "W4W8_rmse_span1",
    "W4W8_rmseup_span1",
}
TIER_B = {
    "W4W8_span2", "W4W8_span3", "W4W8_rmse_span2",
}

HEADLINE_N48 = (
    "W8A8_floor", "ceiling", "W8A8_knee", "W8A8_rmse_knee",
    "W8A8_span1", "W8A8_rmse_span1", "W8A8_rmseup_knee", "W8A8_rmseup_span1",
)
N_DEFAULT = 12
FIELDS = ["tag", "floor", "guide", "cost", "balance", "conservation", "config",
          "n_inits", "tier"]


def config_map(cfgstr):
    d = {}
    for part in (cfgstr or "").split(";"):
        if not part:
            continue
        prec, groups = part.split(":")
        for g in groups.split("|"):
            d[g] = prec
    return d


def group_universe(*row_lists):
    """Every group mentioned anywhere. The all-bf16 ceiling row lists them all."""
    u = set()
    for rows in row_lists:
        for r in rows:
            u |= set(config_map(r.get("config", "")))
    return u


def key(floor, cfgstr, universe=None):
    d = config_map(cfgstr)
    if d and set(d.values()) == {"bf16"} and universe is not None and set(d) == universe:
        floor = "bf16"
    return (floor, tuple(sorted(d.items())))


def measured_keys(path, universe=None):
    return {key(r.get("floor", ""), r.get("config", ""), universe): r["tag"]
            for r in csv.DictReader(open(path))}


def find_reusable(corrected_rows, measured, universe=None):
    """{corrected_tag: measured_tag} for byte-identical (floor, config) pairs."""
    return {r["tag"]: measured[key(r.get("floor", ""), r.get("config", ""), universe)]
            for r in corrected_rows
            if key(r.get("floor", ""), r.get("config", ""), universe) in measured}


def tier_of(tag):
    return "A" if tag in TIER_A else "B" if tag in TIER_B else "C"


def build_manifest(corrected_rows, measured, n_headline=48, max_tier="B",
                   upgrade_n=False, measured_n=None, universe=None):
    allowed = {"A": "A", "B": "AB", "C": "ABC"}[max_tier.upper()[-1]]
    measured_n = measured_n or {}
    reuse = find_reusable(corrected_rows, measured, universe)
    out = []
    for r in corrected_rows:
        needs_more = (upgrade_n and r["tag"] in HEADLINE_N48
                      and measured_n.get(reuse.get(r["tag"], r["tag"]),
                                         N_DEFAULT) < n_headline)
        if r["tag"] in reuse and not needs_more:
            continue
        t = tier_of(r["tag"])
        if t not in allowed and not needs_more:
            continue
        row = dict(r)
        row["tier"] = "N" if (needs_more and t == "C") else t
        row["n_inits"] = n_headline if r["tag"] in HEADLINE_N48 else N_DEFAULT
        out.append(row)
    return sorted(out, key=lambda r: (r["tier"], -r["n_inits"], r["tag"])), reuse


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corrected", default="harness_configs_corrected.csv")
    ap.add_argument("--measured", default=os.path.join("harness_results_merged",
                                                       "harness_results.csv"))
    ap.add_argument("--out", default="harness_configs_merged.csv")
    ap.add_argument("--n-headline", type=int, default=48)
    ap.add_argument("--max-tier", default="B", choices=["A", "B", "C"])
    ap.add_argument("--upgrade-n", action="store_true",
                    help="also re-run already-measured HEADLINE configs to reach "
                         "--n-headline inits (the existing sweep is n=12)")
    a = ap.parse_args(argv)

    corrected = list(csv.DictReader(open(a.corrected)))
    measured_rows = list(csv.DictReader(open(a.measured)))
    universe = group_universe(corrected, measured_rows)
    measured = measured_keys(a.measured, universe)
    measured_n = {r["tag"]: int(r.get("n_inits") or 12)
                  for r in csv.DictReader(open(a.measured))}
    rows, reuse = build_manifest(corrected, measured, a.n_headline, a.max_tier,
                                 a.upgrade_n, measured_n, universe)

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {a.out} ({len(rows)} configs to run)")
    print(f"\n{'tag':26}{'tier':>5}{'n':>5}{'cost':>13}")
    for r in rows:
        print(f"  {r['tag']:24}{r['tier']:>5}{r['n_inits']:>5}{float(r['cost']):13.5g}")
    gpu_h = sum(0.75 * r["n_inits"] / 12 for r in rows)
    print(f"\nestimated {gpu_h:.1f} GPU-h at ~0.75 GPU-h per config per 12 inits")
    print(f"reuse (copy, do NOT re-run): "
          + ", ".join(f"{k}<-{v}" for k, v in sorted(reuse.items())
                      if k.startswith("W8A8_rmseup")) or "none")


if __name__ == "__main__":
    main()
