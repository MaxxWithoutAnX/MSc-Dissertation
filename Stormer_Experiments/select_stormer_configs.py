import physq_path

import os

from allocator import load_distortion_table
from frontiers import CONSISTENCY
from select_harness_configs import select_all

import run_stormer_frontiers as rsf

HERE = os.path.dirname(os.path.abspath(__file__))
FRONTIER_DIR = os.path.join(HERE, "frontier_stormer")
MANIFEST_NAME = "stormer_harness_configs.csv"


def _rank(values, groups):
    """1 = largest. Returns {group: rank}."""
    order = sorted(groups, key=lambda g: -values.get(g, 0.0))
    return {g: i + 1 for i, g in enumerate(order)}


def derive_probe_groups(lead=120, scheme="W8A8"):
    csvs = {scheme: rsf._csv(scheme)}
    d_phys = load_distortion_table(csvs, lead=lead, axes=CONSISTENCY)
    d_rmse = load_distortion_table(csvs, lead=lead, axes=("standard",))
    groups = sorted(set(d_phys) & set(d_rmse))

    phys = {g: sum(d_phys[g][scheme][a] for a in CONSISTENCY) for g in groups}
    rmse = {g: d_rmse[g][scheme]["standard"] for g in groups}
    rp, rr = _rank(phys, groups), _rank(rmse, groups)
    shift = {g: rr[g] - rp[g] for g in groups}          # >0 => physics ranks it higher

    divergent = tuple(sorted(groups, key=lambda g: (-shift[g], -phys[g]))[:2])
    rmse_favoured = tuple(sorted(groups, key=lambda g: (shift[g], -rmse[g]))[:2])
    if set(divergent) & set(rmse_favoured):
        raise RuntimeError(
            f"probe groups overlap: divergent={divergent} rmse_favoured={rmse_favoured}; "
            f"the guides do not disagree enough on this scheme to form disjoint probes")
    return divergent, rmse_favoured


def stormer_schemes():
    return {
        "W4W8": dict(floor="W4",
                     physics=os.path.join(FRONTIER_DIR, "frontier_A_W4W8_physics.csv"),
                     rmse=os.path.join(FRONTIER_DIR, "frontier_A_W4W8_rmse.csv"),
                     order=os.path.join(FRONTIER_DIR, "guide_order_A_W4W8_rmse.csv")),
        "W8A8": dict(floor="W8A8",
                     physics=os.path.join(FRONTIER_DIR, "frontier_B_W8A8_physics.csv"),
                     rmse=os.path.join(FRONTIER_DIR, "frontier_B_W8A8_rmse.csv"),
                     order=os.path.join(FRONTIER_DIR, "guide_order_B_W8A8_rmse.csv"),
                     random=os.path.join(FRONTIER_DIR, "random_W8A8.csv")),
        "W8A8_sq": dict(floor="W8A8_sq",
                        physics=os.path.join(FRONTIER_DIR, "frontier_B_W8A8_sq_physics.csv"),
                        rmse=os.path.join(FRONTIER_DIR, "frontier_B_W8A8_sq_rmse.csv"),
                        order=os.path.join(FRONTIER_DIR, "guide_order_B_W8A8_sq_rmse.csv")),
    }


def select(outdir=HERE, k=3):
    """Write stormer_harness_configs.csv and return the selected configs."""
    divergent, rmse_favoured = derive_probe_groups()
    print(f"derived probes: divergent={divergent}  rmse_favoured={rmse_favoured}",
          flush=True)

    import select_harness_configs as shc
    real_probe_configs = shc.probe_configs

    def stormer_probe_configs(all_groups, floor, divergent=divergent,
                              rmse_favoured=rmse_favoured):
        return real_probe_configs(all_groups, floor, divergent, rmse_favoured)

    shc.probe_configs = stormer_probe_configs
    try:
        return select_all(k=k, extras_scheme="W8A8", outdir=outdir,
                          schemes=stormer_schemes(), manifest_name=MANIFEST_NAME)
    finally:
        shc.probe_configs = real_probe_configs


def main():
    cfgs = select()
    print(f"wrote {MANIFEST_NAME} ({len(cfgs)} unique models)", flush=True)
    by = {}
    for c in cfgs:
        by[c["guide"]] = by.get(c["guide"], 0) + 1
    for g, n in sorted(by.items()):
        print(f"  {g:8} {n}", flush=True)
    print("\nMANIFEST IS UNGATED (no Stormer noise-floor ensemble exists).", flush=True)


if __name__ == "__main__":
    main()
