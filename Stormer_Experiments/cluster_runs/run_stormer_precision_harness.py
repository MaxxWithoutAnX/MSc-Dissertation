"""Drives stormer_precision_harness over the frozen 38-config manifest on held-out 2021."""
import argparse
import datetime
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import physq_path

import stormer_precision_harness as H
from prepare_year_h5 import MODEL_VARIABLES

CKPT = "/vol/bitbucket/mes25/stormer_checkpoints/stormer_1.40625_patch_size_2.ckpt"
ROOT_DIR = "/vol/bitbucket/mes25/wb2_h5df"
NORM_DIR = "/vol/bitbucket/mes25/stormer/normalization_constants"
EXPECTED_TOTAL_PARAMS = 456_413_184
EXPECTED_ALL_PARAMS = 468_751_636
FP32_KEY = "FP32"


# --- pure helpers (unit-tested without any data) --------------------------------------

def file_to_datetime(filepath, data_freq=6):
    """wb2_h5df files are named {year}_{idx:04}.h5 on a `data_freq`-hourly grid. Same
    implementation as ablation_scripts/stormer_ablations_W8A8.py."""
    name = os.path.basename(filepath).split(".")[0]
    year, idx = map(int, name.split("_"))
    return datetime.datetime(year, 1, 1) + datetime.timedelta(hours=idx * data_freq)


def init_key(filepath):
    """'YYYYMMDD_HH' -- the OAT scripts' key format. ablation_comp.recover_dates parses
    it directly, so the NC_PATH 2020-calendar fallback is never reached. It also sorts
    lexicographically in chronological order, which MetricsRun.dates relies on."""
    return f"{file_to_datetime(filepath):%Y%m%d_%H}"


def select_init_indices(inp_file_paths, n_inits, year=2021, per_month=4,
                        hours=(0, 12), leads=(24, 72, 120, 168)):
    from download_wb2_sampled import select
    wanted = set(select(year, per_month, list(hours), list(leads))[0])
    indices = [i for i, p in enumerate(inp_file_paths)
               if file_to_datetime(p) in wanted]
    if n_inits and len(indices) > n_inits:
        stride = -(-len(indices) // n_inits)      # ceil, so the span covers the year
        indices = indices[::stride][:n_inits]
    return indices


def expand_config(cell, floor, all_groups):
    from select_harness_configs import decode_config
    config = decode_config(cell, floor, all_groups)
    unknown = set(config) - set(all_groups)
    if unknown:
        raise KeyError(f"config names groups not in the model: {sorted(unknown)}")
    return config


def filter_unrunnable(rows):
    runnable = [r for r in rows if r["floor"] != "W8A8_sq"]
    skipped = [r for r in rows if r["floor"] == "W8A8_sq"]
    return runnable, skipped


# --- model + data plumbing ------------------------------------------------------------

def build_transforms():
    from torchvision.transforms import transforms

    mean = np.concatenate([dict(np.load(os.path.join(NORM_DIR, "normalize_mean.npz")))[v]
                           for v in MODEL_VARIABLES], axis=0)
    std = np.concatenate([dict(np.load(os.path.join(NORM_DIR, "normalize_std.npz")))[v]
                          for v in MODEL_VARIABLES], axis=0)
    inp_transform = transforms.Normalize(mean, std)

    out_transforms = {}
    for lead in H.INTERVALS:
        diff_std = np.concatenate(
            [dict(np.load(os.path.join(NORM_DIR, f"normalize_diff_std_{lead}.npz")))[v]
             for v in MODEL_VARIABLES], axis=0)
        out_transforms[lead] = transforms.Normalize(np.zeros_like(diff_std), diff_std)

    std_reverse = 1.0 / std
    reverse = transforms.Normalize(-mean * std_reverse, std_reverse)
    return inp_transform, out_transforms, reverse


def make_model_builder(checkpoint_path, device, inp_transform, out_transforms):
    from stormer.models.hub.stormer import Stormer
    from stormer.models.iterative_module import GlobalForecastIterativeModule
    cache = {}

    def build_model():
        net = Stormer(in_img_size=[128, 256], variables=MODEL_VARIABLES, patch_size=2,
                      hidden_size=1024, depth=24, num_heads=16, mlp_ratio=4)
        if "sd" not in cache:
            m = GlobalForecastIterativeModule(net, pretrained_path=checkpoint_path)
            cache["sd"] = {k: v.cpu() for k, v in m.state_dict().items()}
        else:
            m = GlobalForecastIterativeModule(net)
            m.load_state_dict(cache["sd"])
        total = sum(p.numel() for p in m.parameters())
        if total != EXPECTED_ALL_PARAMS:
            raise RuntimeError(
                f"{total:,} params, expected {EXPECTED_ALL_PARAMS:,}. patch_size=4 "
                f"gives 464,156,752 and is a DIFFERENT model from the one the OAT "
                f"sensitivity tables and the Phase 2 cost tables describe. (This is the "
                f"ALL-parameter count; the quantisable-group total Phase 2 froze is "
                f"{EXPECTED_TOTAL_PARAMS:,} and is checked in build_groups_on.)")
        m.set_transforms(inp_transform, out_transforms)
        return m.eval().to(device)

    return build_model


def build_groups_on(build_model):
    """Group map from a throwaway model build, dropped immediately so it does not hold
    VRAM for the whole run."""
    from stormer_groups import build_groups
    model = build_model()
    groups_map, params = build_groups(model)
    grouped = sum(params.values())
    if grouped != EXPECTED_TOTAL_PARAMS:
        raise RuntimeError(
            f"quantisable params = {grouped:,}, expected {EXPECTED_TOTAL_PARAMS:,} "
            f"(457,261,056 would mean patch_size=4). Check patch_size, the variable "
            f"list length.")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return groups_map, params


# --- output ---------------------------------------------------------------------------

def write_outputs(outdir, store, score_lead, leads):
    """Rewrite both CSVs from the whole store. Measured axes are cached into each store
    entry when the config is first scored, so this is pure dict shuffling -- cheap enough
    to run after every config, which means a killed job still leaves consistent CSVs."""
    rows, by_lead = [], []
    for tag, entry in store.items():
        if tag == FP32_KEY:
            continue
        axes = entry["axes"]
        base = dict(entry["row"])
        base["predicted_cost"] = base.pop("cost", "")
        base.update(entry["stats"])
        base["n_inits"] = entry["n_inits"]
        base["balance"] = axes[score_lead]["balance"]
        base["conservation"] = axes[score_lead]["conservation"]
        rows.append(base)
        for lead in leads:
            by_lead.append({"tag": tag, "floor": base["floor"], "guide": base["guide"],
                            "lead": lead, "balance": axes[lead]["balance"],
                            "conservation": axes[lead]["conservation"]})
    H.write_results_csv(os.path.join(outdir, "harness_results.csv"), rows)
    H.write_results_by_lead_csv(
        os.path.join(outdir, "harness_results_by_lead.csv"), by_lead)


# --- CLI ------------------------------------------------------------------------------

def build_arg_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", default="stormer_harness_configs.csv")
    ap.add_argument("--data-split", default="era5_2021", dest="data_split")
    ap.add_argument("--root-dir", default=ROOT_DIR, dest="root_dir")
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--outdir", default="harness_results_2021_stormer")
    ap.add_argument("--n-inits", type=int, default=12, dest="n_inits")
    ap.add_argument("--leads", default="24,72,120,168")
    ap.add_argument("--score-lead", type=int, default=120, dest="score_lead")
    ap.add_argument("--warmup-reps", type=int, default=1, dest="warmup_reps")
    ap.add_argument("--latency-reps", type=int, default=3, dest="latency_reps")
    ap.add_argument("--configs", default=None,
                    help="comma-separated tags; default is every runnable manifest row")
    ap.add_argument("--no-compile", dest="compile_enabled", action="store_false",
                    default=True)
    ap.add_argument("--resume", dest="resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    return ap


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    leads = [int(x) for x in args.leads.split(",")]
    if args.score_lead not in leads:
        raise SystemExit(f"--score-lead {args.score_lead} is not in --leads {leads}")

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    inp_transform, out_transforms, reverse = build_transforms()
    build_model = make_model_builder(args.checkpoint, device, inp_transform,
                                     out_transforms)

    from stormer.data.iterative_dataset import ERA5MultiLeadtimeDataset
    from stormer.data.multi_step_datamodule import collate_fn_val
    dataset = ERA5MultiLeadtimeDataset(
        root_dir=os.path.join(args.root_dir, args.data_split),
        variables=MODEL_VARIABLES, transform=inp_transform,
        list_lead_times=leads, data_freq=6)
    dataset.inp_file_paths = dataset.file_paths
    indices = select_init_indices(dataset.inp_file_paths, args.n_inits)
    loader = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False,
                        collate_fn=collate_fn_val)
    paths = [dataset.inp_file_paths[i] for i in indices]
    keys = [init_key(p) for p in paths]
    dates = [file_to_datetime(p) for p in paths]
    print(f"{len(indices)} inits, {keys[0]} .. {keys[-1]}, leads {leads}", flush=True)

    lat = np.load(os.path.join(args.root_dir, "lat.npy"))
    groups_map, _params = build_groups_on(build_model)
    all_groups = list(groups_map)
    print(f"{len(all_groups)} groups: {all_groups}", flush=True)

    rows = H.read_manifest(args.manifest)
    runnable, skipped = filter_unrunnable(rows)
    for r in skipped:
        print(f"SKIP {r['tag']}: floor W8A8_sq needs the SmoothQuant "
              f"PREPARE->calibrate->CONVERT flow, which is not wired (as on Aurora)",
              flush=True)
    if args.configs:
        wanted = {t.strip() for t in args.configs.split(",")}
        runnable = [r for r in runnable if r["tag"] in wanted]
    print(f"{len(runnable)} configs to run ({len(skipped)} skipped)", flush=True)

    pt_path = os.path.join(args.outdir, "harness_results.pt")
    store = torch.load(pt_path, weights_only=False) if (
        args.resume and os.path.exists(pt_path)) else {}

    fp32_metrics, fp32_winds = _fp32_baseline(store, pt_path, build_model, loader, leads,
                                              lat, reverse, keys)

    from plot_common import MetricsRun
    fp32_run = MetricsRun(FP32_KEY, fp32_metrics)

    timing_batch = next(iter(loader))
    for row in runnable:
        tag = row["tag"]
        if args.resume and tag in store and "axes" in store[tag]:
            print(f"skip {tag} (already in {pt_path})", flush=True)
            continue
        config = expand_config(row["config"], row["floor"], all_groups)
        print(f"=== {tag}  floor={row['floor']}  "
              f"{row['config'] or '(uniform floor)'}", flush=True)
        metrics, stats, _winds = H.measure_config(
            tag, config, groups_map, build_model, loader, leads, MODEL_VARIABLES, lat,
            reverse, init_key_of=lambda i: keys[i], timing_batch=timing_batch,
            compile_enabled=args.compile_enabled, warmup_reps=args.warmup_reps,
            timed_reps=args.latency_reps, ref_winds=fp32_winds)
        axes = H.measured_consistency_axes_all_leads(
            fp32_run, MetricsRun(tag, metrics), tag, leads, dates)
        store[tag] = {"metrics": metrics, "stats": stats, "row": row, "axes": axes,
                      "n_inits": len(keys)}
        torch.save(store, pt_path)                       # incremental: resume-safe
        write_outputs(args.outdir, store, args.score_lead, leads)
        print(f"saved {tag} -> {pt_path} "
              f"(balance {axes[args.score_lead]['balance']:.4f}, "
              f"conservation {axes[args.score_lead]['conservation']:.4f})", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_outputs(args.outdir, store, args.score_lead, leads)
    print(f"done: {len([k for k in store if k != FP32_KEY])} configs in {args.outdir}",
          flush=True)


def _fp32_baseline(store, pt_path, build_model, loader, leads, lat, reverse, keys):
    """(metrics, winds). Metrics come from the store when resuming; the wind reference
    for dke_pert is always recomputed because it is far too large to persist."""
    cached = FP32_KEY in store
    print("recomputing FP32 wind reference" if cached else "measuring FP32 baseline",
          flush=True)
    model = build_model()
    metrics, winds = H.stormer_metrics_over_inits(
        model, loader, leads, MODEL_VARIABLES, lat, reverse,
        init_key_of=lambda i: keys[i], keep_winds=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not cached:
        store[FP32_KEY] = {"metrics": metrics}
        torch.save(store, pt_path)
    return store[FP32_KEY]["metrics"], winds


if __name__ == "__main__":
    main()
