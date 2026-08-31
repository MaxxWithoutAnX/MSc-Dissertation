"""Thin driver for the precision harness: reads harness_configs.csv and scores each config."""
import physq_path

import argparse
import json
import os

import numpy as np
import torch
import xarray as xr

import precision_harness as ph
from ablation_scripts.groups import build_groups
from distortion import NoiseFloor
from plot_common import MetricsRun
from select_harness_configs import decode_config

CKPT = "/vol/bitbucket/mes25/aurora_checkpoints/aurora-0.25-pretrained.ckpt"

SURF = {"2t": "2m_temperature", "10u": "10m_u_component_of_wind",
        "10v": "10m_v_component_of_wind", "msl": "mean_sea_level_pressure"}
ATMOS = {"t": "temperature", "u": "u_component_of_wind", "v": "v_component_of_wind",
         "q": "specific_humidity", "z": "geopotential"}
STATIC = {"lsm": "land_sea_mask", "z": "geopotential_at_surface", "slt": "soil_type"}


def build_arg_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="harness_configs.csv")
    ap.add_argument("--data", default="data/era5_sampled_2021_4pm_aurora_0p25.nc")
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--n-inits", type=int, default=48, dest="n_inits")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--leads", default="24,72,120,168")
    ap.add_argument("--score-lead", type=int, default=120, dest="score_lead")
    ap.add_argument("--warmup-reps", type=int, default=1, dest="warmup_reps")
    ap.add_argument("--latency-reps", type=int, default=3, dest="latency_reps")
    ap.add_argument("--configs", default=None)
    ap.add_argument("--resume", dest="resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    return ap


def parse_args(argv=None):
    ap = build_arg_parser()
    args = ap.parse_args(argv)
    leads = [int(x) for x in args.leads.split(",")]
    if args.score_lead not in leads:
        ap.error(f"--score-lead {args.score_lead} not in --leads {leads}")
    for lead in leads:
        if lead // 6 - 1 >= args.steps:
            ap.error(f"--leads includes {lead}h, which needs step index "
                    f"{lead // 6 - 1}, but --steps is only {args.steps}")
    args.leads = leads
    return args


def compute_scored_dates(ds, keys):
    times = ds.time.values
    return [times[i].astype("datetime64[s]").tolist() for i in sorted(keys)]


def select_init_indices(ds, n_inits=0):
    init_ts = [np.datetime64(s) for s in json.loads(ds.attrs["init_times"])]
    times = ds.time.values
    idx = []
    for t in init_ts:
        found = np.where(times == t)[0]
        if len(found):
            idx.append(int(found[0]))
    if n_inits and len(idx) > n_inits:
        idx = idx[:: max(1, len(idx) // n_inits)][:n_inits]
    return idx


def build_dataset_accessors(ds):
    """make_batch/gt_lookup/pred_to_tensor/variables/lat_eval, built once in main()."""
    from aurora import Batch, Metadata
    levels = tuple(int(l) for l in ds.level.values)
    lat = torch.from_numpy(ds.latitude.values)
    lon = torch.from_numpy(ds.longitude.values)
    lat_eval = ds.latitude.values[:-1]
    variables = (list(SURF.values())
                + [f"{long}_{lvl}" for long in ATMOS.values() for lvl in levels])

    def make_batch(i):
        history = lambda name: torch.from_numpy(ds[name].isel(time=[i - 1, i]).values[None])
        return Batch(
            surf_vars={k: history(v) for k, v in SURF.items()},
            static_vars={k: torch.from_numpy(ds[v].values) for k, v in STATIC.items()},
            atmos_vars={k: history(v) for k, v in ATMOS.items()},
            metadata=Metadata(lat=lat, lon=lon,
                              time=(ds.time.values[i].astype("datetime64[s]").tolist(),),
                              atmos_levels=levels))

    def pred_to_tensor(b):
        chans = [b.surf_vars[k][0, -1].float().cpu().numpy() for k in SURF]
        for k in ATMOS:
            chans.extend(b.atmos_vars[k][0, -1].float().cpu().numpy())
        return torch.from_numpy(np.stack(chans).astype(np.float32))[None]

    times = ds.time.values
    idx_of = {t: k for k, t in enumerate(times)}

    def gt_lookup(i, lead):
        t0 = ds.time.values[i]
        j = idx_of.get(t0 + np.timedelta64(lead, "h"))
        if j is None:
            return None
        chans = [ds[v].isel(time=j).values[:-1] for v in SURF.values()]
        for v in ATMOS.values():
            chans.extend(ds[v].isel(time=j).values[:, :-1])
        return torch.from_numpy(np.stack(chans).astype(np.float32))[None]

    return {"variables": variables, "lat_eval": lat_eval, "make_batch": make_batch,
            "pred_to_tensor": pred_to_tensor, "gt_lookup": gt_lookup}


def load_dataset(path):
    if os.path.exists(path):
        return xr.open_dataset(path)
    raise FileNotFoundError(f"{path} not found")


def _output_paths(outdir):
    return {
        "csv": os.path.join(outdir, "harness_results.csv"),
        "by_lead_csv": os.path.join(outdir, "harness_results_by_lead.csv"),
        "pt": os.path.join(outdir, "harness_results.pt"),
    }


def make_model_builder(checkpoint_path, device):
    from aurora import AuroraPretrained
    cache = {}

    def build_model():
        m = AuroraPretrained(autocast=True)
        if checkpoint_path and os.path.exists(checkpoint_path):
            if "sd" not in cache:
                m.load_checkpoint_local(checkpoint_path)
                cache["sd"] = {k: v.cpu() for k, v in m.state_dict().items()}
            else:
                m.load_state_dict(cache["sd"])
        return m.eval().to(device)

    return build_model


def run_one(tag, config, groups_map, build_model, acc, init_indices, steps,
           lead_by_step, compile_enabled, warmup_reps, latency_reps):
    return ph.measure_config(
        tag, config, groups_map, build_model, acc["make_batch"], acc["gt_lookup"],
        acc["pred_to_tensor"], acc["variables"], acc["lat_eval"], init_indices, steps,
        lead_by_step, compile_enabled=compile_enabled, warmup_reps=warmup_reps,
        timed_reps=latency_reps)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)
    paths = _output_paths(args.outdir)

    manifest_rows = ph.read_manifest(args.manifest)
    if args.configs:
        wanted = set(args.configs.split(","))
        manifest_rows = [r for r in manifest_rows if r["tag"] in wanted]

    skipped = [r for r in manifest_rows if r["floor"] == "W8A8_sq"]
    for r in skipped:
        print(f"SKIP {r['tag']}: floor=W8A8_sq not implemented "
             f"'SmoothQuant (W8A8_sq)')", flush=True)
    manifest_rows = [r for r in manifest_rows if r["floor"] != "W8A8_sq"]

    checkpoint_path = args.checkpoint
    device = "cuda" if torch.cuda.is_available() else "cpu"
    build_model = make_model_builder(checkpoint_path, device)

    template = build_model()
    groups_map, _ = build_groups(template)
    all_groups = sorted(groups_map)
    del template

    ds = load_dataset(args.data)
    acc = build_dataset_accessors(ds)
    init_indices = select_init_indices(ds, args.n_inits)
    lead_by_step = {lead // 6 - 1: lead for lead in args.leads}
    steps = args.steps

    per_config_metrics = {}
    if args.resume and os.path.exists(paths["pt"]):
        per_config_metrics = torch.load(paths["pt"], weights_only=False)
    csv_rows, by_lead_rows = [], []
    if args.resume and os.path.exists(paths["csv"]):
        csv_rows = ph.read_manifest(paths["csv"])
    if args.resume and os.path.exists(paths["by_lead_csv"]):
        by_lead_rows = ph.read_manifest(paths["by_lead_csv"])

    if "FP32" not in per_config_metrics:
        print("measuring FP32 baseline...", flush=True)
        fp32_config = {g: "bf16" for g in all_groups}
        fp32_metrics, _ = run_one("FP32", fp32_config, groups_map, build_model, acc,
                                  init_indices, steps, lead_by_step,
                                  compile_enabled=False, warmup_reps=0, latency_reps=1)
        per_config_metrics["FP32"] = fp32_metrics
        ph.write_results_pt(paths["pt"], per_config_metrics)

    fp32_run = MetricsRun("FP32", per_config_metrics["FP32"])
    scored_dates = compute_scored_dates(ds, fp32_run.dates)
    rows_todo = ph.resume_skip(manifest_rows, {r["tag"] for r in csv_rows})

    noise_floor = NoiseFloor.from_detailed()
    if noise_floor.is_null:
        print("NOISE FLOOR: NULL (validation/noise_floor_detailed.pt not found) -- "
             "gating disabled, sigma=0 for every metric", flush=True)
    else:
        print("NOISE FLOOR: LOADED from validation/noise_floor_detailed.pt", flush=True)

    for row in rows_todo:
        tag, floor = row["tag"], row["floor"]
        config = decode_config(row["config"], floor, all_groups)
        print(f"measuring {tag} ({floor})...", flush=True)
        metrics, stats = run_one(tag, config, groups_map, build_model, acc,
                                 init_indices, steps, lead_by_step,
                                 compile_enabled=True,
                                 warmup_reps=args.warmup_reps,
                                 latency_reps=args.latency_reps)
        per_config_metrics[tag] = metrics
        config_run = MetricsRun(tag, metrics)

        axes_by_lead = ph.measured_consistency_axes_all_leads(
            fp32_run, config_run, tag, args.leads, scored_dates, noise_floor=noise_floor)
        for lead in args.leads:
            by_lead_rows.append({
                "tag": tag, "floor": floor, "guide": row.get("guide", ""), "lead": lead,
                "balance": f"{axes_by_lead[lead]['balance']:.6g}",
                "conservation": f"{axes_by_lead[lead]['conservation']:.6g}",
            })
        headline = axes_by_lead[args.score_lead]

        csv_rows.append({
            "tag": tag, "floor": floor, "guide": row.get("guide", ""),
            "predicted_cost": row["cost"],
            "balance": f"{headline['balance']:.6g}",
            "conservation": f"{headline['conservation']:.6g}",
            "config": row["config"],
            "model_size_gb": f"{stats['model_size_gb']:.6g}",
            "latency_s_per_step": f"{stats['latency_s_per_step']:.6g}",
            "vram_peak_gb": "" if stats["vram_peak_gb"] is None else f"{stats['vram_peak_gb']:.6g}",
            "vram_reserved_gb": "" if stats["vram_reserved_gb"] is None else f"{stats['vram_reserved_gb']:.6g}",
            "n_inits": len(init_indices),
        })
        ph.write_results_pt(paths["pt"], per_config_metrics)
        ph.write_results_csv(paths["csv"], csv_rows)
        ph.write_results_by_lead_csv(paths["by_lead_csv"], by_lead_rows)
        print(f"  wrote {tag} -> {paths['csv']} ({len(csv_rows)}/{len(manifest_rows)})",
             flush=True)

    print(f"done: {len(csv_rows)} configs measured -> {paths['csv']}", flush=True)


if __name__ == "__main__":
    main()
