"""Stormer noise-floor ensemble member
"""
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import physq_path

import stormer_precision_harness as H
import run_stormer_precision_harness as D
from prepare_year_h5 import MODEL_VARIABLES

SEED = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
EPS = 0.0 if SEED == 0 else float(os.environ.get("NF_EPS", "1e-6"))   # seed 0 = reference
SPLIT = os.environ.get("NF_SPLIT", "era5_2021")
ROOT = os.environ.get("NF_ROOT", D.ROOT_DIR)
CKPT = os.environ.get("NF_CKPT", D.CKPT)
N_INITS = int(os.environ.get("NF_N_INITS", "47"))
LEADS = [int(x) for x in os.environ.get("NF_LEADS", "24,72,120,168").split(",")]
OUT = os.environ.get("NF_OUT", "noise_floor_ens_n47")
COMPILE = os.environ.get("NF_COMPILE", "0") == "1"


def perturb_scale(inp_transform):
    """mean/std per channel as a [1, C, 1, 1] fp32 tensor -- the `mean/std` term of the
    identity in the module docstring. transforms.Normalize keeps the numpy arrays it was
    built from on .mean/.std (stormer/data/iterative_dataset.py reads them the same way),
    and build_transforms concatenated them in MODEL_VARIABLES order, so channel k here is
    channel k of the loader's tensor."""
    mean = np.asarray(inp_transform.mean, dtype=np.float64)
    std = np.asarray(inp_transform.std, dtype=np.float64)
    return torch.from_numpy((mean / std).astype(np.float32))[None, :, None, None]


def perturbed_batches(loader, scale, rng):

    for inp, gt_by_lead, vars_b in loader:
        if EPS != 0.0:
            g = torch.from_numpy(rng.standard_normal(tuple(inp.shape)))
            x = inp.double()
            inp = (x + EPS * (x + scale.double()) * g).to(inp.dtype)
        yield inp, gt_by_lead, vars_b


def main():
    os.makedirs(OUT, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"member seed={SEED} eps={EPS} compile={COMPILE} device={device}", flush=True)

    inp_transform, out_transforms, reverse = D.build_transforms()
    build_model = D.make_model_builder(CKPT, device, inp_transform, out_transforms)

    from stormer.data.iterative_dataset import ERA5MultiLeadtimeDataset
    from stormer.data.multi_step_datamodule import collate_fn_val
    dataset = ERA5MultiLeadtimeDataset(
        root_dir=os.path.join(ROOT, SPLIT), variables=MODEL_VARIABLES,
        transform=inp_transform, list_lead_times=LEADS, data_freq=6)
    dataset.inp_file_paths = dataset.file_paths
    indices = D.select_init_indices(dataset.inp_file_paths, N_INITS)
    loader = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False,
                        collate_fn=collate_fn_val)
    keys = [D.init_key(dataset.inp_file_paths[i]) for i in indices]
    print(f"{len(keys)} inits, {keys[0]} .. {keys[-1]}, leads {LEADS}", flush=True)
    if len(keys) != N_INITS:
        print(f"WARN: asked for {N_INITS} inits, split yielded {len(keys)}. sigma scales "
              f"as 1/sqrt(n) -- this floor only gates a run at n={len(keys)}.", flush=True)

    lat = np.load(os.path.join(ROOT, "lat.npy"))
    model = build_model()
    if COMPILE:
        H.apply_dynamo_hardening()
        H.compile_heavy_submodules(model)

    rng = np.random.default_rng(SEED)
    metrics, _winds = H.stormer_metrics_over_inits(
        model, perturbed_batches(loader, perturb_scale(inp_transform), rng),
        LEADS, MODEL_VARIABLES, lat, reverse, init_key_of=lambda i: keys[i])

    path = os.path.join(OUT, f"member_{SEED:03d}.pt")
    torch.save({"seed": SEED, "eps": EPS, "split": SPLIT, "leads": LEADS,
                "compiled": COMPILE, "keys": keys, "metrics": metrics}, path)
    print(f"saved {path} ({len(metrics)} inits, eps={EPS})", flush=True)


if __name__ == "__main__":
    main()
