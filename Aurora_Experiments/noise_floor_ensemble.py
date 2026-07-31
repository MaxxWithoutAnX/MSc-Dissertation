import os
import json

import numpy as np
import torch
import xarray as xr
from aurora import AuroraPretrained, Batch, Metadata, rollout
from eval_metrics import compute_all_metrics

CKPT = "/vol/bitbucket/mes25/aurora_checkpoints/aurora-0.25-pretrained.ckpt"

SEED = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
EPS = 0.0 if SEED == 0 else float(os.environ.get("NF_EPS", "1e-6"))   # seed 0 = reference
YEAR = int(os.environ.get("SAMPLED_YEAR", "2021"))
PER_MONTH = int(os.environ.get("SAMPLED_PER_MONTH", "4"))
N_INITS = int(os.environ.get("NF_N_INITS", "12"))
OUT = os.environ.get("NF_OUT", "noise_floor_ens")
os.makedirs(OUT, exist_ok=True)

SURF = {"2t": "2m_temperature", "10u": "10m_u_component_of_wind",
        "10v": "10m_v_component_of_wind", "msl": "mean_sea_level_pressure"}
ATMOS = {"t": "temperature", "u": "u_component_of_wind", "v": "v_component_of_wind",
         "q": "specific_humidity", "z": "geopotential"}
STATIC = {"lsm": "land_sea_mask", "z": "geopotential_at_surface", "slt": "soil_type"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"member seed={SEED} eps={EPS} device={device}", flush=True)
model = AuroraPretrained(autocast=True)
model.load_checkpoint_local(CKPT)
model.eval().to(device)

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                    f"era5_sampled_{YEAR}_{PER_MONTH}pm_aurora_0p25.nc")
ds = xr.open_dataset(DATA)
levels = tuple(int(l) for l in ds.level.values)
lat = torch.from_numpy(ds.latitude.values)          # full 721 grid -> model input
lon = torch.from_numpy(ds.longitude.values)
lat_eval = ds.latitude.values[:-1]                  # cropped 720 -> metrics scoring
VARIABLES = (list(SURF.values())
             + [f"{long}_{lvl}" for long in ATMOS.values() for lvl in levels])

rng = np.random.default_rng(SEED)


def _perturb(a):
    """Round-off-scale multiplicative Gaussian perturbation (identity for the reference)."""
    if EPS == 0.0:
        return a
    return (a * (1.0 + EPS * rng.standard_normal(a.shape))).astype(a.dtype)


def make_batch(i):
    hist = lambda name: torch.from_numpy(_perturb(ds[name].isel(time=[i - 1, i]).values[None]))
    return Batch(
        surf_vars={k: hist(v) for k, v in SURF.items()},
        static_vars={k: torch.from_numpy(ds[v].values) for k, v in STATIC.items()},
        atmos_vars={k: hist(v) for k, v in ATMOS.items()},
        metadata=Metadata(lat=lat, lon=lon,
                          time=(ds.time.values[i].astype("datetime64[s]").tolist(),),
                          atmos_levels=levels))


def pred_to_tensor(b):
    """Prediction Batch -> [1, V, H, W] fp32 tensor, ordered like VARIABLES."""
    chans = [b.surf_vars[k][0, -1].float().cpu().numpy() for k in SURF]
    for k in ATMOS:
        chans.extend(b.atmos_vars[k][0, -1].float().cpu().numpy())
    return torch.from_numpy(np.stack(chans).astype(np.float32))[None]


def gt_to_tensor(j):
    """ERA5 truth at time index j -> [1, V, H, W], lat cropped to 720 to match preds."""
    chans = [ds[v].isel(time=j).values[:-1] for v in SURF.values()]
    for v in ATMOS.values():
        chans.extend(ds[v].isel(time=j).values[:, :-1])
    return torch.from_numpy(np.stack(chans).astype(np.float32))[None]


LEADS = {24: 3, 72: 11, 120: 19, 168: 27}      # lead hours -> rollout step (matches harness)
STEPS = max(LEADS.values()) + 1
LEAD_BY_STEP = {step: lead for lead, step in LEADS.items()}

times = ds.time.values
idx_of = {t: k for k, t in enumerate(times)}
init_ts = [np.datetime64(s) for s in json.loads(ds.attrs["init_times"])]
init_indices = []
for t in init_ts:
    found = np.where(times == t)[0]
    if len(found):
        init_indices.append(int(found[0]))
if N_INITS and len(init_indices) > N_INITS:
    init_indices = init_indices[:: max(1, len(init_indices) // N_INITS)][:N_INITS]
print(f"{len(init_indices)} inits, leads {sorted(LEADS)}, steps {STEPS}", flush=True)

out = {}
for i in init_indices:
    out[i] = {}
    t0 = ds.time.values[i]
    with torch.inference_mode():
        for k, pred in enumerate(rollout(model, make_batch(i), steps=STEPS)):
            if k in LEAD_BY_STEP:
                lead = LEAD_BY_STEP[k]
                j = idx_of.get(t0 + np.timedelta64(lead, "h"))
                if j is None:
                    pass
                else:
                    out[i][lead] = compute_all_metrics(
                        pred_to_tensor(pred), gt_to_tensor(j), None, VARIABLES, lat_eval, lead)
            del pred
    print(f"  scored init {i}", flush=True)

path = os.path.join(OUT, f"member_{SEED:03d}.pt")
torch.save({"seed": SEED, "eps": EPS, "year": YEAR, "metrics": out}, path)
print(f"saved {path} ({len(out)} inits, eps={EPS})", flush=True)
