print('hi', flush=True)
from aurora import AuroraPretrained, Batch, Metadata, rollout
import torch
import xarray as xr
import datetime, os, json
import numpy as np
from torchao import quantize_
from torchao.quantization.quant_api import Int8DynamicActivationInt8WeightConfig

CKPT = "/vol/bitbucket/mes25/aurora_checkpoints/aurora-0.25-pretrained.ckpt"
model = AuroraPretrained(autocast=True)
model.load_checkpoint_local(CKPT)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}", flush=True)
model.eval().to(device)

# int8 dynamic-activation + int8 weight (W8A8); every quantisable Linear in the full model (no exclusions).
quantisation_scheme = "_W8A8"

def _is_linear(m, fqn):
    return(
        isinstance(m, torch.nn.Linear)
        and m.weight.dim() == 2
        and m.weight.shape[1] > 1
    )

quantize_(model, Int8DynamicActivationInt8WeightConfig(), filter_fn=_is_linear)
print('quantised model!', flush=True)

# Sampled (seasonally-spread) ERA5 produced by wb2_download_sampled.py.
YEAR = int(os.environ.get("SAMPLED_YEAR", "2020"))
PER_MONTH = int(os.environ.get("SAMPLED_PER_MONTH", "4"))
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                    f"era5_sampled_{YEAR}_{PER_MONTH}pm_aurora_0p25.nc")
ds = xr.open_dataset(DATA)
out_dir = f"results{quantisation_scheme}_sampled_{YEAR}_{PER_MONTH}pm_Autocast"
os.makedirs(out_dir, exist_ok=True)

levels = tuple(int(l) for l in ds.level.values)
lat = torch.from_numpy(ds.latitude.values)
lon = torch.from_numpy(ds.longitude.values)

SURF   = {"2t":"2m_temperature","10u":"10m_u_component_of_wind","10v":"10m_v_component_of_wind","msl":"mean_sea_level_pressure"}
ATMOS  = {"t":"temperature","u":"u_component_of_wind","v":"v_component_of_wind","q":"specific_humidity","z":"geopotential"}
STATIC = {"lsm":"land_sea_mask","z":"geopotential_at_surface","slt":"soil_type"}

VARIABLES = (
    list(SURF.values())
    + [f"{long}_{lvl}" for long in ATMOS.values() for lvl in levels]
)

def pred_to_array(b):
    """Aurora prediction Batch -> [V, H, W] fp32 numpy, ordered like VARIABLES.
    surf_vars[k] is (B, T, H, W), atmos_vars[k] is (B, T, C, H, W); forward yields
    a single new step so T == 1 (take the last)."""
    assert tuple(b.metadata.atmos_levels) == levels, "atmos level order mismatch"
    chans = [b.surf_vars[k][0, -1].float().cpu().numpy() for k in SURF]   # surf x (H,W)
    for k in ATMOS:
        chans.extend(b.atmos_vars[k][0, -1].float().cpu().numpy())        # C x (H,W)
    return np.stack(chans).astype(np.float32)

def make_batch(i):
    history = lambda name: torch.from_numpy(ds[name].isel(time=[i-1, i]).values[None])
    return Batch(
        surf_vars = {k: history(v) for k, v in SURF.items()},
        static_vars = {k: torch.from_numpy(ds[v].values) for k, v in STATIC.items()},
        atmos_vars = {k: history(v) for k, v in ATMOS.items()},
        metadata = Metadata(lat=lat, lon=lon,
                            time = (ds.time.values[i].astype("datetime64[s]").tolist(),),
                            atmos_levels = levels),
    )

LEADS = {24: 3, 72: 11, 120:19, 168: 27}
STEPS = max(LEADS.values()) + 1

init_ts = [np.datetime64(s) for s in json.loads(ds.attrs["init_times"])]
times = ds.time.values
init_indices = []
for t in init_ts:
    idx = np.where(times == t)[0]
    if len(idx) == 0:
        continue
    init_indices.append(int(idx[0]))

torch.save({"variables": VARIABLES, "lat": ds.latitude.values[:-1], "lon": ds.longitude.values,
            "levels": levels, "lead_times": sorted(LEADS)},
           os.path.join(out_dir, "_meta.pt"))

LEAD_BY_STEP = {k: lead for lead, k in LEADS.items()}   # {3:24, 11:72, 19:120, 27:168}

print(f"starting inference now! {len(init_indices)} inits. Pray for the GPU", flush=True)

for i in init_indices:
    init_time = ds.time.values[i].astype("datetime64[s]").tolist()
    out_path = os.path.join(out_dir, f"{init_time:%Y%m%d_%H}.pt")
    if os.path.exists(out_path):
        print(f"skip {init_time:%Y-%m-%d %H}Z (exists)", flush=True)
        continue
    batch = make_batch(i)
    results = {}
    with torch.inference_mode():
        for k, pred in enumerate(rollout(model, batch, steps=STEPS)):
            if k in LEAD_BY_STEP:
                lead = LEAD_BY_STEP[k]
                gt_time = init_time + datetime.timedelta(hours=lead)
                results[lead] = {"pred": pred_to_array(pred), "gt": gt_time}
            del pred
    torch.save({"init_time": init_time, "results": results}, out_path)
    print(f"saved {init_time:%Y-%m-%d %H}Z", flush=True)
