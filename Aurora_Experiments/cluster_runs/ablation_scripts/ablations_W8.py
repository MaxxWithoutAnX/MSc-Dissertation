""" Aurora W8 ablation anaysis (one group quantised per ablation)
"""
import torch
import torch.nn as nn
from aurora import AuroraPretrained, Batch, Metadata, rollout
import xarray as xr
import datetime, os, json
import numpy as np
from groups import build_groups
from eval_metrics import compute_all_metrics, difference_kinetic_energy
#from physics_loss import physics_loss         # WIP; re-enable once physics_loss(metrics, fp32_metrics) exists
from torchao import quantize_
from torchao.quantization.quant_api import Int8WeightOnlyConfig

CKPT = "/vol/bitbucket/mes25/aurora_checkpoints/aurora-0.25-pretrained.ckpt"

_CACHED_SD = None

def fresh_model():
    global _CACHED_SD
    m = AuroraPretrained(autocast=True)
    if _CACHED_SD is None:
        m.load_checkpoint_local(CKPT)                                  # full load, once
        _CACHED_SD = {k: v.cpu() for k, v in m.state_dict().items()}
    else:
        m.load_state_dict(_CACHED_SD)                                  # cheap reuse
    print('fresh model', flush=True)
    return m.eval().to(device)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}", flush=True)
model = fresh_model()

groups, params = build_groups(model)
config = Int8WeightOnlyConfig()                                   # W8 weight-only (W8A32), as in aurora_inference_W8
print(f'total number of groups: {len(groups)}')

YEAR = int(os.environ.get("SAMPLED_YEAR", "2020"))
PER_MONTH = int(os.environ.get("SAMPLED_PER_MONTH", "4"))
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                    f"era5_sampled_{YEAR}_{PER_MONTH}pm_aurora_0p25.nc")
ds = xr.open_dataset(DATA)
out_dir = f"ablation_W8_medium_sampled_{YEAR}_{PER_MONTH}pm"
os.makedirs(out_dir, exist_ok=True)

levels = tuple(int(l) for l in ds.level.values)
lat = torch.from_numpy(ds.latitude.values)            # full grid (721) -> model input
lon = torch.from_numpy(ds.longitude.values)
lat_eval = ds.latitude.values[:-1]                    # numpy -> metrics scoring

SURF   = {"2t":"2m_temperature","10u":"10m_u_component_of_wind","10v":"10m_v_component_of_wind","msl":"mean_sea_level_pressure"}
ATMOS  = {"t":"temperature","u":"u_component_of_wind","v":"v_component_of_wind","q":"specific_humidity","z":"geopotential"}
STATIC = {"lsm":"land_sea_mask","z":"geopotential_at_surface","slt":"soil_type"}

VARIABLES = (
    list(SURF.values())
    + [f"{long}_{lvl}" for long in ATMOS.values() for lvl in levels]
)

WIND_VARS = [v for v in VARIABLES if v.startswith(("u_component_of_wind_", "v_component_of_wind_"))]
WIND_IDX = [VARIABLES.index(v) for v in WIND_VARS]

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

def pred_to_array(b):
    """Aurora prediction Batch -> [V, H, W] fp16 numpy, ordered like VARIABLES.
    surf_vars[k] is (B, T, H, W), atmos_vars[k] is (B, T, C, H, W); forward yields
    a single new step so T == 1 (take the last)."""
    assert tuple(b.metadata.atmos_levels) == levels, "atmos level order mismatch"
    chans = [b.surf_vars[k][0, -1].float().cpu().numpy() for k in SURF]   # surf x (H,W)
    for k in ATMOS:
        chans.extend(b.atmos_vars[k][0, -1].float().cpu().numpy())        # C x (H,W)
    return np.stack(chans).astype(np.float32)

def pred_to_tensor(b):
    """Prediction Batch -> [1, V, H, W] fp32 tensor, ordered like VARIABLES."""
    return torch.from_numpy(pred_to_array(b))[None]

def gt_to_tensor(i):
    """ERA5 truth at time index i -> [1, V, H, W] fp32 tensor, same order as VARIABLES.
    Latitude is cropped to 720 ([:-1]) to match the model's cropped predictions."""
    chans = [ds[v].isel(time=i).values[:-1] for v in SURF.values()]      # surf x (H,W)
    for v in ATMOS.values():
        chans.extend(ds[v].isel(time=i).values[:, :-1])                  # C x (H,W) over levels
    return torch.from_numpy(np.stack(chans).astype(np.float32))[None]


LEADS = {24: 3, 72: 11, 120: 19, 168: 27}
STEPS = max(LEADS.values()) + 1
LEAD_BY_STEP = {step: lead for lead, step in LEADS.items()}

# timestamp -> row index, for looking up each init's truth valid-times on the sparse axis.
times = ds.time.values
idx_of = {t: k for k, t in enumerate(times)}

init_ts = [np.datetime64(s) for s in json.loads(ds.attrs["init_times"])]
init_indices = []
for t in init_ts:
    idx = np.where(times == t)[0]
    if len(idx) == 0:
        continue
    init_indices.append(int(idx[0]))

N_INITS = int(os.environ.get("ABLATION_N_INITS", "0"))   # 0 -> use all inits
if N_INITS and len(init_indices) > N_INITS:
    init_indices = init_indices[:: max(1, len(init_indices) // N_INITS)][:N_INITS]
print(f"{len(init_indices)} inits, leads {sorted(LEADS)}", flush=True)

def metrics_over_inits(m, ref_preds=None, keep_preds=False):
    out = {}
    preds = {} if keep_preds else None
    for i in init_indices:
        out[i] = {}
        if keep_preds:
            preds[i] = {}
        t0 = ds.time.values[i]
        with torch.inference_mode():
            for k, pred in enumerate(rollout(m, make_batch(i), steps=STEPS)):
                if k in LEAD_BY_STEP:
                    lead = LEAD_BY_STEP[k]
                    j = idx_of.get(t0 + np.timedelta64(lead, "h"))
                    if j is None:
                        pass
                    else:
                        pred_t = pred_to_tensor(pred)
                        out[i][lead] = compute_all_metrics(
                            pred_t, gt_to_tensor(j),
                            None, VARIABLES, lat_eval, lead)
                        out[i][lead].pop('bias')
                        if ref_preds is not None or keep_preds:
                            wind = pred_t[:, WIND_IDX]                     # [1, 26, H, W]
                            if ref_preds is not None:
                                out[i][lead]["dke_pert"] = difference_kinetic_energy(
                                    wind, ref_preds[i][lead], WIND_VARS, lat_eval, lead)
                            if keep_preds:
                                preds[i][lead] = wind
                del pred
        print(f"  scored init {i} at leads {sorted(LEADS)}", flush=True)
    return out, preds


fp32_path = os.path.join(out_dir, "fp32_metrics.pt")
fp_preds_path = os.path.join(out_dir, "fp32_wind_preds.pt")   # ~5 GB; reference for dke_pert
if os.path.exists(fp32_path) and os.path.exists(fp_preds_path):
    fp32_metrics = torch.load(fp32_path)
    fp_preds = torch.load(fp_preds_path)
    print(f"loaded fp32 baseline + wind preds from {out_dir}", flush=True)
else:
    fp32_metrics, fp_preds = metrics_over_inits(model, keep_preds=True)
    torch.save(fp32_metrics, fp32_path)
    torch.save(fp_preds, fp_preds_path)
    print(f"saved fp32 baseline + wind preds to {out_dir}", flush=True)

del model
torch.cuda.empty_cache()


oat_path = os.path.join(out_dir, "oat_results.pt")
results = torch.load(oat_path) if os.path.exists(oat_path) else {}
if results:
    print(f"resuming: {len(results)} group(s) already done {sorted(results)}", flush=True)

for group, names in groups.items():
    if group in results:
        print(f"skip {group} (exists)", flush=True)
        continue
    members = set(names)
    m = fresh_model()
    quantize_(m, config=config,
              filter_fn=lambda mod, fqn, members=members: isinstance(mod, nn.Linear) and fqn in members)
    metrics, _ = metrics_over_inits(m, ref_preds=fp_preds)
    results[group] = metrics
    del m
    torch.save(results, oat_path)                                   # checkpoint after each group
    print(f"saved {group} -> {oat_path} ({len(results)}/{len(groups)} groups)", flush=True)
