""" Score a saved rollout into all_metrics.pt files for downstream analysis.
"""
from collections import defaultdict
from eval_metrics import compute_all_metrics, difference_kinetic_energy
import glob
import os
import sys
import numpy as np
import torch
import xarray as xr

YEAR = "2020"
PER_MONTH = "4"
era5_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data",
    f"era5_sampled_{YEAR}_{PER_MONTH}pm_aurora_0p25.nc",
)

key = sys.argv[1]  if len(sys.argv) > 1 else "FP32"
baseline_dir = f"results_FP32"
results_dir = f"results_{key}"
metrics_file = f"all_metrics_{key}.pt"
compute_pert = baseline_dir != results_dir

metadata = torch.load(os.path.join(results_dir, "_meta.pt"), weights_only=False)
VARIABLES = metadata["variables"]      # 69 names, e.g. "2m_temperature", "geopotential_500"
lat = np.asarray(metadata["lat"])      # (720,) - final ERA5 latitude row dropped to match preds
lon = metadata["lon"]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}", flush=True)
print(f"era5:   {era5_path}", flush=True)
print(f"key:    {results_dir}", flush=True)

ds = xr.open_dataset(era5_path)
H = lat.shape[0]                       # 720
W = ds.sizes["longitude"]             # 1440
level_pos = {int(l): k for k, l in enumerate(ds.level.values)}

def to_cpu(obj):
    """Recursively move tensors in a (possibly nested) metrics dict back to CPU, so the
    saved file loads without a GPU and create_plots' .numpy() calls work."""
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    return obj


def load_y(valid_time):
    snap = ds.sel(time=np.datetime64(valid_time))   # one timestep, lazy until .values
    cache = {}

    def block(name, atmos):
        if name not in cache:
            dims = ("level", "latitude", "longitude") if atmos else ("latitude", "longitude")
            cache[name] = snap[name].transpose(*dims).values
        return cache[name]

    out = np.empty((len(VARIABLES), H, W), dtype=np.float32)
    for i, v in enumerate(VARIABLES):
        if v in ds.data_vars:                       # surface variable (name is a data_var)
            arr = block(v, atmos=False)             # (lat, lon)
        else:                                       # "<base>_<level>", e.g. geopotential_500
            base, _, lvl = v.rpartition("_")
            arr = block(base, atmos=True)[level_pos[int(lvl)]]   # (lat, lon) at that level
        out[i] = arr[:H, :]                         # drop final latitude row -> (720, 1440)
    return torch.from_numpy(out)


result_files = sorted(glob.glob(os.path.join(results_dir, "20*.pt")))
all_metrics = defaultdict(dict)
for path in result_files:
    data = torch.load(path, weights_only=False)
    date = data['init_time']
    print(f"computing metrics for {date}", flush=True)

    base_results = None
    if compute_pert:
        base_path = os.path.join(baseline_dir, os.path.basename(path))
        base_results = torch.load(base_path, weights_only=False)["results"]
    
    for lead_time, preds in data['results'].items():
        pred = torch.from_numpy(np.asarray(preds["pred"])).float().unsqueeze(0).to(device)  # [1, V, H, W]
        y = load_y(preds["gt"]).unsqueeze(0).to(device)  

        metrics = compute_all_metrics(pred, y, None, VARIABLES, lat, lead_time)
        metrics.pop("bias", None)
    
        if base_results is not None and lead_time in base_results:
            pred_fp = torch.from_numpy(
                np.asarray(base_results[lead_time]["pred"])).float().unsqueeze(0).to(device)
            metrics["dke_pert"] = difference_kinetic_energy(pred, pred_fp, VARIABLES, lat, lead_time)
        
        all_metrics[date][lead_time] = to_cpu(metrics)

torch.save({k: dict(v) for k, v in all_metrics.items()}, metrics_file)
print(f"wrote {metrics_file}", flush=True)