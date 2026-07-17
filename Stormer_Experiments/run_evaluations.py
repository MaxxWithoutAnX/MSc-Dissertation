from collections import defaultdict
from eval_metrics import compute_all_metrics, difference_kinetic_energy
import datetime
import glob
import os
import sys
import numpy as np
import torch
import xarray as xr

key = sys.argv[1] if len(sys.argv) > 1 else "FP32"
baseline_dir = "results_FP32"
results_dir = f"results_{key}"
metrics_file = f"all_metrics_{key}.pt"
compute_pert = baseline_dir != results_dir

metadata = torch.load(os.path.join(results_dir, "_meta.pt"), weights_only=False)
VARIABLES = metadata["variables"]      # 69 names, e.g. "2m_temperature", "geopotential_500"
lat = np.asarray(metadata["lat"])      # (128,) ascending, 1.40625 deg
H = lat.shape[0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}", flush=True)
print(f"key:    {results_dir} -> {metrics_file}", flush=True)

norm_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stormer", "normalization_constants")
norm_mean = dict(np.load(os.path.join(norm_dir, "normalize_mean.npz")))
norm_std = dict(np.load(os.path.join(norm_dir, "normalize_std.npz")))
mean_t = torch.from_numpy(np.concatenate([norm_mean[v] for v in VARIABLES])).float().view(-1, 1, 1)
std_t = torch.from_numpy(np.concatenate([norm_std[v] for v in VARIABLES])).float().view(-1, 1, 1)


def denorm(arr):
    """Normalised fp16 ndarray [V, H, W] or [1, V, H, W] -> physical float32 [1, V, H, W]."""
    t = torch.from_numpy(np.asarray(arr)).float() * std_t + mean_t
    return t if t.dim() == 4 else t.unsqueeze(0)


# --- climatology for ACC; vars without a slice get NaN and are skipped by acc() ---
clim_path = r"C:\Users\maxsh\era5_processed\clim_1p40625.nc"
clim_ds = xr.open_dataset(clim_path) if os.path.exists(clim_path) else None
if clim_ds is None:
    print("no climatology found -- ACC will be skipped", flush=True)


def clim_slice_for(valid_time, W):
    """Return [V, H, W] tensor aligned with VARIABLES, NaN where no climatology exists."""
    if clim_ds is None:
        return None
    doy = valid_time.timetuple().tm_yday  # 1..366
    hour = valid_time.hour
    out = np.full((len(VARIABLES), H, W), np.nan, dtype=np.float32)
    for i, v in enumerate(VARIABLES):
        if v in clim_ds.data_vars:                      # surface variables
            da = clim_ds[v]
        else:                                           # pressure-level vars stored as `<var>_<level>`
            base, _, lvl_str = v.rpartition("_")
            if not base or base not in clim_ds.data_vars:
                continue
            try:
                lvl = int(lvl_str)
            except ValueError:
                continue
            level_dim = "level" if "level" in clim_ds[base].dims else "pressure_level"
            if lvl not in clim_ds[base][level_dim].values:
                continue
            da = clim_ds[base].sel({level_dim: lvl})
        out[i] = da.sel(dayofyear=doy, hour=hour).transpose("latitude", "longitude").values.astype(np.float32)
    return torch.from_numpy(out)


def to_cpu(obj):
    """Recursively move tensors in a (possibly nested) metrics dict back to CPU, so the
    saved file loads without a GPU and create_plots' .numpy() calls work."""
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    return obj


result_files = sorted(glob.glob(os.path.join(results_dir, "20*.pt")))
all_metrics = defaultdict(dict)
for path in result_files:
    data = torch.load(path, map_location="cpu", weights_only=False)
    date = data["init_time"]
    print(f"computing metrics for {date}", flush=True)

    base_results = None
    if compute_pert:
        base_path = os.path.join(baseline_dir, os.path.basename(path))
        if os.path.exists(base_path):
            base_results = torch.load(base_path, map_location="cpu", weights_only=False)["results"]
        else:
            pass

    for lead_time, preds in data["results"].items():
        pred = denorm(preds["pred"]).to(device)   # [1, V, H, W]
        y = denorm(preds["gt"]).to(device)

        valid_time = date + datetime.timedelta(hours=int(lead_time))
        clim = clim_slice_for(valid_time, pred.shape[-1])

        metrics = compute_all_metrics(pred, y, clim, VARIABLES, lat, lead_time)
        metrics.pop("bias", None)

        if base_results is not None and lead_time in base_results:
            pred_fp = denorm(base_results[lead_time]["pred"]).to(device)
            metrics["dke_pert"] = difference_kinetic_energy(pred, pred_fp, VARIABLES, lat, lead_time)

        all_metrics[date][lead_time] = to_cpu(metrics)

torch.save({k: dict(v) for k, v in all_metrics.items()}, metrics_file)
print(f"wrote {metrics_file}", flush=True)
