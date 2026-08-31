print('hi', flush=True)
import os
import numpy as np
import torch
from torchvision.transforms import transforms
from stormer.data.iterative_dataset import ERA5MultiLeadtimeDataset
from stormer.models.iterative_module import GlobalForecastIterativeModule
from stormer.models.hub.stormer import Stormer
from stormer.data.multi_step_datamodule import collate_fn_val
from torch.utils.data import DataLoader, Subset
from stormer.utils.metrics import lat_weighted_rmse
from datetime import datetime, timedelta

variables = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "geopotential_50",
    "geopotential_100",
    "geopotential_150",
    "geopotential_200",
    "geopotential_250",
    "geopotential_300",
    "geopotential_400",
    "geopotential_500",
    "geopotential_600",
    "geopotential_700",
    "geopotential_850",
    "geopotential_925",
    "geopotential_1000",
    "u_component_of_wind_50",
    "u_component_of_wind_100",
    "u_component_of_wind_150",
    "u_component_of_wind_200",
    "u_component_of_wind_250",
    "u_component_of_wind_300",
    "u_component_of_wind_400",
    "u_component_of_wind_500",
    "u_component_of_wind_600",
    "u_component_of_wind_700",
    "u_component_of_wind_850",
    "u_component_of_wind_925",
    "u_component_of_wind_1000",
    "v_component_of_wind_50",
    "v_component_of_wind_100",
    "v_component_of_wind_150",
    "v_component_of_wind_200",
    "v_component_of_wind_250",
    "v_component_of_wind_300",
    "v_component_of_wind_400",
    "v_component_of_wind_500",
    "v_component_of_wind_600",
    "v_component_of_wind_700",
    "v_component_of_wind_850",
    "v_component_of_wind_925",
    "v_component_of_wind_1000",
    "temperature_50",
    "temperature_100",
    "temperature_150",
    "temperature_200",
    "temperature_250",
    "temperature_300",
    "temperature_400",
    "temperature_500",
    "temperature_600",
    "temperature_700",
    "temperature_850",
    "temperature_925",
    "temperature_1000",
    "specific_humidity_50",
    "specific_humidity_100",
    "specific_humidity_150",
    "specific_humidity_200",
    "specific_humidity_250",
    "specific_humidity_300",
    "specific_humidity_400",
    "specific_humidity_500",
    "specific_humidity_600",
    "specific_humidity_700",
    "specific_humidity_850",
    "specific_humidity_925",
    "specific_humidity_1000",
]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f'device: {device}', flush=True)
# load pretrained model
net = Stormer(
    in_img_size=[128, 256],
    variables=variables,
    patch_size=4,
    hidden_size=1024,
    depth=24,
    num_heads=16,
    mlp_ratio=4,
)
pretrained_path = r'C:\Users\maxsh\Stormer\stormer_1.40625_patch_size_4.ckpt'
model = GlobalForecastIterativeModule(net, pretrained_path=pretrained_path).to(device)
model.eval()
print(f'model loaded!', flush=True)

# load data
root_dir = r'C:\Users\maxsh\era5_processed'
norm_dir = r'C:\Users\maxsh\Stormer\stormer\normalization_constants'
normalize_mean = dict(np.load(os.path.join(norm_dir, "normalize_mean.npz")))
normalize_mean = np.concatenate([normalize_mean[v] for v in variables], axis=0)
normalize_std = dict(np.load(os.path.join(norm_dir, "normalize_std.npz")))
normalize_std = np.concatenate([normalize_std[v] for v in variables], axis=0)
inp_transform = transforms.Normalize(normalize_mean, normalize_std)
dataset = ERA5MultiLeadtimeDataset(
    root_dir=os.path.join(root_dir, "test"),
    variables=variables,
    transform=inp_transform,
    list_lead_times=[24, 72, 120, 168],  # 1, 3, 5, 7 days
    data_freq=6,
)

# DataLoader
indices = list(range(90, 360, 2))  # ~716 init times
era5_loader = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False, collate_fn=collate_fn_val)

# Data normalisations
out_transforms = {}
for l in [6, 12, 24]:
    normalize_diff_std = dict(np.load(os.path.join(norm_dir, f"normalize_diff_std_{l}.npz")))
    normalize_diff_std = np.concatenate([normalize_diff_std[v] for v in variables], axis=0)
    out_transforms[l] = transforms.Normalize(np.zeros_like(normalize_diff_std), normalize_diff_std)
model.set_transforms(inp_transform, out_transforms)


def file_to_datetime(filepath, data_freq = 6):
    name = os.path.basename(filepath).split('.')[0]
    year, idx = map(int, name.split('_'))
    return datetime(year, 1, 1) + timedelta(hours=idx * data_freq)

#Loop through
intervals = [6, 12, 24]
out_dir = 'results'
plots_dir = os.path.join(out_dir, 'plots')
os.makedirs(plots_dir, exist_ok=True)
lat = np.load(os.path.join(root_dir, "lat.npy"))
lon = np.load(os.path.join(root_dir, "lon.npy"))
torch.save({"variables": variables, "lat": lat, "lon": lon, "lead_time": [24, 72, 120, 168]},
           os.path.join(out_dir, '_meta.pt'))

# Every time step
for i, (inp, out_data_dic, vars) in enumerate(era5_loader):
    subset_idx = era5_loader.dataset.indices[i]
    filepath = era5_loader.dataset.dataset.inp_file_paths[subset_idx]
    init_time = file_to_datetime(filepath)
    inp = inp.to(device)
    print(f'timestep: {init_time}', flush=True)
    sample = {}
    # Every lead time
    for lead_time in out_data_dic.keys():
        # Every interval (average over intervals later)
        all_preds = []
        for inter in intervals:
            with torch.no_grad():
                out = model.forward_validation(inp, vars, inter, lead_time//inter) # I trust that lead time and inter will always be divisible as I have hard coded it.
            all_preds.append(out)
        mean_pred = torch.stack(all_preds, dim=0).mean(0).detach().cpu().to(torch.float16).numpy()
        gt = out_data_dic[lead_time].detach().cpu().to(torch.float16).numpy()
        sample[lead_time] = {'pred': mean_pred, 'gt': gt}

    out_path = os.path.join(out_dir, f"{init_time:%Y%m%d_%H}.pt")
    torch.save({'init_time': init_time, 'results': sample}, out_path)
    del sample, all_preds, mean_pred, gt

# Stream results back from disk for RMSE so we never hold the full year in RAM
import glob as _glob
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

init_times = []
rmse_by_lead = defaultdict(lambda: defaultdict(list))
for fpath in sorted(_glob.glob(os.path.join(out_dir, '20*.pt'))):
    blob = torch.load(fpath, weights_only=False)
    init_times.append(blob['init_time'])
    for lead_time, data in blob['results'].items():
        # promote back to fp32 for the metric computation
        pred = torch.from_numpy(data['pred']).float()
        gt = torch.from_numpy(data['gt']).float()
        rmse = lat_weighted_rmse(pred, gt, model.reverse_inp_transform, variables, lat, log_postfix=str(lead_time))
        for vari in variables:
            rmse_by_lead[lead_time][vari].append(rmse[f"w_rmse_{vari}_{lead_time}"])

rmse_per_init = {
    lt: {v: torch.stack(vals).cpu() for v, vals in by_var.items()}
    for lt, by_var in rmse_by_lead.items()
}
mean_rmses = {
    lt: {v: t.mean().item() for v, t in by_var.items()}
    for lt, by_var in rmse_per_init.items()
}
torch.save({'mean_rmses': mean_rmses,
            'rmse_per_init': rmse_per_init,
            'init_times': init_times},
           os.path.join(out_dir, '_rmse.pt'))

# Headline variables to plot (var name, display label, units)
headline = [
    ("geopotential_500", "Z500", "m"),
    ("temperature_850", "T850", "K"),
    ("2m_temperature", "T2m", "K"),
    ("10m_u_component_of_wind", "U10", "m/s"),
    ("mean_sea_level_pressure", "MSLP", "Pa"),
    ("specific_humidity_700", "Q700", "kg/kg"),
]
lead_times_sorted = sorted(mean_rmses.keys())

# Plot 1: RMSE vs lead time
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, (var, label, unit) in zip(axes.flat, headline):
    rmses = [mean_rmses[lt][var] for lt in lead_times_sorted]
    ax.plot(lead_times_sorted, rmses, marker='o')
    ax.set_xlabel('Lead time (h)')
    ax.set_ylabel(f'RMSE ({unit})')
    ax.set_title(label)
    ax.grid(True, alpha=0.3)
fig.suptitle('Lat-weighted RMSE vs lead time')
fig.tight_layout()
fig.savefig(os.path.join(plots_dir, 'rmse_vs_lead.png'), dpi=150)
plt.close(fig)

# Plot 2: RMSE seasonality (per-init RMSE through the test year)
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, (var, label, unit) in zip(axes.flat, headline):
    for lt in lead_times_sorted:
        vals = rmse_per_init[lt][var].numpy()
        ax.plot(init_times, vals, label=f'{lt}h', linewidth=0.8)
    ax.set_xlabel('Init time')
    ax.set_ylabel(f'RMSE ({unit})')
    ax.set_title(label)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
fig.suptitle('Lat-weighted RMSE seasonality')
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(os.path.join(plots_dir, 'rmse_seasonality.png'), dpi=150)
plt.close(fig)
