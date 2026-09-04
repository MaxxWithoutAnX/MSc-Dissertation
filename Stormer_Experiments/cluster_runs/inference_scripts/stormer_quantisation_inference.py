""" Rollout of quantised Stormer
"""
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
from torchao import quantize_
from torchao.quantization.quant_api import Int8WeightOnlyConfig

quantisation_scheme = 'W8'
out_dir = rf'C:\Users\maxsh\Stormer\results_{quantisation_scheme}'

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
full_model = GlobalForecastIterativeModule(net, pretrained_path=pretrained_path).to(device)
full_model.eval()
print(f'model loaded!', flush=True)

quant_model = full_model

def _is_linear(m, fqn):
    return(
        isinstance(m, torch.nn.Linear)
        and m.weight.dim() == 2
        and m.weight.shape[1] > 1
    )

quantize_(quant_model, Int8WeightOnlyConfig(), filter_fn=_is_linear)
print('quantised model!')

# load data
root_dir = os.environ.get("STORMER_H5_ROOT", r'C:\Users\maxsh\era5_processed')
split = os.environ.get("STORMER_SPLIT", "era5_2020")
norm_dir = r'C:\Users\maxsh\Stormer\stormer\normalization_constants'
normalize_mean = dict(np.load(os.path.join(norm_dir, "normalize_mean.npz")))
normalize_mean = np.concatenate([normalize_mean[v] for v in variables], axis=0)
normalize_std = dict(np.load(os.path.join(norm_dir, "normalize_std.npz")))
normalize_std = np.concatenate([normalize_std[v] for v in variables], axis=0)
inp_transform = transforms.Normalize(normalize_mean, normalize_std)
dataset = ERA5MultiLeadtimeDataset(
    root_dir=os.path.join(root_dir, split),
    variables=variables,
    transform=inp_transform,
    list_lead_times=[24, 72, 120, 168],  # 1, 3, 5, 7 days
    data_freq=6,
)


# DataLoader: one month every 3 months (Jan, Apr, Jul, Oct 2020), every 12h
_year = 2020
_data_freq = 6
def _dt_to_idx(dt):
    return int((dt - datetime(_year, 1, 1)).total_seconds() // (3600 * _data_freq))
indices = []
for _m in (1, 4, 7, 10):
    _start = datetime(_year, _m, 1)
    _end = datetime(_year + (_m == 12), (_m % 12) + 1, 1)
    indices.extend(range(_dt_to_idx(_start), _dt_to_idx(_end), 2))
era5_loader = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False, collate_fn=collate_fn_val)

# Data normalisations
out_transforms = {}
for l in [6, 12, 24]:
    normalize_diff_std = dict(np.load(os.path.join(norm_dir, f"normalize_diff_std_{l}.npz")))
    normalize_diff_std = np.concatenate([normalize_diff_std[v] for v in variables], axis=0)
    out_transforms[l] = transforms.Normalize(np.zeros_like(normalize_diff_std), normalize_diff_std)
quant_model.set_transforms(inp_transform, out_transforms)


def file_to_datetime(filepath, data_freq = 6):
    name = os.path.basename(filepath).split('.')[0]
    year, idx = map(int, name.split('_'))
    return datetime(year, 1, 1) + timedelta(hours=idx * data_freq)

#Loop through
intervals = [6, 12, 24]
os.makedirs(out_dir, exist_ok=True)
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

    lead_times = list(out_data_dic.keys())
    max_lead = max(lead_times)
    preds_by_lead = {lt: [] for lt in lead_times}

    # For each interval, do ONE rollout to max_lead and snapshot at requested leads.
    for inter in intervals:
        step_to_lead = {lt // inter: lt for lt in lead_times}
        interval_tensor = (torch.tensor([inter], device=inp.device, dtype=inp.dtype) / 10.0).repeat(inp.shape[0])
        x = inp
        with torch.no_grad():
            for step in range(1, max_lead // inter + 1):
                pred_diff = quant_model(x, vars, interval_tensor)
                pred_diff = quant_model.replace_constant(pred_diff, vars)
                pred_diff = quant_model.reverse_diff_transform[inter](pred_diff)
                x = quant_model.inp_transform(quant_model.reverse_inp_transform(x) + pred_diff)
                if step in step_to_lead:
                    preds_by_lead[step_to_lead[step]].append(x)

    sample = {}
    for lead_time in lead_times:
        mean_pred = torch.stack(preds_by_lead[lead_time], dim=0).mean(0).detach().cpu().to(torch.float16).numpy()
        gt = out_data_dic[lead_time].detach().cpu().to(torch.float16).numpy()
        sample[lead_time] = {'pred': mean_pred, 'gt': gt}

    out_path = os.path.join(out_dir, f"{init_time:%Y%m%d_%H}.pt")
    torch.save({'init_time': init_time, 'results': sample}, out_path)
    del sample, preds_by_lead, mean_pred, gt

