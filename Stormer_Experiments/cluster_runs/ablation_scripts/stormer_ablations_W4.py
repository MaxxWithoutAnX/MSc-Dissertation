print('hi', flush=True)
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from stormer.data.iterative_dataset import ERA5MultiLeadtimeDataset
from stormer.models.iterative_module import GlobalForecastIterativeModule
from stormer.models.hub.stormer import Stormer
from stormer.data.multi_step_datamodule import collate_fn_val
from torch.utils.data import DataLoader, Subset
from datetime import datetime, timedelta
from stormer_groups import build_groups
from eval_metrics import compute_all_metrics, difference_kinetic_energy
from torchao import quantize_
from torchao.quantization.quant_api import UIntXWeightOnlyConfig

CKPT = '/vol/bitbucket/mes25/stormer_checkpoints/stormer_1.40625_patch_size_2.ckpt'

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

# Normalisation transforms (needed by fresh_model, so built before it)
root_dir = '/vol/bitbucket/mes25/wb2_h5df'
norm_dir = '/vol/bitbucket/mes25/stormer/normalization_constants'
normalize_mean = dict(np.load(os.path.join(norm_dir, "normalize_mean.npz")))
normalize_mean = np.concatenate([normalize_mean[v] for v in variables], axis=0)
normalize_std = dict(np.load(os.path.join(norm_dir, "normalize_std.npz")))
normalize_std = np.concatenate([normalize_std[v] for v in variables], axis=0)
inp_transform = transforms.Normalize(normalize_mean, normalize_std)
out_transforms = {}
for l in [6, 12, 24]:
    normalize_diff_std = dict(np.load(os.path.join(norm_dir, f"normalize_diff_std_{l}.npz")))
    normalize_diff_std = np.concatenate([normalize_diff_std[v] for v in variables], axis=0)
    out_transforms[l] = transforms.Normalize(np.zeros_like(normalize_diff_std), normalize_diff_std)

std_reverse = 1.0 / normalize_std
mean_reverse = -normalize_mean * std_reverse
reverse_inp_transform = transforms.Normalize(mean_reverse, std_reverse)

_CACHED_SD = None

def fresh_model():
    global _CACHED_SD
    net = Stormer(
        in_img_size=[128, 256],
        variables=variables,
        patch_size=2,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
    )
    if _CACHED_SD is None:
        m = GlobalForecastIterativeModule(net, pretrained_path=CKPT)   # full load, once
        _CACHED_SD = {k: v.cpu() for k, v in m.state_dict().items()}
    else:
        m = GlobalForecastIterativeModule(net)                         # cheap reuse
        m.load_state_dict(_CACHED_SD)
    m.set_transforms(inp_transform, out_transforms)
    print('fresh model', flush=True)
    return m.eval().to(device)


model = fresh_model()

groups, params = build_groups(model)
config = UIntXWeightOnlyConfig(dtype=torch.uint4, group_size=64)   # W4 (uint4 weight-only, g64), as in stormer_inference_fullyear_W4
print(f'total number of groups: {len(groups)}')

dataset = ERA5MultiLeadtimeDataset(
    root_dir=os.path.join(root_dir, "test"),
    variables=variables,
    transform=inp_transform,
    list_lead_times=[24, 72, 120, 168],  # 1, 3, 5, 7 days
    data_freq=6,
)
N_INITS = int(os.environ.get("ABLATION_N_INITS", "48"))   # 0 -> use all 12h-stride inits
indices = list(range(0, len(dataset), 2))
if N_INITS and len(indices) > N_INITS:
    indices = indices[:: max(1, len(indices) // N_INITS)][:N_INITS]
era5_loader = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False, collate_fn=collate_fn_val)
print(f"{len(indices)} inits, leads [24, 72, 120, 168]", flush=True)

out_dir = f"ablation_W4_medium_fullyear_{len(indices)}i"
os.makedirs(out_dir, exist_ok=True)

lat = np.load(os.path.join(root_dir, "lat.npy"))

WIND_VARS = [v for v in variables if v.startswith(("u_component_of_wind_", "v_component_of_wind_"))]
WIND_IDX = [variables.index(v) for v in WIND_VARS]

intervals = [6, 12, 24]


def file_to_datetime(filepath, data_freq=6):
    name = os.path.basename(filepath).split('.')[0]
    year, idx = map(int, name.split('_'))
    return datetime(year, 1, 1) + timedelta(hours=idx * data_freq)


def metrics_over_inits(m, ref_preds=None, keep_preds=False):
    out = {}
    preds = {} if keep_preds else None
    for i, (inp, out_data_dic, vars_b) in enumerate(era5_loader):
        subset_idx = era5_loader.dataset.indices[i]
        filepath = era5_loader.dataset.dataset.inp_file_paths[subset_idx]
        init_time = file_to_datetime(filepath)
        key = f"{init_time:%Y%m%d_%H}"
        inp = inp.to(device)

        lead_times = list(out_data_dic.keys())
        max_lead = max(lead_times)
        preds_by_lead = {lt: [] for lt in lead_times}

        for inter in intervals:
            step_to_lead = {lt // inter: lt for lt in lead_times}
            interval_tensor = (torch.tensor([inter], device=inp.device, dtype=inp.dtype) / 10.0).repeat(inp.shape[0])
            x = inp
            with torch.inference_mode():
                for step in range(1, max_lead // inter + 1):
                    pred_diff = m(x, vars_b, interval_tensor)
                    pred_diff = m.replace_constant(pred_diff, vars_b)
                    pred_diff = m.reverse_diff_transform[inter](pred_diff)
                    x = m.inp_transform(m.reverse_inp_transform(x) + pred_diff)
                    if step in step_to_lead:
                        preds_by_lead[step_to_lead[step]].append(x)

        out[key] = {}
        if keep_preds:
            preds[key] = {}
        for lead in lead_times:
            mean_pred = torch.stack(preds_by_lead[lead], dim=0).mean(0).detach().cpu().float()
            gt = out_data_dic[lead].detach().cpu().float()
            pred_phys = reverse_inp_transform(mean_pred)
            gt_phys = reverse_inp_transform(gt)
            metrics = compute_all_metrics(pred_phys, gt_phys, None, variables, lat, lead)
            metrics.pop('bias')
            metrics["dke"] = difference_kinetic_energy(pred_phys, gt_phys, variables, lat, str(lead))
            wind = pred_phys[:, WIND_IDX]                              # [1, 26, H, W]
            if ref_preds is not None:
                metrics["dke_pert"] = difference_kinetic_energy(
                    wind, ref_preds[key][lead], WIND_VARS, lat, str(lead))
            if keep_preds:
                preds[key][lead] = wind
            out[key][lead] = metrics
        del preds_by_lead
        print(f"  scored init {key} at leads {sorted(lead_times)}", flush=True)
    return out, preds


fp32_path = os.path.join(out_dir, "fp32_metrics.pt")
fp_preds_path = os.path.join(out_dir, "fp32_wind_preds.pt")   # reference for dke_pert
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
    torch.cuda.empty_cache()
    torch.save(results, oat_path)                                   # checkpoint after each group
    print(f"saved {group} -> {oat_path} ({len(results)}/{len(groups)} groups)", flush=True)
