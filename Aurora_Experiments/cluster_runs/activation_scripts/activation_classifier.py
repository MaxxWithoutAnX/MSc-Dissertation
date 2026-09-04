""" Summarises per layer activation statistics during a rollout
"""
import os, datetime
import torch
import numpy as np
import xarray as xr
from aurora import AuroraPretrained, Batch, Metadata, rollout

CKPT = "/vol/bitbucket/mes25/aurora_checkpoints/aurora-0.25-pretrained.ckpt"
model = AuroraPretrained(autocast=True)
model.load_checkpoint_local(CKPT)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.eval().to(device)


def _is_linear(m, fqn):
    return (
        isinstance(m, torch.nn.Linear)
        and m.weight.dim() == 2
        and m.weight.shape[1] > 1
    )


class ChannelStats:
    """Online per-channel (last-dim) stats over all tokens seen by a layer."""
    def __init__(self, num_features):
        self.absmax = torch.zeros(num_features)          # per-channel max|x|
        self.sumabs = torch.zeros(num_features, dtype=torch.float64)
        self.sumsq  = torch.zeros(num_features, dtype=torch.float64)
        self.count  = 0                                  # tokens (rows)
        self.gmax   = 0.0                                # global max|x| (tensor-wide)
        self.sum4   = 0.0                                # for kurtosis
        self.sum2   = 0.0

    @torch.no_grad()
    def update(self, x):
        x = x.reshape(-1, x.shape[-1]).float().cpu()     # (N, C)
        self.absmax = torch.maximum(self.absmax, x.abs().amax(0))
        self.sumabs += x.abs().sum(0).double()
        self.sumsq  += (x * x).sum(0).double()
        self.count  += x.shape[0]
        self.gmax    = max(self.gmax, float(x.abs().max()))
        self.sum2   += float((x * x).sum())
        self.sum4   += float((x ** 4).sum())

    def summary(self):
        ch_max = self.absmax
        med = float(ch_max.median())
        n = self.count * ch_max.numel()
        var = self.sum2 / n
        kurt = (self.sum4 / n) / (var ** 2 + 1e-12)      # raw kurtosis (Gaussian=3)
        return {
            "C": ch_max.numel(),
            "tensor_max": self.gmax,
            "outlier_ratio": self.gmax / (med + 1e-12),  # SmoothQuant signal
            "kurtosis": kurt,
            "frac_outlier_ch": float((ch_max > 6 * med).float().mean()),
            "top_channels": torch.topk(ch_max, k=min(5, ch_max.numel())).indices.tolist(),
        }


stats = {}
handles = []

def make_hook(name):
    def hook(module, inputs):                            # forward_pre_hook: inputs is a tuple
        x = inputs[0]
        if name not in stats:
            stats[name] = ChannelStats(x.shape[-1])
        stats[name].update(x)
    return hook

for name, module in model.named_modules():
    if _is_linear(module, name):
        handles.append(module.register_forward_pre_hook(make_hook(name)))

print(f"hooked {len(handles)} Linear layers", flush=True)

# --- one forward pass over a real batch ---------------------------------------
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                    "era5_2020-01-01_2020-02-07_aurora_0p25.nc")
ds = xr.open_dataset(DATA)
levels = tuple(int(l) for l in ds.level.values)
lat = torch.from_numpy(ds.latitude.values)
lon = torch.from_numpy(ds.longitude.values)
SURF   = {"2t":"2m_temperature","10u":"10m_u_component_of_wind","10v":"10m_v_component_of_wind","msl":"mean_sea_level_pressure"}
ATMOS  = {"t":"temperature","u":"u_component_of_wind","v":"v_component_of_wind","q":"specific_humidity","z":"geopotential"}
STATIC = {"lsm":"land_sea_mask","z":"geopotential_at_surface","slt":"soil_type"}

def make_batch(i):
    history = lambda name: torch.from_numpy(ds[name].isel(time=[i-1, i]).values[None])
    return Batch(
        surf_vars={k: history(v) for k, v in SURF.items()},
        static_vars={k: torch.from_numpy(ds[v].values) for k, v in STATIC.items()},
        atmos_vars={k: history(v) for k, v in ATMOS.items()},
        metadata=Metadata(lat=lat, lon=lon,
                          time=(ds.time.values[i].astype("datetime64[s]").tolist(),),
                          atmos_levels=levels),
    )

STEPS = 28

with torch.inference_mode():
    for k, _ in enumerate(rollout(model, make_batch(2), steps=STEPS)):
        pass   # hooks fire on every Linear during each step

for h in handles:
    h.remove()

# --- rank layers by outlier severity ------------------------------------------
rows = [(n, s.summary()) for n, s in stats.items()]
rows.sort(key=lambda r: r[1]["outlier_ratio"], reverse=True)

print(f"\n{'layer':60s} {'C':>5s} {'out_ratio':>10s} {'kurtosis':>9s} {'frac_out':>8s}")
for name, s in rows:
    print(f"{name:60s} {s['C']:5d} {s['outlier_ratio']:10.1f} "
          f"{s['kurtosis']:9.1f} {s['frac_outlier_ch']:8.3f}")

torch.save({n: s.summary() for n, s in stats.items()},
           os.path.join(os.path.dirname(os.path.abspath(__file__)), f"activation_stats_{STEPS}.pt"))
print(f"\nsaved activation_stats_{STEPS}.pt", flush=True)
