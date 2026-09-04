""" Checks model parameter count, weight size, and peak VRAM per quantisation scheme
"""
import torch
import io
import gc
import time
import numpy as np
import os
import xarray as xr
from aurora import AuroraPretrained, Batch, Metadata, rollout
from torchao import quantize_
from torchao.quantization.quant_api import Int8WeightOnlyConfig, Int8DynamicActivationInt8WeightConfig, UIntXWeightOnlyConfig

import torch._dynamo
import torch._inductor.config as _inductor_config
torch._dynamo.config.cache_size_limit = 64
for _obj in (getattr(_inductor_config, "triton", None), _inductor_config):
    if _obj is not None and hasattr(_obj, "assert_indirect_indexing"):
        _obj.assert_indirect_indexing = False


print("imported scripts", flush=True)

def import_model():
    CKPT = "/vol/bitbucket/mes25/aurora_checkpoints/aurora-0.25-pretrained.ckpt"
    model = AuroraPretrained(autocast=True)
    model.load_checkpoint_local(CKPT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    return model

def clear_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return None

def check_size(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    print(f"{buf.getbuffer().nbytes/1e9:.1f} GB", flush=True)
    buf.close()
    del buf
    gc.collect()

def check_VRAM_model():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    print(f"weights resident: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

def check_VRAM_peak():
    torch.cuda.synchronize()
    print(f"peak allocated: {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)
    print(f"peak reserved : {torch.cuda.max_memory_reserved()/1e9:.2f} GB", flush=True)


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


def timed_rollout(model, label="rollout time"):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in rollout(model, make_batch(2), steps=2):
            pass
    torch.cuda.synchronize()
    print(f"{label}: {time.perf_counter() - t0:.2f} s", flush=True)


print("FP32", flush=True)
model = import_model()
check_size(model)
check_VRAM_model()
timed_rollout(model)
check_VRAM_peak()


print("W8", flush=True)
model = clear_model(model)
model = import_model()
EXCLUDE = ("decoder.surf_heads", "decoder.atmos_heads")
def _is_linear(m, fqn):
    return(
        isinstance(m, torch.nn.Linear)
        and m.weight.dim() == 2
        and m.weight.shape[1] > 1
        and not any(s in fqn for s in EXCLUDE)
    )
quantize_(model, Int8WeightOnlyConfig(), filter_fn=_is_linear)
print('quantised model!')
check_size(model)
check_VRAM_model()
timed_rollout(model)
check_VRAM_peak()


print("W8A8 (torch.compile)", flush=True)
model = clear_model(model)
model = import_model()
quantize_(model, Int8DynamicActivationInt8WeightConfig(), filter_fn=_is_linear)
print('quantised model!')
check_size(model)                 # state_dict size is compile-independent
model.backbone = torch.compile(model.backbone)
model.encoder.level_agg = torch.compile(model.encoder.level_agg)
model.decoder.level_decoder = torch.compile(model.decoder.level_decoder)
if hasattr(model.decoder, "level_decoder_alternate"):  # only set when separate_perceiver is used
    model.decoder.level_decoder_alternate = torch.compile(model.decoder.level_decoder_alternate)
print("compiling (warmup rollout)...", flush=True)
timed_rollout(model, "compile+warmup time")
check_VRAM_model()                          # reset peak stats now that kernels are compiled
timed_rollout(model, "rollout time (warm)")
check_VRAM_peak()                           # clean steady-state peak

print("W4", flush=True)
model = clear_model(model)
model = import_model()
quantize_(model, UIntXWeightOnlyConfig(dtype=torch.uint4, group_size=64), filter_fn=_is_linear)
check_size(model)
check_VRAM_model()
timed_rollout(model)
check_VRAM_peak()