import io
import gc
import os
import time
import numpy as np
import torch
from torchvision.transforms import transforms
from stormer.models.iterative_module import GlobalForecastIterativeModule
from stormer.models.hub.stormer import Stormer
from torchao import quantize_
from torchao.quantization.quant_api import (
    Int8WeightOnlyConfig,
    Int8DynamicActivationInt8WeightConfig,
    UIntXWeightOnlyConfig,
)

import torch._dynamo
import torch._inductor.config as _inductor_config
torch._dynamo.config.cache_size_limit = 64
for _obj in (getattr(_inductor_config, "triton", None), _inductor_config):
    if _obj is not None and hasattr(_obj, "assert_indirect_indexing"):
        _obj.assert_indirect_indexing = False

print("imported scripts", flush=True)

PRETRAINED = "/vol/bitbucket/mes25/stormer_checkpoints/stormer_1.40625_patch_size_2.ckpt"
NORM_DIR = "/vol/bitbucket/mes25/stormer/normalization_constants"
IMG_SIZE = [128, 256]
ROLLOUT_INTERVAL = 6
ROLLOUT_STEPS = 4

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
print(f"device: {device}", flush=True)


# --- normalisation / data plumbing (matches stormer_inference_fullyear_W8A8.py) ---
normalize_mean = dict(np.load(os.path.join(NORM_DIR, "normalize_mean.npz")))
normalize_mean = np.concatenate([normalize_mean[v] for v in variables], axis=0)
normalize_std = dict(np.load(os.path.join(NORM_DIR, "normalize_std.npz")))
normalize_std = np.concatenate([normalize_std[v] for v in variables], axis=0)
inp_transform = transforms.Normalize(normalize_mean, normalize_std)

out_transforms = {}
for l in [6, 12, 24]:
    normalize_diff_std = dict(np.load(os.path.join(NORM_DIR, f"normalize_diff_std_{l}.npz")))
    normalize_diff_std = np.concatenate([normalize_diff_std[v] for v in variables], axis=0)
    out_transforms[l] = transforms.Normalize(np.zeros_like(normalize_diff_std), normalize_diff_std)

inp = torch.randn(1, len(variables), IMG_SIZE[0], IMG_SIZE[1], device=device)
vars = variables
print("synthetic input ready!", flush=True)


def import_model():
    net = Stormer(
        in_img_size=[128, 256],
        variables=variables,
        patch_size=2,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
    )
    model = GlobalForecastIterativeModule(net, pretrained_path=PRETRAINED).to(device)
    model.eval()
    model.set_transforms(inp_transform, out_transforms)
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


def rollout(model, label="rollout time"):
    # short rollout at one interval, mirroring the inner loop of the inference scripts.
    inter = ROLLOUT_INTERVAL
    interval_tensor = (torch.tensor([inter], device=inp.device, dtype=inp.dtype) / 10.0).repeat(inp.shape[0])
    x = inp
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(ROLLOUT_STEPS):
            pred_diff = model(x, vars, interval_tensor)
            pred_diff = model.replace_constant(pred_diff, vars)
            pred_diff = model.reverse_diff_transform[inter](pred_diff)
            x = model.inp_transform(model.reverse_inp_transform(x) + pred_diff)
    torch.cuda.synchronize()
    print(f"{label}: {time.perf_counter() - t0:.2f} s", flush=True)


def _is_linear(m, fqn):
    return (
        isinstance(m, torch.nn.Linear)
        and m.weight.dim() == 2
        and m.weight.shape[1] > 1
    )


print("FP32", flush=True)
model = import_model()
check_size(model)
check_VRAM_model()
rollout(model)
check_VRAM_peak()


print("W8", flush=True)
model = clear_model(model)
model = import_model()
quantize_(model, Int8WeightOnlyConfig(), filter_fn=_is_linear)
print("quantised model!", flush=True)
check_size(model)
check_VRAM_model()
rollout(model)
check_VRAM_peak()


print("W8A8 (torch.compile)", flush=True)
model = clear_model(model)
model = import_model()
quantize_(model, Int8DynamicActivationInt8WeightConfig(), filter_fn=_is_linear)
print("quantised model!", flush=True)
check_size(model)                          # state_dict size is compile-independent
model.net = torch.compile(model.net)
print("compiling (warmup rollout)...", flush=True)
rollout(model, "compile+warmup time")
check_VRAM_model()                          # reset peak stats now that kernels are compiled
rollout(model, "rollout time (warm)")
check_VRAM_peak()                           # clean steady-state peak

print("W4", flush=True)
model = clear_model(model)
model = import_model()
quantize_(model, UIntXWeightOnlyConfig(dtype=torch.uint4, group_size=64), filter_fn=_is_linear)
print("quantised model!", flush=True)
check_size(model)
check_VRAM_model()
rollout(model)
check_VRAM_peak()
