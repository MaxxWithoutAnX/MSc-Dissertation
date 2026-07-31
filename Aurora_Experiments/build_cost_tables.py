import datetime
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ablation_scripts"))
from groups import build_groups
from aurora import AuroraPretrained, Batch, Metadata

LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
H, W = 721, 1440

_REAL_SDPA = F.scaled_dot_product_attention


def _sdpa_stub(query, key, value, *args, **kwargs):
    """Shape-preserving attention: out = (*query.shape[:-1], value.shape[-1]). Values are
    irrelevant to Linear input shapes, so this changes no computed FLOP and only saves
    memory during the shape-capture pass."""
    return query.new_zeros((*query.shape[:-1], value.shape[-1]))


def _synth_batch(dtype=torch.float32, device="cpu"):
    surf = {k: torch.randn(1, 2, H, W, dtype=dtype, device=device) for k in ("2t", "10u", "10v", "msl")}
    static = {k: torch.randn(H, W, dtype=dtype, device=device) for k in ("lsm", "z", "slt")}
    atmos = {k: torch.randn(1, 2, len(LEVELS), H, W, dtype=dtype, device=device)
             for k in ("t", "u", "v", "q", "z")}
    return Batch(surf_vars=surf, static_vars=static, atmos_vars=atmos,
                 metadata=Metadata(  # the model re-casts metadata to its own dtype and re-validates;
                                   lat=torch.linspace(90, -90, H, device=device),
                                   lon=torch.linspace(0, 350, W, device=device),
                                   time=(datetime.datetime(2020, 1, 1, 0),),
                                   atmos_levels=LEVELS))


def capture_linear_tokens(model, dtype, device):
    """One forward; accumulate each nn.Linear's input token count across all its calls."""
    tokens = {}

    def mk(name):
        def hook(_mod, inp):
            x = inp[0]
            tokens[name] = tokens.get(name, 0) + int(x.numel() // x.shape[-1])
        return hook

    handles = [m.register_forward_pre_hook(mk(n))
               for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    import importlib
    orig_post_init = Metadata.__post_init__
    guard_patches = []
    for modname in ("encoder", "decoder"):
        mod = importlib.import_module(f"aurora.model.{modname}")
        if hasattr(mod, "check_lat_lon_dtype"):
            guard_patches.append((mod, mod.check_lat_lon_dtype))
            mod.check_lat_lon_dtype = lambda *a, **k: None
    try:
        Metadata.__post_init__ = lambda self: None
        F.scaled_dot_product_attention = _sdpa_stub
        with torch.no_grad():
            model(_synth_batch(dtype, device))
    finally:
        Metadata.__post_init__ = orig_post_init
        F.scaled_dot_product_attention = _REAL_SDPA
        for mod, fn in guard_patches:
            mod.check_lat_lon_dtype = fn
        for h in handles:
            h.remove()
    return tokens


def main():
    model = AuroraPretrained(autocast=False).eval()      # full architecture
    try:
        model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt")
        print("loaded full 0.25 pretrained checkpoint", flush=True)
    except Exception as e:
        print(f"checkpoint load skipped ({type(e).__name__}: {e}); param COUNTS and FLOPs "
              f"are architecture-determined, so cost_tables.pt is unaffected", flush=True)

    groups, params = build_groups(model)   # {group: [names]}, {group: numel}
    name2group = {n: g for g, names in groups.items() for n in names}

    device, dtype = "cpu", torch.float32
    print(f"forward @ {H}x{W}, {len(LEVELS)} levels on {device}/{dtype} "
          f"(shape capture only; may spill to swap)...", flush=True)
    tokens = capture_linear_tokens(model.to(device=device, dtype=dtype), dtype, device)

    mods = dict(model.named_modules())
    flops = {g: 0.0 for g in groups}
    missing = []
    for name, g in name2group.items():
        t = tokens.get(name)
        if t is None:
            missing.append(name)
            continue
        out_f, in_f = mods[name].weight.shape
        flops[g] += 2.0 * in_f * out_f * t

    table = {g: {"params": int(params[g]), "flops": float(flops[g])} for g in groups}
    torch.save(table, "cost_tables.pt")
    tot_p = sum(v["params"] for v in table.values())
    tot_f = sum(v["flops"] for v in table.values())
    print(f"\n{'group':16} {'params':>13} {'param%':>7} {'flops':>11} {'flop%':>7}", flush=True)
    for g, v in sorted(table.items(), key=lambda kv: -kv[1]["flops"]):
        print(f"{g:16} {v['params']:>13,} {100*v['params']/tot_p:>6.1f}% "
              f"{v['flops']:>11.3e} {100*v['flops']/tot_f:>6.1f}%", flush=True)
    print(f"wrote cost_tables.pt ({len(table)} groups; total {tot_f:.3e} FLOPs)", flush=True)


if __name__ == "__main__":
    main()
