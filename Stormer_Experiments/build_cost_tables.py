""" Sane as build_cost_tables.py in Aurora_Experiments but built for Stormer
"""
import os

import torch
import torch.nn as nn

import stormer.models.hub.stormer as stormer_hub
from stormer_groups import build_groups

IMG_SIZE = [128, 256]
PATCH_SIZE = 2
HIDDEN_SIZE = 1024
DEPTH = 24
NUM_HEADS = 16
MLP_RATIO = 4
N_VARS = 69

EXPECTED_TOTAL_PARAMS = 456_413_184

CKPT = os.environ.get(
    "STORMER_CKPT",
    "/vol/bitbucket/mes25/stormer_checkpoints/stormer_1.40625_patch_size_2.ckpt")

OUT_PATH = os.environ.get(
    "STORMER_COST_TABLE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "cost_tables.pt"))


def _mea_stub(query, key, value, attn_bias=None, p=0.0, **kwargs):
    return value.new_zeros((*query.shape[:-1], value.shape[-1]))


def capture_linear_tokens(model, variables):
    """One instrumented forward; accumulate each nn.Linear's input token count over all
    of its calls. Returns {module_name: tokens}."""
    tokens = {}

    def mk(name):
        def hook(_mod, inp):
            x = inp[0]
            tokens[name] = tokens.get(name, 0) + int(x.numel() // x.shape[-1])
        return hook

    handles = [m.register_forward_pre_hook(mk(n))
               for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    real_attn = stormer_hub.memory_efficient_attention
    stormer_hub.memory_efficient_attention = _mea_stub
    try:
        x = torch.randn(1, len(variables), IMG_SIZE[0], IMG_SIZE[1])
        with torch.no_grad():
            model(x, variables, torch.tensor([6.0]))      # output discarded on purpose
    finally:
        stormer_hub.memory_efficient_attention = real_attn
        for h in handles:
            h.remove()
    return tokens


def _multihead_attention_tokens(name, tokens):
    block = {t for n, t in tokens.items()
             if (n.startswith("blocks.0.") or ".blocks.0." in n) and t > 1}
    if len(block) != 1:
        raise RuntimeError(
            f"cannot infer token count for {name!r}: expected exactly one distinct "
            f"multi-token count among blocks.0's Linears, got {sorted(block)}")
    return next(iter(block))


def _load_checkpoint(net):
    if not os.path.exists(CKPT):
        print(f"checkpoint not found at {CKPT}; params and FLOPs are "
              f"architecture-determined so cost_tables.pt is unaffected", flush=True)
        return
    try:
        sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"checkpoint at {CKPT} could not be deserialised "
              f"({type(e).__name__}: {str(e).splitlines()[0]}); skipping the weight-level "
              f"cross-check. Params and FLOPs are architecture-determined and are still "
              f"guarded by EXPECTED_TOTAL_PARAMS.", flush=True)
        return
    sd = sd.get("state_dict", sd)
    stripped = {k.split("net.", 1)[1]: v for k, v in sd.items() if k.startswith("net.")}
    missing, _unexpected = net.load_state_dict(stripped or sd, strict=False)
    bad = [k for k in missing
           if "adaLN" in k or k.startswith("blocks.") or k.startswith("head.")]
    if bad:
        raise RuntimeError(
            f"checkpoint {CKPT} does not match the architecture built here "
            f"({len(bad)} core weights missing, e.g. {bad[:3]}). The cost table would "
            f"describe a different model than the OAT measured -- check PATCH_SIZE.")
    print(f"loaded checkpoint {CKPT}", flush=True)


def build_table():
    """Return {group: {'params': int, 'flops': float}} for the 11 quantisable
    groups. Raises if any group would carry zero FLOPs."""
    variables = [f"v{i}" for i in range(N_VARS)]
    net = stormer_hub.Stormer(
        in_img_size=IMG_SIZE, variables=variables, patch_size=PATCH_SIZE,
        hidden_size=HIDDEN_SIZE, depth=DEPTH, num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO).eval()
    _load_checkpoint(net)

    groups, params = build_groups(net)
    total = sum(params.values())
    if total != EXPECTED_TOTAL_PARAMS:
        raise RuntimeError(
            f"quantisable params = {total:,}, expected {EXPECTED_TOTAL_PARAMS:,}. The "
            f"architecture built here is not the one the OAT measured -- check PATCH_SIZE "
            f"(457,261,056 would mean patch_size=4) and N_VARS.")
    name2group = {n: g for g, names in groups.items() for n in names}
    tokens = capture_linear_tokens(net, variables)
    mods = dict(net.named_modules())

    flops = {g: 0.0 for g in groups}
    for name, g in name2group.items():
        t = tokens.get(name)
        if t is None:
            if not name.endswith("out_proj"):
                raise RuntimeError(
                    f"Linear {name!r} was never called during the shape-capture forward "
                    f"and is not a MultiheadAttention out_proj; its group would be costed "
                    f"at zero FLOPs. Investigate before trusting cost_tables.pt.")
            t = _multihead_attention_tokens(name, tokens)
        out_f, in_f = mods[name].weight.shape
        flops[g] += 2.0 * in_f * out_f * t

    zero = [g for g, f in flops.items() if f <= 0.0]
    if zero:
        raise RuntimeError(f"groups with zero FLOPs: {zero}")
    return {g: {"params": int(params[g]), "flops": float(flops[g])} for g in groups}


def main():
    table = build_table()
    torch.save(table, OUT_PATH)
    tot_p = sum(v["params"] for v in table.values())
    tot_f = sum(v["flops"] for v in table.values())
    print(f"{'group':16}{'params':>14}{'param%':>8}{'flops':>12}{'flop%':>8}", flush=True)
    for g, v in sorted(table.items(), key=lambda kv: -kv[1]["flops"]):
        print(f"{g:16}{v['params']:>14,}{100*v['params']/tot_p:>7.1f}%"
              f"{v['flops']:>12.3e}{100*v['flops']/tot_f:>7.2f}%", flush=True)
    print(f"wrote {OUT_PATH} ({len(table)} groups; {tot_p:,} params; {tot_f:.3e} FLOPs)",
          flush=True)


if __name__ == "__main__":
    main()
