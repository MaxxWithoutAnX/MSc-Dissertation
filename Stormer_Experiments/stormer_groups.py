""" Breaks Stormer into layer groups
"""
import re
from collections import OrderedDict

EXCLUDE = ("t_embedder",)

BAND_SIZE = 6           # 24 blocks -> 4 depth bands


def group_of(name):
    """Group key for a Linear's module name, or None to leave it in full precision."""
    if any(tok in name for tok in EXCLUDE):
        return None

    # conditioning projections (in-block and head): one group across all depths
    if "adaLN_modulation" in name:
        return "adaln"

    # backbone transformer blocks: [net.]blocks.{b}.<...>
    m = re.search(r"(?:^|\.)blocks\.(\d+)\.", name)
    if m:
        if "attn.qkv" in name:    comp = "qkv"
        elif "attn.proj" in name: comp = "proj"
        elif "mlp.fc1" in name:   comp = "fc1"
        elif "mlp.fc2" in name:   comp = "fc2"
        else:                     comp = None
        if comp is None:
            return None                                   # any other in-block Linear: skip
        b = int(m.group(1))
        band = f"b{(b // BAND_SIZE) * BAND_SIZE}-{(b // BAND_SIZE) * BAND_SIZE + BAND_SIZE - 1}"
        block = "attn" if comp in ("qkv", "proj") else "mlp"
        return f"{band}_{block}"

    # prediction head
    if "head.linear" in name:
        return "head"
    # variable-aggregation cross-attention out-projection
    if "channel_agg" in name:
        return "embed"
    return "other"


def build_groups(model):
    """OrderedDict{group_name: [module_name, ...]} over the model's quantisable Linears,
    plus a parallel dict of per-group parameter counts (weights only)."""
    import torch.nn as nn
    groups, params = OrderedDict(), {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            g = group_of(name)
            if g is None:
                continue
            groups.setdefault(g, []).append(name)
            params[g] = params.get(g, 0) + mod.weight.numel()
    return groups, params


def build_groups_from_names(names):
    """Same grouping from a bare list of module names (for testing without loading the model)."""
    groups = OrderedDict()
    for name in names:
        g = group_of(name)
        if g is None:
            continue
        groups.setdefault(g, []).append(name)
    return groups


if __name__ == "__main__":
    # Self-test on a representative slice of the Stormer module-name dump (no model load).
    sample = [
        "net.t_embedder.mlp",                              # excluded (interval conditioning)
        "net.blocks.0.adaLN_modulation.1",                 # adaln
        "net.blocks.23.adaLN_modulation.1",                # adaln
        "net.head.adaLN_modulation.1",                     # adaln
        "net.blocks.0.attn.qkv",
        "net.blocks.0.attn.proj",
        "net.blocks.5.mlp.fc1",
        "net.blocks.6.mlp.fc2",
        "net.blocks.11.attn.qkv",
        "net.blocks.12.mlp.fc1",
        "net.blocks.17.attn.proj",
        "net.blocks.18.mlp.fc2",
        "net.blocks.23.attn.qkv",
        "net.head.linear",
        "net.embedding.channel_agg.out_proj",
    ]
    g = build_groups_from_names(sample)
    print(f"=== {len(g)} groups ===")
    for k, v in g.items():
        print(f"  {k:16} {len(v)} layers")
