import re
from collections import OrderedDict

COND = ("time_mlp", "pos_embed", "scale_embed", "lead_time_embed",
        "absolute_time_embed", "atmos_levels_embed")


def group_of(name):
    """Group key for a Linear's module name, or None to leave it in full precision."""
    if "ln_modulation" in name:
        return "film"
    if any(tok in name for tok in COND):
        return "cond_embed"

    # backbone transformer blocks: backbone.{encoder|decoder}_layers.{s}.blocks.{b}.<...>
    m = re.search(r"backbone\.(encoder|decoder)_layers\.(\d+)\.blocks\.\d+\.", name)
    if m:
        side = "enc" if m.group(1) == "encoder" else "dec"
        stage = m.group(2)
        if "attn.qkv" in name:   comp = "qkv"
        elif "attn.proj" in name: comp = "proj"
        elif "mlp.fc1" in name:   comp = "fc1"
        elif "mlp.fc2" in name:   comp = "fc2"
        else:                     comp = None
        if comp is None:
            return None                                   # any other in-block Linear: skip
        block = "attn" if comp in ("qkv", "proj") else "mlp"
        return f"{side}{stage}_{block}"

    # stage transitions
    if "downsample" in name:
        return "downsample"
    if "upsample" in name:
        return "upsample"

    # input encoder (patch embed / surface MLP / level-aggregation perceiver)
    if name.startswith("encoder."):
        return "encoder_io"
    # output decoder (variable heads + level-decoder perceiver)
    if name.startswith("decoder."):
        return "decoder_heads" if "heads" in name else "decoder_io"
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
    # Self-test on a representative slice of the full-Aurora dump (no model load needed).
    sample = [
        "backbone.encoder_layers.0.blocks.0.norm1.ln_modulation.1",   # film
        "backbone.encoder_layers.1.blocks.3.norm2.ln_modulation.1",   # film
        "backbone.time_mlp.2",                                         # cond_embed
        "encoder.pos_embed",                                           # cond_embed
        "encoder.lead_time_embed",                                     # cond_embed
        "decoder.atmos_levels_embed",                                  # cond_embed
        "backbone.encoder_layers.0.blocks.0.attn.qkv",
        "backbone.encoder_layers.0.blocks.0.attn.proj",
        "backbone.encoder_layers.0.blocks.5.mlp.fc1",
        "backbone.encoder_layers.0.blocks.5.mlp.fc2",
        "backbone.encoder_layers.1.blocks.9.mlp.fc1",
        "backbone.encoder_layers.2.blocks.3.attn.qkv",
        "backbone.decoder_layers.0.blocks.2.mlp.fc1",
        "backbone.decoder_layers.2.blocks.5.attn.proj",
        "backbone.encoder_layers.0.downsample.reduction",
        "backbone.decoder_layers.0.upsample.lin1",
        "encoder.surf_mlp.net.0",
        "encoder.level_agg.layers.0.0.to_q",
        "decoder.atmos_heads.t",
        "decoder.surf_heads.2t",
        "decoder.level_decoder.layers.0.0.to_q",
    ]
    g = build_groups_from_names(sample)
    print(f"=== {len(g)} groups ===")
    for k, v in g.items():
        print(f"  {k:16} {len(v)} layers")
