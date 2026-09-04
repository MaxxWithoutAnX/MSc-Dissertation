"""Function for the predicted cost calcualtions for allocator
"""
# bf16 is actually FP32. Key is wrong but left to ensure nothing breaks.
STORAGE_BITS = {"bf16": 32, "W8": 8, "W8A8": 8, "W4": 4}

def weight_bits(params, precision):
    """Total weight bits for a group at a precision (storage / resident-VRAM axis).
    Args:
        params [int | float]: quantisable parameter count for the group, from
            cost_tables.pt's ct[group]["params"]
        precision [str]: STORAGE_BITS 
    Returns:
        float: params * bits-per-weight.

    """
    return params * STORAGE_BITS[precision]


def int8_flop_fraction(config, flops):
    """Fraction of total FLOPs still executed at int8 (Frontier B compute proxy):
    groups NOT protected to bf16 run int8. Lower = more protection = higher cost.
    Args:
        config [dict]   : {group: precision}
        flops [dict]    : {group: FLOP count}
    Returns:
        float: int8 FLOPs / total FLOPs"""
    total = sum(flops.values())
    if total <= 0:
        return 0.0
    int8 = sum(flops[g] for g, p in config.items() if p not in ("bf16",))
    return float(int8 / total)
