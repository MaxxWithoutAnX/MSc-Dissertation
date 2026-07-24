PRECISION_BITS = {"bf16": 16, "W8": 8, "W8A8": 8, "W4": 4}


def weight_bits(params, precision):
    """Total weight bits for a group at a precision (storage / resident-VRAM axis)."""
    return params * PRECISION_BITS[precision]


def int8_flop_fraction(config, flops):
    """Fraction of total FLOPs still executed at int8 (Frontier B compute proxy):
    groups NOT protected to bf16 run int8. Lower = more protection = higher cost."""
    total = sum(flops.values())
    if total <= 0:
        return 0.0
    int8 = sum(flops[g] for g, p in config.items() if p not in ("bf16",))
    return float(int8 / total)
