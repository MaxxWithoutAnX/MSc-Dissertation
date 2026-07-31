import os
import numpy as np
import torch


class NoiseFloor:
    def __init__(self, detailed=None):
        # detailed: {floor_type: {"per_family": {family: {lead: value}}}} or None
        self._detailed = detailed
        self.is_null = detailed is None

    @classmethod
    def null(cls):
        return cls(None)

    @classmethod
    def from_detailed(cls, path=os.path.join("validation", "noise_floor_detailed.pt")):
        if not os.path.exists(path):
            return cls.null()
        return cls(torch.load(path, map_location="cpu", weights_only=False))

    def sigma(self, family_key, lead):
        """Conservative (max over floor types) run-to-run floor for a family at a
        lead; 0.0 when unknown so an unmatched metric is never gated."""
        if self._detailed is None:
            return 0.0
        vals = []
        for ftype in self._detailed.values():
            fam = ftype.get("per_family", {}).get(family_key, {})
            if lead in fam:
                vals.append(float(fam[lead]))
        return max(vals) if vals else 0.0


def floor_family_of(spec):
    label = spec.label
    if label.startswith("RMSE "):           return "w_rmse"
    if label.startswith("SpecDiv"):         return "spec_div"
    if label.startswith("SpecRes"):         return "spec_res"
    if label.startswith("Vag/Vg"):          return "wbal_ageo_geo"
    if label.startswith("HypsRel"):         return "hyps_rms"
    if label.startswith("|DryAir"):         return "dryair_Md_err"
    if label.startswith("neg-q fraction"):  return "negq_frac"
    return None


def distortion(delta, iqr, sigma):
    if not np.isfinite(iqr) or iqr <= 0:
        return 0.0
    return float(max(0.0, abs(delta) - max(0.0, sigma)) / iqr)
