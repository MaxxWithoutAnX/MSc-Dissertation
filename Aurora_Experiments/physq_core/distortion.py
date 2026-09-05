""" Distortion metric and noise floor functions/class. Quantisation change is only credited if damage is above
the noise floor from the 15 member ensemble run.
"""
import os
import numpy as np
import torch


class NoiseFloor:
    def __init__(self, detailed=None):
        """
        Numerical noise floor class.
        Args:
            detailed [dict | None]: {floor_type: {"per_family": {family: {lead: value}},
                                     "per_metric": {metric label: {lead: value}},
                                     "n_members": int, "quantile": float}} or None.
                                     Values are in each metric's own units. sigma_for reads per_metric.
        """
        self._detailed = detailed
        self.is_null = detailed is None

    @classmethod
    def null(cls):
        """ Null floor. Used when using a noise floor would be circular (ie building ensemble floor) and when 
            no noise floor exists, code can still run.
        """
        return cls(None)

    @classmethod
    def from_detailed(cls, path="noise_floor_detailed.pt"):
        if not os.path.exists(path):
            return cls.null()
        return cls(torch.load(path, map_location="cpu", weights_only=False))

    def sigma_for(self, spec, lead):
        """per-metric run-to-run floor. 0.0 when that metric has no measured floor, so it is never gated.
        Args:
            spec [MetricSpec]: Metric to look up.
            lead [int] : A single lead time in hours. Should have a value 24, 72, 120, or 168 to match the rest of the code
        Returns:
            float : run to run floor in that metric's own units.
        """
        if self._detailed is None:
            return 0.0
        vals = []
        for ftype in self._detailed.values():
            per_metric = ftype.get("per_metric", {}).get(spec.label, {})
            if lead in per_metric:
                vals.append(float(per_metric[lead]))
        return max(vals) if vals else 0.0


def floor_family_of(spec):
    """ Matches metric to a string for identification
    Args: 
        spec [MetricSpec] : Metric to look up
    Returns:
        str | None : Key of metric or None as a fallback if no match
    """
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
    """ Distortion calculation function
    Args:
        delta [float]   : Difference between quantised and full precision run
        iqr [float]     : Full precision run spread across initialisations
        sigma [float]   : Measured noise floor
    Returns:
        float : Calculated distortion
    """
    if not np.isfinite(iqr) or iqr <= 0:
        return 0.0
    return float(max(0.0, abs(delta) - max(0.0, sigma)) / iqr)
