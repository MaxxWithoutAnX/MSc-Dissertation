"""The rows that fig09/fig10 (and figA32/figA33) plot, and that the ablation rank, share"""
from collections import namedtuple

HEADLINE_LEVELS = (50, 250, 500, 700, 850)
BALANCE_LEVELS = (50, 250, 500, 850)          # no moisture level needed for wind balance

UPPER = (("Z", "geopotential"), ("T", "temperature"), ("Q", "specific_humidity"),
         ("U", "u_component_of_wind"), ("V", "v_component_of_wind"))
SURFACE = (("T2M", "2m_temperature"), ("U10", "10m_u_component_of_wind"),
           ("V10", "10m_v_component_of_wind"), ("MSLP", "mean_sea_level_pressure"))
UPPER_BLOCK = {"Z": "GEOPOTENTIAL", "T": "TEMPERATURE", "Q": "HUMIDITY",
               "U": "U-WIND", "V": "V-WIND"}

#
Row = namedtuple("Row", "label block kind group key agg")


def _raw(label, block, group, key, agg="mean"):
    return Row(label, block, "raw", group, key, agg)


def _spec(label, block, agg="mean"):
    return Row(label, block, "spec", None, None, agg)


def build_cards():
    """{card name: [Row, ...]} in draw order."""
    rmse = []
    for short, var in UPPER:
        for L in HEADLINE_LEVELS:
            rmse.append(_raw(f"{short} {L}", UPPER_BLOCK[short], "RMSE",
                             f"w_rmse_{var}_{L}_{{lt}}", agg="rms"))
    for short, var in SURFACE:
        rmse.append(_raw(short, "SURFACE", "RMSE", f"w_rmse_{var}_{{lt}}", agg="rms"))

    phys = [_spec(f"RMSE {v}", "RMSE", agg="rms") for v in ("Z500", "T850", "Q700")]
    for lbl, group, fmt in (("|Vag|", "wind_balance", "wbal_vag_pred_{L}_{{lt}}"),
                            ("Vag/Vg", "wind_balance", "wbal_ageo_geo_pred_{L}_{{lt}}")):
        for L in BALANCE_LEVELS:
            phys.append(_raw(f"{lbl} {L}hPa", "BALANCE", group, fmt.format(L=L)))
    phys.append(_spec("HypsRel 600-500", "BALANCE"))
    for m in ("|DryAir Md err|", "neg-q fraction", "|neg-q mass|"):
        phys.append(_spec(m, "CONSERVATION"))
    for fam in ("SpecDivW1", "SpecResLog"):
        for v in ("Z500", "T850", "Q700"):
            phys.append(_spec(f"{fam} {v}", "SPECTRAL"))
    for v in ("Z500", "T850", "Q700"):
        phys.append(_spec(f"|RQE| {v}", "EXTREMES"))
    return {"rmse": rmse, "physics": phys}


CARDS = build_cards()
