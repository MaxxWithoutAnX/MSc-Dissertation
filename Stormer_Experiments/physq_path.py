import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT = os.path.abspath(os.path.join(_HERE, os.pardir, "Aurora_Experiments", "physq_core"))

CORE = os.path.abspath(os.environ.get("PHYSQ_CORE") or _DEFAULT)

if not os.path.isdir(CORE):
    raise RuntimeError(
        f"shared core not found at {CORE}. Set PHYSQ_CORE to the physq_core/ directory, "
        f"or place Aurora_Experiments/ as a sibling checkout next to Stormer/.")
if CORE not in sys.path:
    sys.path.insert(0, CORE)

_CORE_PARENT = os.path.abspath(os.path.join(CORE, os.pardir))
if os.path.isfile(os.path.join(_CORE_PARENT, "run_label_spread.py")) \
        and _CORE_PARENT not in sys.path:
    sys.path.append(_CORE_PARENT)

# The cluster job drivers live in cluster_runs/ (both here and under Aurora_Experiments/);
# physq_core modules import some of them by name.
for _cr in (os.path.join(_HERE, "cluster_runs"),
            os.path.join(_CORE_PARENT, "cluster_runs")):
    if os.path.isdir(_cr) and _cr not in sys.path:
        sys.path.insert(0, _cr)

# The download/regrid/HDF5 scripts live in data_prep/. The harness and the noise-floor
# ensemble import prepare_2021_h5 (MODEL_VARIABLES) and download_wb2_sampled (select) by name.
_DP = os.path.join(_HERE, "data_prep")
if os.path.isdir(_DP) and _DP not in sys.path:
    sys.path.insert(0, _DP)
