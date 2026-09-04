""" Share analysis core path as physq_core only lives in Aurora_Experiments. Import at the beginning.
"""
import os
import sys

_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "physq_core")

CORE = os.path.abspath(os.environ.get("PHYSQ_CORE") or _DEFAULT)

if not os.path.isdir(CORE):
    raise RuntimeError(
        f"shared core not found at {CORE}. Set PHYSQ_CORE to the physq_core/ directory, "
        f"or leave it unset to use {_DEFAULT}.")
if CORE not in sys.path:
    sys.path.insert(0, CORE)

_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
if os.path.isdir(_SCRIPTS) and _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

# The cluster job drivers live in cluster_runs/; physq_core modules import them
# by name (e.g. rescore_with_floor -> run_precision_harness).
_CLUSTER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cluster_runs")
if os.path.isdir(_CLUSTER) and _CLUSTER not in sys.path:
    sys.path.insert(0, _CLUSTER)

# The download/regrid scripts live in data_prep/ (mirrors Stormer_Experiments/data_prep).
_DP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_prep")
if os.path.isdir(_DP) and _DP not in sys.path:
    sys.path.insert(0, _DP)
