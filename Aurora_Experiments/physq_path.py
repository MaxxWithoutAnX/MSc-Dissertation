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
