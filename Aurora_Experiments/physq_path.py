import os
import sys

CORE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "physq_core")

if not os.path.isdir(CORE):
    raise RuntimeError(f"shared core not found at {CORE}")
if CORE not in sys.path:
    sys.path.insert(0, CORE)
