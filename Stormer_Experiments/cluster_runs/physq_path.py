"""Path shim for the cluster job scripts, which live one level below the experiment root.

Mirrors the root physq_path.py but resolves upward, so the drivers in this directory can
still import the shared core (physq_core/) and the experiment-root modules they depend on
(precision_harness.py, eval_metrics.py, plot_common.py).
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CORE = os.path.abspath(os.environ.get("PHYSQ_CORE") or os.path.join(_ROOT, "physq_core"))
if not os.path.isdir(CORE):
    raise RuntimeError(
        f"shared core not found at {CORE}. Set PHYSQ_CORE to the physq_core/ directory.")

for _p in (CORE, os.path.join(_ROOT, "scripts"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
