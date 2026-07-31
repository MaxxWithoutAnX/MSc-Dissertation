# equivalence.py
import numpy as np


def relative_bound(q, f, idx_matrix, margin_pct=5.0):
    q = np.asarray(q, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    base = float(f.mean())
    d = q - f
    if base == 0.0:
        return {"rel_pct": float("nan"), "ci_lo_pct": float("nan"),
                "ci_hi_pct": float("nan"), "bound_pct": float("inf"),
                "within_margin": False}
    boot = np.array([100.0 * float(d[i].mean()) / base for i in idx_matrix])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"rel_pct": 100.0 * float(d.mean()) / base,
            "ci_lo_pct": float(lo), "ci_hi_pct": float(hi),
            "bound_pct": float(max(abs(lo), abs(hi))),
            "within_margin": bool(abs(lo) < margin_pct and abs(hi) < margin_pct)}


def _pearson(x, y):
    x = np.asarray(x, dtype=np.float64) - np.mean(x)
    y = np.asarray(y, dtype=np.float64) - np.mean(y)
    den = np.sqrt(float((x * x).sum()) * float((y * y).sum()))
    return float((x * y).sum() / den) if den > 0 else 0.0


def correlation_bound(x, y, idx_matrix):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    boot = np.array([_pearson(x[i], y[i]) for i in idx_matrix])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"rho": _pearson(x, y), "ci_lo": float(lo), "ci_hi": float(hi),
            "abs_bound": float(max(abs(lo), abs(hi)))}
