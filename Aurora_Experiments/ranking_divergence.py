# ranking_divergence.py
import physq_path

import numpy as np

from allocator import load_distortion_table
from frontiers import CONSISTENCY, B2_VARIANTS


def sensitivities(scheme_csvs, scheme, lead=120, reducer=None, families=None):
    d_phys = load_distortion_table(scheme_csvs, lead=lead, axes=CONSISTENCY,
                                   reducer=reducer, families=families)
    d_rmse = load_distortion_table(scheme_csvs, lead=lead, axes=("standard",))
    groups = sorted(set(d_phys) & set(d_rmse))
    phys = {g: d_phys[g][scheme]["balance"] + d_phys[g][scheme]["conservation"]
            for g in groups}
    rmse = {g: d_rmse[g][scheme]["standard"] for g in groups}
    return phys, rmse


def _rankdata(a):
    """1-based ranks with ties averaged (a self-contained scipy.stats.rankdata)."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    raw = np.empty(len(a), dtype=float)
    raw[order] = np.arange(1, len(a) + 1)
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, raw)                 # sum of raw ranks per distinct value
    return sums[inv] / counts[inv]            # -> mean rank for each tie group


def _pearson(x, y):
    x = np.asarray(x, float) - np.mean(x)
    y = np.asarray(y, float) - np.mean(y)
    denom = np.sqrt(float((x * x).sum()) * float((y * y).sum()))
    return float((x * y).sum() / denom) if denom > 0 else 0.0


def spearman_rho(phys, rmse):
    """Spearman rho between the two per-group orderings (Pearson on ranks). Near +1 =
    the guides agree (RMSE-blindness harmless); near 0 / negative = they disagree."""
    groups = list(phys)
    return _pearson(_rankdata([rmse[g] for g in groups]),
                    _rankdata([phys[g] for g in groups]))


def permutation_pvalue(phys, rmse, n=10000, seed=0):
    groups = list(phys)
    rx = _rankdata([rmse[g] for g in groups])
    ry = _rankdata([phys[g] for g in groups])
    obs = abs(_pearson(rx, ry))
    rng = np.random.default_rng(seed)
    ge = sum(abs(_pearson(rx, rng.permutation(ry))) >= obs - 1e-12 for _ in range(n))
    return float((ge + 1) / (n + 1))


def rank_table(phys, rmse):
    """Per-group rows for the CSV / slopegraph: rank 1 = most sensitive on each axis;
    rank_delta = rmse_rank - physics_rank (large positive = physics cares far more
    than RMSE = an RMSE-blind-dangerous layer)."""
    groups = list(phys)
    # rank so that the MOST sensitive group is rank 1 (descending): rank on -value.
    pr = dict(zip(groups, _rankdata([-phys[g] for g in groups])))
    rr = dict(zip(groups, _rankdata([-rmse[g] for g in groups])))
    return [{"group": g, "physics": phys[g], "rmse": rmse[g],
             "rank_physics": pr[g], "rank_rmse": rr[g],
             "rank_delta": rr[g] - pr[g]} for g in groups]


def rho_matrix(scheme_csvs_by_scheme, lead=120, variants=None):
    """{scheme: {variant_name: rho}} across the B2 reducer/family variants -- the
    robustness panel. `scheme_csvs_by_scheme`: {scheme: {scheme: csv_path}} so each
    scheme loads from its own sensitivity.csv."""
    variants = variants or B2_VARIANTS
    out = {}
    for scheme, scheme_csvs in scheme_csvs_by_scheme.items():
        out[scheme] = {name: spearman_rho(*sensitivities(scheme_csvs, scheme,
                                                          lead=lead, **kw))
                       for name, kw in variants.items()}
    return out
