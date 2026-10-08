"""Gram statistics and spectral calibration.

Statistics are n times the V-statistic; ridge denotes the matrix penalty n*lambda.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from scipy.stats import chi2


def _matrix(K):
    K = np.asarray(K, dtype=float)
    if K.ndim != 2 or K.shape[0] != K.shape[1] or K.shape[0] < 2:
        raise ValueError("Gram matrix must be square with n >= 2")
    if not np.all(np.isfinite(K)):
        raise ValueError("Gram matrix contains non-finite values")
    return (K + K.T) * 0.5


def center(K):
    """Return H K H without constructing the dense centering matrix."""
    K = _matrix(K)
    mean = K.mean(axis=0)
    return (K - mean[:, None] - mean[None, :] + mean.mean())


def _observations(X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2 or not np.all(np.isfinite(X)):
        raise ValueError("Observations must be a finite one- or two-dimensional array")
    return X


def rbf(X, sigma=1.0):
    """exp(-||x-x'||^2 / (2 sigma^2)), without sample standardization."""
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be strictly positive")
    X = _observations(X)
    return np.exp(-cdist(X, X, metric="sqeuclidean") / (2 * sigma**2))


def _checked_eigh(M, vectors=False):
    """Reject material negative eigenvalues and clip numerical negatives only."""
    M = _matrix(M)
    result = eigh(M, eigvals_only=not vectors, check_finite=False, driver="evr")
    values = result[0] if vectors else result
    scale = max(1.0, float(np.max(np.abs(values), initial=0.0)))
    tolerance = 1e-10 * scale
    if values[0] < -tolerance:
        raise ValueError(f"Non-PSD matrix: minimum eigenvalue {values[0]:.6g}, tolerance {tolerance:.6g}")
    values = np.maximum(values, 0.0)
    return (values, result[1]) if vectors else values


def _trim_weights(values, trace_rtol=1e-8):
    if not np.isfinite(trace_rtol) or not 0 <= trace_rtol < 1:
        raise ValueError("trace_rtol must lie in [0, 1)")
    values = np.asarray(values, dtype=float).reshape(-1)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Mixture weights must be finite and nonnegative")
    values = np.sort(values[values > 0])
    total = float(values.sum())
    count = 0
    if trace_rtol and total:
        count = int(np.searchsorted(np.cumsum(values), trace_rtol * total, side="right"))
    dropped = values[:count]
    kept = values[count:][::-1].copy()
    return kept, {
        "dropped_trace": float(dropped.sum()),
        "dropped_trace_ratio": float(dropped.sum()) / total if total else 0.0,
        "dropped_sum_squares": float(dropped @ dropped),
        "spectral_trace": total,
        "spectral_sum_squares": float(values @ values),
        "retained_trace": float(kept.sum()),
        "retained_sum_squares": float(kept @ kept),
        "weights_count": int(kept.size),
        "untrimmed_positive_count": int(values.size),
        "trace_rtol": float(trace_rtol),
    }


def _result(statistic, values, trace_rtol):
    if not np.isfinite(statistic):
        raise ValueError("Non-finite statistic")
    if statistic < -1e-10:
        raise ValueError(f"Negative squared-norm statistic: {statistic}")
    weights, metadata = _trim_weights(values, trace_rtol)
    return {"statistic": max(0.0, float(statistic)), "weights": weights, **metadata}


def unconditional(KX, KY, trace_rtol=1e-8):
    """Matched HSIC and CSIC, with centred quadratic-feature null spectrum."""
    start = perf_counter()
    KXc, KYc = center(KX), center(KY)
    if KXc.shape != KYc.shape:
        raise ValueError("Gram matrices must have the same shape")
    n = KXc.shape[0]
    Q = KYc * KYc
    lx = _checked_eigh(KXc / n)
    ly = _checked_eigh(KYc / n)
    lq = _checked_eigh(center(Q) / n)
    results = {
        "HSIC": _result(np.sum(KXc * KYc) / n, np.multiply.outer(lx, ly), trace_rtol),
        "CSIC": _result(np.sum(KXc * Q) / n, np.multiply.outer(lx, lq), trace_rtol),
    }
    results["seconds"] = perf_counter() - start
    return results


def prepare_conditional(KX, KY, KZ):
    """Share kernels and the exact KZ eigendecomposition across ridge settings."""
    start = perf_counter()
    KX, KY, KZ = map(_matrix, (KX, KY, KZ))
    if KX.shape != KY.shape or KX.shape != KZ.shape:
        raise ValueError("Gram matrices must have the same shape")
    KYc = center(KY)
    d, V = _checked_eigh(KZ, vectors=True)
    return {
        "KAc": center(KX * KZ),
        "KYc": KYc,
        "Q": KYc * KYc,
        "KZ_eigenvalues": d,
        "KZ_eigenvectors": V,
        "n": len(KZ),
        "preparation_seconds": perf_counter() - start,
    }


def conditional_prepared(prepared, ridge, trace_rtol=1e-8, return_matrices=False):
    """Evaluate a previously prepared conditional problem at one matrix ridge."""
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge is the positive MATRIX penalty n * lambda_population")
    start = perf_counter()
    n = prepared["n"]
    d, V = prepared["KZ_eigenvalues"], prepared["KZ_eigenvectors"]
    residual = ridge / (d + ridge)
    smoother = d / (d + ridge)
    A = (V * residual) @ V.T
    # Square KYc before residualizing: (A KYc A)^2 is a different target.
    LA = _matrix(A @ prepared["KAc"] @ A)
    LB = _matrix(A @ prepared["KYc"] @ A)
    LQ = _matrix(A @ prepared["Q"] @ A)
    residual_seconds = perf_counter() - start
    joint_b = _checked_eigh(center(LA * LB) / n)
    joint_q = _checked_eigh(center(LA * LQ) / n)
    result = {
        "KCI": _result(np.sum(LA * LB) / n, joint_b, trace_rtol),
        "CSIC_CI": _result(np.sum(LA * LQ) / n, joint_q, trace_rtol),
        "trace_S2_over_n": float(smoother @ smoother / n),
        "ridge_matrix": float(ridge),
        "lambda_population": float(ridge / n),
        "preparation_seconds": float(prepared["preparation_seconds"]),
        "residual_seconds": residual_seconds,
        "seconds": perf_counter() - start + prepared["preparation_seconds"],
    }
    if return_matrices:
        result["matrices"] = {"A": A, "LA": LA, "LB": LB, "LQ": LQ}
    return result


def conditional(KX, KY, KZ, ridge, trace_rtol=1e-8, return_matrices=False):
    """Matched KCI/CSIC-CI using centred *joint-score* spectra."""
    return conditional_prepared(prepare_conditional(KX, KY, KZ), ridge, trace_rtol, return_matrices)


def _mc_summary(statistic, draws, alpha):
    B = len(draws)
    tail = int(np.count_nonzero(draws >= statistic))
    p = (1 + tail) / (B + 1)
    k = B - int(np.floor(alpha * (B + 1))) + 1
    critical = float(np.partition(draws, k - 1)[k - 1]) if k <= B else float("inf")
    return {
        "pvalue": float(p),
        "critical": critical,
        "reject": bool(p <= alpha),
        "alpha": float(alpha),
        "B": B,
        "tail_count": tail,
        "mc_se": float(np.sqrt(p * (1 - p) / (B + 1))),
        "critical_index_one_based": k,
        "critical_comparison": "strict_greater_than",
        "ties_at_observed": int(np.count_nonzero(draws == statistic)),
    }


def mixture_pvalue(statistic, weights, rng, B=4999, alpha=0.05, block_size=256):
    """Calibrate a right-tail test using weighted chi-square draws.
    
    Uses a fixed B and the plus-one p-value; empty and equal-weight spectra
    use their exact distributions.
    """
    start = perf_counter()
    statistic = float(statistic)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if not np.isfinite(statistic) or not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("Finite statistic and finite nonnegative weights are required")
    if not 0 < alpha < 1 or int(B) != B or B < 1 or int(block_size) != block_size or block_size < 1:
        raise ValueError("Require 0 < alpha < 1 and positive integer B and block_size")
    weights = weights[weights > 0]
    metadata = {"weights_count": int(len(weights)), "alpha": float(alpha), "B_requested": int(B)}
    if not len(weights):
        p, critical = float(statistic <= 0), 0.0
        result = {"pvalue": p, "critical": critical, "reject": bool(p <= alpha),
                  "mc_se": 0.0, "B": 0, "tail_count": None, "calibration": "exact_degenerate"}
    elif np.all(weights == weights[0]):
        # This identity is exact, unlike a two-moment Gamma approximation.
        df, scale = int(len(weights)), float(weights[0])
        p = float(chi2.sf(statistic / scale, df))
        result = {"pvalue": p, "critical": float(scale * chi2.isf(alpha, df)),
                  "reject": bool(p <= alpha), "mc_se": 0.0, "B": 0, "tail_count": None,
                  "calibration": "exact_equal_weights_chi2", "chi2_df": df, "chi2_scale": scale}
    else:
        draws = np.empty(int(B), dtype=float)
        # At most ~16 MiB of Gaussian variates even for an n^2 product spectrum.
        rows = max(1, min(int(block_size), 2_000_000 // max(1, len(weights))))
        for first in range(0, int(B), rows):
            count = min(rows, int(B) - first)
            gaussian = rng.standard_normal((count, len(weights)))
            np.square(gaussian, out=gaussian)
            draws[first:first + count] = np.sum(gaussian * weights, axis=1)
        result = {**_mc_summary(statistic, draws, alpha), "calibration": "spectral_mc_plus_one",
                  "gaussian_block_rows": rows, "simulated_mean": float(draws.mean()),
                  "simulated_variance": float(draws.var(ddof=1)) if B > 1 else 0.0}
    return {**result, **metadata, "seconds": perf_counter() - start}
