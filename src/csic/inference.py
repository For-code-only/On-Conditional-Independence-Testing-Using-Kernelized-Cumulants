"""Single-sample CSIC and baseline tests with the paper's default settings."""

from __future__ import annotations

from numbers import Integral

import numpy as np
from scipy.spatial.distance import pdist

from .kernels import conditional, mixture_pvalue, rbf, unconditional


METHODS = ("HSIC", "CSIC", "KCI", "CSIC_CI")


def _positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _sample_size(n):
    if isinstance(n, bool) or not isinstance(n, Integral) or n < 2:
        raise ValueError("n must be an integer >= 2")
    return int(n)


def _calibration_seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return int(seed)


def median_bandwidth(values):
    """Off-diagonal Euclidean median; include tied pairs, fallback to 1."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or len(values) < 2 or values.shape[1] < 1:
        raise ValueError("Observations must be an n-by-p array with n>=2,p>=1")
    if not np.all(np.isfinite(values)):
        raise ValueError("Observations must be finite")
    median = float(np.median(pdist(values, metric="euclidean")))
    return (median if median > 0 else 1.0), median == 0


def matrix_penalty(rho400, n):
    """rho_n=rho400*log(n)/log(400); population lambda_n=rho_n/n."""
    n = _sample_size(n)
    return _positive(rho400, "rho400") * np.log(n) / np.log(400)


def build_matrices(x, y, z=None, *, method, sigma_x_multiplier=1.0,
                   sigma_y_multiplier=None, sigma_z_multiplier=1.0):
    """Build Gaussian Gram matrices from paired observations.
    
    X uses its median distance; Y defaults to twice its median for CSIC/CSIC_CI
    and its median for HSIC/KCI. Z uses (1+RBF)/2 with width sqrt(D).
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    conditional_method = method in ("KCI", "CSIC_CI")
    if conditional_method != (z is not None):
        raise ValueError("KCI/CSIC_CI require Z; HSIC/CSIC require Z=None")
    if len(x) != len(y) or (z is not None and len(x) != len(z)):
        raise ValueError("X, Y, and Z must have matching sample sizes")
    sx_base, x_fallback = median_bandwidth(x)
    sy_base, y_fallback = median_bandwidth(y)
    mx = _positive(sigma_x_multiplier, "sigma_x_multiplier")
    if sigma_y_multiplier is None:
        sigma_y_multiplier = 2.0 if method in ("CSIC", "CSIC_CI") else 1.0
    my = _positive(sigma_y_multiplier, "sigma_y_multiplier")
    mz = _positive(sigma_z_multiplier, "sigma_z_multiplier")
    sx, sy = sx_base * mx, sy_base * my
    kz, sz = None, None
    if z is not None:
        z = np.asarray(z, dtype=float)
        if z.ndim == 1:
            z = z[:, None]
        if z.ndim != 2 or z.shape[1] == 0:
            raise ValueError("Z must have at least one feature")
        sz = float(np.sqrt(z.shape[1]) * mz)
        kz = (1.0 + rbf(z, sigma=sz)) / 2.0
    return {"KX": rbf(x, sigma=sx), "KY": rbf(y, sigma=sy), "KZ": kz,
            "metadata": {"sigma_x": sx, "sigma_y": sy, "sigma_z": sz,
                         "sigma_x_multiplier": mx, "sigma_y_multiplier": my,
                         "sigma_z_multiplier": mz, "median_x": sx_base,
                         "median_y": sy_base, "median_zero_fallback_x": x_fallback,
                         "median_zero_fallback_y": y_fallback,
                         "rbf_definition": "exp(-distance_squared/(2*sigma**2))",
                         "z_kernel": "(1+RBF)/2" if z is not None else None,
                         "sample_standardization": False}}


def evaluate_sample(sample, method, *, seed, rho400=None, B=4999, alpha=0.05,
                    trace_rtol=1e-8, sigma_x_multiplier=1.0,
                    sigma_y_multiplier=None, sigma_z_multiplier=1.0):
    """Evaluate a sample dictionary containing x, y and optional z.
    
    Conditional rho400 defaults to 0.01; unconditional methods omit it.
    The calibration seed is supplied separately from the data seed.
    """
    seed = _calibration_seed(seed)
    matrices = build_matrices(sample["x"], sample["y"], sample.get("z"), method=method,
                              sigma_x_multiplier=sigma_x_multiplier,
                              sigma_y_multiplier=sigma_y_multiplier,
                              sigma_z_multiplier=sigma_z_multiplier)
    n = len(sample["x"])
    if matrices["KZ"] is None:
        if rho400 is not None:
            raise ValueError("An unconditional test has no ridge parameter")
        results = unconditional(matrices["KX"], matrices["KY"], trace_rtol)
        rho = population_lambda = None
    else:
        if rho400 is None:
            rho400 = 0.01
        rho = matrix_penalty(rho400, n)
        population_lambda = rho / n
        results = conditional(matrices["KX"], matrices["KY"], matrices["KZ"],
                              rho, trace_rtol)
    estimate = results[method]
    calibration = mixture_pvalue(estimate["statistic"], estimate["weights"],
                                 np.random.default_rng(seed), B=B, alpha=alpha)
    return {"method": method, "estimate": estimate, "calibration": calibration,
            "metadata": {**matrices["metadata"], "n": n, "rho400": rho400,
                         "ridge_matrix": rho, "lambda_population": population_lambda,
                         "calibration_seed": int(seed), "B_requested": int(B),
                         "trace_rtol": float(trace_rtol),
                         "implementation": ("matched custom KCI reimplementation"
                                            if method == "KCI" else "exact Gram implementation"),
                         "source_sample": dict(sample.get("metadata", {}))}}


def test(x, y, z=None, *, method="CSIC", seed, rho400=None, B=4999,
         alpha=0.05, trace_rtol=1e-8, sigma_y_multiplier=None):
    """Test paired observations with a specified calibration seed.
    
    x, y and optional z are arrays with observations in rows. KCI/CSIC_CI
    require z; HSIC/CSIC omit it. Y bandwidth defaults to twice the median
    distance for CSIC/CSIC_CI and to the median for HSIC/KCI. X uses its median.
    
    For conditional methods, rho_n=rho400*log(n)/log(400), with rho400=0.01
    by default and population penalty lambda_n=rho_n/n.
    
    Returns the statistic n*V_n, retained weights, calibration and parameters.
    Calibration uses B spectral draws, or the exact distribution for an empty
    or equal-weight spectrum.
    """
    return evaluate_sample(
        {"x": x, "y": y, "z": z}, method, seed=seed, rho400=rho400, B=B,
        alpha=alpha, trace_rtol=trace_rtol,
        sigma_y_multiplier=sigma_y_multiplier)
