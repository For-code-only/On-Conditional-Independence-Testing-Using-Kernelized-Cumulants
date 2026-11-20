"""Stable binary feature representation, retaining the ridge intercept term."""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh
from scipy.spatial.distance import cdist, pdist
from scipy.stats import chi2

CONFIGS = ("first_y1", "second_y2")


def matrix(a):
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not np.isfinite(a).all():
        raise ValueError("Expected a finite square matrix")
    return (a + a.T) / 2


def center_matrix(a):
    a = matrix(a)
    means = a.mean(axis=0)
    return matrix(a - means[:, None] - means[None, :] + means.mean())


def center_features(f):
    # Translation before averaging also gives exactly zero for constant rows.
    f = np.asarray(f, dtype=float)
    shifted = f - f[0]
    return shifted - shifted.mean(axis=0)


def rbf(x, sigma):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if not np.isfinite(x).all() or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Invalid observations or bandwidth")
    return np.exp(-cdist(x, x, "sqeuclidean") / (2 * sigma**2))


def median_bandwidth(x):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("Need at least two finite observations")
    raw = float(np.median(pdist(x)))
    return (raw if raw > 0 else 1.0), raw == 0, raw


def binary_parameters(y):
    y = np.asarray(y, dtype=float).reshape(-1)
    if len(y) < 2 or not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("This implementation requires binary labels encoded 0/1")
    n, k = len(y), int(y.sum())
    p = k / n
    return dict(n=n, k=k, p=p, e=(n*y-k)/n, alpha=(n-2*k)/n,
                beta=k*(n-k)/(n*n), balanced=2*k == n,
                constant=k in (0, n))


def feature_distance(sigma):
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Positive finite Y bandwidth required")
    distance = float(-2 * np.expm1(-1 / (2 * sigma**2)))
    if not np.isfinite(distance) or distance <= 0:
        raise FloatingPointError("Positive binary kernel distance is not representable")
    return distance


def checked_eigh(a, vectors=False):
    a = matrix(a)
    out = eigh(a, eigvals_only=not vectors, driver="evr", check_finite=False)
    raw = out[0] if vectors else out
    tolerance = 1e-10 * max(1., float(np.max(np.abs(raw), initial=0.)))
    if raw[0] < -tolerance:
        raise FloatingPointError(f"Material negative eigenvalue {raw[0]} < {-tolerance}")
    meta = dict(minimum_raw_eigenvalue=float(raw[0]), psd_tolerance=tolerance,
                clipped_negative_mass=float(-raw[raw < 0].sum()),
                clipped_negative_count=int((raw < 0).sum()))
    clipped = np.maximum(raw, 0.)
    return (clipped, out[1], meta) if vectors else (clipped, meta)


def trim_weights(values, trace_rtol=1e-8):
    values = np.asarray(values, dtype=float).reshape(-1)
    if not 0 <= trace_rtol < 1 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Invalid spectrum or trace tolerance")
    values = np.sort(values[values > 0])
    total = float(values.sum())
    count = int(np.searchsorted(np.cumsum(values), trace_rtol*total, side="right")) if total and trace_rtol else 0
    kept, dropped = values[count:][::-1].copy(), values[:count]
    return kept, dict(spectral_trace=total, retained_trace=float(kept.sum()),
                     dropped_trace=float(dropped.sum()),
                     dropped_trace_ratio=float(dropped.sum())/total if total else 0.,
                     spectral_sum_squares=float(values @ values),
                     retained_sum_squares=float(kept @ kept),
                     dropped_sum_squares=float(dropped @ dropped),
                     weights_count=len(kept), untrimmed_positive_count=len(values),
                     trace_rtol=trace_rtol)


def mixture_pvalue(statistic, weights, seed, B=4999, alpha=.05):
    """Same plus-one/exact branch laws as the released implementation."""
    weights = np.asarray(weights, dtype=float)
    if not np.isfinite(statistic) or statistic < 0 or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Finite nonnegative statistic and weights required")
    if not isinstance(B, (int, np.integer)) or B < 1 or not 0 < alpha < 1:
        raise ValueError("Invalid B/alpha")
    weights = weights[weights > 0]
    common = dict(alpha=alpha, B_requested=int(B), weights_count=len(weights))
    if not len(weights):
        p = float(statistic <= 0)
        return dict(**common, pvalue=p, critical=0., reject=p <= alpha, B=0,
                    calibration="exact_degenerate", tail_count=None, mc_se=0.)
    if np.all(weights == weights[0]):
        p = float(chi2.sf(statistic / weights[0], len(weights)))
        return dict(**common, pvalue=p, critical=float(weights[0]*chi2.isf(alpha, len(weights))),
                    reject=p <= alpha, B=0, calibration="exact_equal_weights_chi2",
                    tail_count=None, mc_se=0., chi2_df=len(weights), chi2_scale=float(weights[0]))
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    draws = np.empty(B)
    rows = max(1, min(256, 2_000_000 // len(weights)))
    for first in range(0, B, rows):
        normals = rng.standard_normal((min(rows, B-first), len(weights)))
        draws[first:first+len(normals)] = np.sum(normals**2 * weights, axis=1)
    tail = int((draws >= statistic).sum())
    p = (1+tail)/(B+1)
    k = B-int(np.floor(alpha*(B+1)))+1
    critical = float(np.partition(draws, k-1)[k-1]) if k <= B else float("inf")
    return dict(**common, pvalue=p, critical=critical, reject=p <= alpha, B=B,
                calibration="spectral_mc_plus_one", tail_count=tail,
                mc_se=float(np.sqrt(p*(1-p)/(B+1))),
                critical_index_one_based=k, ties_at_observed=int((draws == statistic).sum()),
                critical_comparison="strict_greater_than", gaussian_block_rows=rows,
                simulated_mean=float(draws.mean()), simulated_variance=float(draws.var(ddof=1)) if B > 1 else 0.)


def score_estimate(scores, seed=0, B=4999, trace_rtol=1e-8):
    """Test-scale squared mean and centered score spectrum, without scale floors."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or len(scores) < 2 or not np.isfinite(scores).all():
        raise ValueError("Invalid score array")
    n = len(scores)
    total = scores.sum(axis=0)
    statistic = float(total @ total / n)
    centered = center_features(scores)
    gram = matrix(centered @ centered.T / n)
    values, psd = checked_eigh(gram)
    weights, spectrum = trim_weights(values, trace_rtol)
    if not len(weights) and statistic > 0:
        # Zero covariance need not imply a zero mean.  Verify this legitimate
        # degenerate case from the score rows instead of zeroing/rejecting it.
        if not np.all(scores == scores[0]):
            raise FloatingPointError("Zero spectrum with nonconstant positive-mean scores requires high-precision verification")
        reference = float(n * (scores[0] @ scores[0]))
        if reference <= 0 or abs(statistic-reference) > 1e-8*reference:
            raise FloatingPointError("Constant-score positive statistic failed its analytic reference")
        spectrum["degenerate_positive_mean_verification"] = "identical_score_rows_and_n_times_row_norm_squared"
    cal = mixture_pvalue(statistic, weights, seed, B)
    return dict(statistic=statistic, weights=weights, untrimmed_weights=values[::-1],
                calibration=cal, spectrum={**spectrum, **psd}, gram=gram)


def _restore(estimate, scale):
    """Calibrate once in normalized units; restore all dimensional quantities."""
    if not np.isfinite(scale) or scale <= 0:
        raise FloatingPointError("Positive calibration scale is not representable")
    for value in (estimate["statistic"], float(estimate["weights"].sum())):
        if value > 0 and (value*scale == 0 or not np.isfinite(value*scale)):
            raise FloatingPointError("Reported physical scale under/overflow")
    spectrum = dict(estimate["spectrum"])
    for key in ("spectral_trace", "retained_trace", "dropped_trace", "minimum_raw_eigenvalue", "clipped_negative_mass"):
        spectrum[key] *= scale
    for key in ("spectral_sum_squares", "retained_sum_squares", "dropped_sum_squares"):
        spectrum[key] *= scale**2
    # PSD tolerance was applied to the normalized matrix and is labeled as such.
    spectrum["normalized_psd_tolerance"] = spectrum.pop("psd_tolerance")
    cal = dict(estimate["calibration"])
    cal["normalized_critical"] = cal["critical"]
    cal["critical"] *= scale
    for key in ("simulated_mean", "chi2_scale"):
        if key in cal:
            cal[key] *= scale
    if "simulated_variance" in cal:
        cal["simulated_variance"] *= scale**2
    return dict(statistic=estimate["statistic"]*scale,
                weights=estimate["weights"]*scale,
                untrimmed_weights=estimate["untrimmed_weights"]*scale,
                calibration=cal, spectrum=spectrum,
                normalized_statistic=estimate["statistic"],
                normalized_weights=estimate["weights"],
                calibration_scale=scale)


def _zero(seed, B, reason):
    values = np.zeros(1)
    kept, metadata = trim_weights(values)
    return dict(statistic=0., weights=kept, untrimmed_weights=values,
                normalized_statistic=0., normalized_weights=kept, calibration_scale=0.,
                calibration=mixture_pvalue(0., kept, seed, B),
                spectrum=metadata, diagnostics={"structural_zero": reason})


def prepare_sample(sample, sigma_x=None, rho=None):
    x, y, z = (np.asarray(sample[k], dtype=float) for k in ("x", "y", "z"))
    if z.ndim != 2 or len(x) != len(y) or len(z) != len(y):
        raise ValueError("Expected paired X/Y/Z with matrix Z")
    bp = binary_parameters(y)
    n = bp["n"]
    mx, fx, raw_x = median_bandwidth(x)
    my, fy, raw_y = median_bandwidth(y)
    sx = mx if sigma_x is None else float(sigma_x)
    sz = float(np.sqrt(z.shape[1]))
    rho = .01*np.log(n)/np.log(400) if rho is None else float(rho)
    if rho <= 0 or not np.isfinite(rho):
        raise ValueError("Invalid matrix penalty")
    kx, kz = rbf(x, sx), .5*(1+rbf(z, sz))
    kac = center_matrix(kx*kz)
    values, vectors, psd = checked_eigh(kac, vectors=True)
    feature = center_features(vectors*np.sqrt(values))
    factor = cho_factor(kz+rho*np.eye(n), lower=True, check_finite=True)
    A = matrix(rho*cho_solve(factor, np.eye(n)))
    residual_feature = A @ feature
    r, c = A @ bp["e"], A @ np.ones(n)
    metadata = dict(n=n, D=z.shape[1], sigma_x=sx, sigma_z=sz,
                    median_x_raw=raw_x, median_y_raw=raw_y, median_y=my,
                    median_zero_fallback_x=fx, median_zero_fallback_y=fy,
                    rho400=.01, ridge_matrix=rho, lambda_population=rho/n,
                    sample_standardization=False, K_A_psd=psd,
                    source_sample=dict(sample.get("metadata", {})))
    return dict(binary=bp, A=A, KX=kx, KZ=kz, KAc=kac, feature=feature,
                U=residual_feature, LA=residual_feature @ residual_feature.T,
                r=r, c=c, metadata=metadata)


def evaluate_prepared(context, config, seed, B=4999, sigma_y=None, trace_rtol=1e-8):
    # Additional config IDs are admitted only for algebraic audit comparisons.
    if config not in ("first_y1", "first_y2", "second_y1", "second_y2"):
        raise ValueError("Unknown configuration")
    bp, U = context["binary"], context["U"]
    second = config.startswith("second")
    sigma_y = context["metadata"]["median_y"]*(2 if config.endswith("2") else 1) if sigma_y is None else float(sigma_y)
    distance = feature_distance(sigma_y)
    r, c, n = context["r"], context["c"], bp["n"]
    response = bp["alpha"]*r + bp["beta"]*c if second else r
    metadata = {**context["metadata"], "config": config, "sigma_y": sigma_y,
                "binary_feature_distance": distance, "calibration_seed": int(seed),
                "B_requested": B, "implementation": "binary_factorized_same_sample_v1"}
    diagnostics = dict(n_one=bp["k"], p_hat=bp["p"], sample_balanced=bp["balanced"],
                       constant_response=bp["constant"], alpha_hat=bp["alpha"], beta_hat=bp["beta"],
                       A1_empirical_norm=float(np.linalg.norm(c)/np.sqrt(n)),
                       Ae_empirical_norm=float(np.linalg.norm(r)/np.sqrt(n)),
                       response_empirical_norm=float(np.linalg.norm(response)/np.sqrt(n)))
    if bp["constant"]:
        result = _zero(seed, B, "constant_binary_response")
        return {**result, "metadata": metadata, "diagnostics": {**diagnostics, **result["diagnostics"]}}
    amplitude = float(np.max(np.abs(response)))
    if amplitude == 0:
        raise FloatingPointError("Unexpected zero binary residual; requires independent verification")
    response_unit = response/amplitude
    normalized = score_estimate(response_unit[:, None]*U, seed, B, trace_rtol)
    scale = (distance**2 if second else distance)*amplitude**2
    result = _restore(normalized, scale)
    if second:
        first_sum = U.T @ (bp["alpha"]*r)
        constant_sum = U.T @ (bp["beta"]*c)
        diagnostics.update(
            proportional_component=float(distance**2*(first_sum @ first_sum)/n),
            cross_component=float(distance**2*2*(first_sum @ constant_sum)/n),
            constant_component=float(distance**2*(constant_sum @ constant_sum)/n))
    diagnostics["zero_spectrum_positive_statistic"] = len(result["weights"]) == 0 and result["statistic"] > 0
    return {**result, "metadata": metadata, "diagnostics": diagnostics}


def evaluate_sample(sample, config, seed, B=4999):
    return evaluate_prepared(prepare_sample(sample), config, seed, B)
