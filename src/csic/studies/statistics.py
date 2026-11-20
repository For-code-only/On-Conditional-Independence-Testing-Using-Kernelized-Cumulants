"""Null-rank calibration, pointwise intervals and paired comparisons."""
from __future__ import annotations

import hashlib
import math
from numbers import Integral

import numpy as np
from scipy.stats import binom, binomtest, norm


CONFIG_ORDER = ("first_y1", "second_y1", "first_y2", "second_y2")


def _probability(value, name, *, open_interval=False):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a numeric probability")
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric probability") from error
    valid = 0 < value < 1 if open_interval else 0 <= value <= 1
    if not math.isfinite(value) or not valid:
        raise ValueError(f"{name} must be {'in (0, 1)' if open_interval else 'in [0, 1]'}")
    return value


def _scores(values, *, allow_empty=False):
    try:
        array = np.asarray(values, dtype=float)
    except (ValueError, TypeError) as error:
        raise ValueError("Scores must be a one-dimensional vector of numeric p-values") from error
    if array.ndim != 1 or (not len(array) and not allow_empty):
        raise ValueError("Scores must be a nonempty one-dimensional vector")
    if not np.isfinite(array).all() or np.any((array < 0) | (array > 1)):
        raise ValueError("Scores must be finite p-values in [0, 1]")
    return array


def _decisions(values, *, columns=None):
    array = np.asarray(values)
    ndim = 1 if columns is None else 2
    if array.ndim != ndim or not len(array):
        raise ValueError("Decisions must have nonempty replication rows")
    if columns is not None and array.shape[1] != columns:
        raise ValueError(f"Expected exactly {columns} decision columns")
    if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
        raise ValueError("Decisions must be booleans or finite binary numbers")
    if not np.isin(array, (0, 1)).all():
        raise ValueError("Decisions must be binary; failures cannot be encoded as decisions")
    return array.astype(bool, copy=False)


def _rank(alpha, size):
    # Match the returned floating p_out <= alpha comparison at representable
    # boundaries (e.g. .29 * 100 can round below 29). Do not add an arbitrary
    # epsilon that can accept a genuinely larger p-value.
    denominator = size + 1
    rank = int(math.floor(alpha * denominator))
    if (rank + 1) / denominator <= alpha:
        rank += 1
    if rank / denominator > alpha:
        rank -= 1
    return rank


def fit_null_calibration(scores, alpha=.05, confidence=.95):
    """Freeze a conservative outer-rank test and quantile stability interval.

    Returned objects are JSON serializable.  ``cutoff`` is an observed score,
    never an interpolated quantile.  Reject when raw_score < cutoff.  If rank
    is zero, no rejection is possible and cutoff is None.

    The stability interval uses one-based ranks l,u with
    P(l <= Bin(M,alpha) <= u-1) >= confidence.  None endpoints denote the
    extended lower/upper endpoints (-infinity/+infinity), respectively.  For
    discrete scores the interval is conservative; it is not a guarantee of
    an exactly 5% rejection probability after conditioning on this bank.
    """
    alpha = _probability(alpha, "alpha", open_interval=True)
    confidence = _probability(confidence, "confidence", open_interval=True)
    bank = np.sort(_scores(scores), kind="stable")
    size = len(bank)
    rank = _rank(alpha, size)
    cutoff = float(bank[rank - 1]) if rank else None
    tail = (1 - confidence) / 2
    lower_rank = int(binom.ppf(tail, size, alpha))
    upper_rank = int(binom.ppf(1 - tail, size, alpha)) + 1
    coverage = float(binom.cdf(upper_rank - 1, size, alpha)
                     - binom.cdf(lower_rank - 1, size, alpha))
    if coverage + 1e-14 < confidence:
        raise ArithmeticError("Binomial order-statistic interval failed its coverage check")
    return {
        "version": 1,
        "score": "original_numeric_pvalue",
        "n_calibration": int(size),
        "alpha": alpha,
        "rank_one_based": rank,
        "cutoff": cutoff,
        "comparison": "strict_less_than" if rank else "never_reject",
        "ties": "conservative",
        "sorted_scores": bank.tolist(),
        "scores_sha256": hashlib.sha256(bank.astype("<f8", copy=False).tobytes()).hexdigest(),
        "cutoff_ties": int(np.count_nonzero(bank == cutoff)) if rank else 0,
        "stability": {
            "confidence": confidence,
            "lower_rank_one_based": lower_rank,
            "upper_rank_one_based": upper_rank,
            "lower_score": float(bank[lower_rank - 1]) if lower_rank >= 1 else None,
            "upper_score": float(bank[upper_rank - 1]) if upper_rank <= size else None,
            "binomial_coverage": coverage,
            "estimand": "population_score_quantile_stability_not_frozen_test_rate_interval",
        },
    }


def _calibration_bank(calibration):
    if not isinstance(calibration, dict):
        raise ValueError("Calibration must be a frozen calibration object")
    bank = _scores(calibration.get("sorted_scores"))
    if np.any(bank[:-1] > bank[1:]):
        raise ValueError("Frozen calibration bank is not sorted")
    size = calibration.get("n_calibration")
    if isinstance(size, bool) or not isinstance(size, Integral) or size != len(bank):
        raise ValueError("Calibration bank size disagrees with metadata")
    alpha = _probability(calibration.get("alpha"), "alpha", open_interval=True)
    rank = _rank(alpha, len(bank))
    if calibration.get("rank_one_based") != rank:
        raise ValueError("Frozen calibration rank disagrees with its rule")
    cutoff = float(bank[rank - 1]) if rank else None
    if calibration.get("cutoff") != cutoff:
        raise ValueError("Frozen calibration cutoff disagrees with its scores")
    digest = hashlib.sha256(bank.astype("<f8", copy=False).tobytes()).hexdigest()
    if calibration.get("scores_sha256") != digest:
        raise ValueError("Frozen calibration score hash mismatch")
    return bank, alpha, rank


def apply_null_calibration(scores, calibration):
    """Return NumPy pvalue/reject arrays; evaluation scores never refit the bank."""
    observed = _scores(scores, allow_empty=True)
    bank, alpha, rank = _calibration_bank(calibration)
    counts = np.searchsorted(bank, observed, side="right")
    pvalues = (1.0 + counts) / (len(bank) + 1)
    # Integer comparison is identical to p_out <= alpha without a floating
    # division comparison at the rank boundary.
    rejected = 1 + counts <= rank
    return {"pvalue": pvalues, "reject": rejected}


def wilson(k, n, confidence=.95):
    """Wilson score interval, with correct endpoints at zero and one."""
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, Integral) for x in (k, n)):
        raise ValueError("Wilson counts must be integers")
    k, n = int(k), int(n)
    if n < 1 or not 0 <= k <= n:
        raise ValueError("Wilson counts require 0 <= k <= n and n > 0")
    confidence = _probability(confidence, "confidence", open_interval=True)
    z = float(norm.ppf((1 + confidence) / 2))
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (0.0 if k == 0 else max(0.0, center - half),
            1.0 if k == n else min(1.0, center + half))


def rate_summary(decisions, confidence=.95):
    decisions = _decisions(decisions)
    n, k = len(decisions), int(decisions.sum())
    p = k / n
    low, high = wilson(k, n, confidence)
    return {"replications": n, "rejections": k, "rate": p,
            "mcse": math.sqrt(p * (1 - p) / n), "ci_low": low,
            "ci_high": high, "ci_method": "wilson", "confidence": float(confidence)}


def paired_summary(first, second, confidence=.95):
    """First-minus-second difference, Newcombe method 10 CI, exact McNemar p."""
    a, b = _decisions(first), _decisions(second)
    if len(a) != len(b):
        raise ValueError("Paired decisions must have identical lengths")
    confidence = _probability(confidence, "confidence", open_interval=True)
    n = len(a)
    e = int(np.count_nonzero(a & b))
    f = int(np.count_nonzero(a & ~b))
    g = int(np.count_nonzero(~a & b))
    h = n - e - f - g
    p1, p2 = (e + f) / n, (e + g) / n
    delta = p1 - p2
    l1, u1 = wilson(e + f, n, confidence)
    l2, u2 = wilson(e + g, n, confidence)
    denominator = math.sqrt(float((e + f) * (g + h) * (e + g) * (f + h)))
    covariance_numerator = e * h - f * g
    if covariance_numerator > 0:
        covariance_numerator = max(covariance_numerator - n / 2, 0.0)
    correlation = covariance_numerator / denominator if denominator else 0.0
    correlation = min(1.0, max(-1.0, correlation))
    dl1, du1, dl2, du2 = p1 - l1, u1 - p1, p2 - l2, u2 - p2
    lower_distance = math.sqrt(max(0.0, dl1 * dl1 - 2 * correlation * dl1 * du2 + du2 * du2))
    upper_distance = math.sqrt(max(0.0, du1 * du1 - 2 * correlation * du1 * dl2 + dl2 * dl2))
    sample_variance = max(0.0, (f + g - n * delta * delta) / (n - 1)) if n > 1 else None
    mcse = math.sqrt(sample_variance / n) if sample_variance is not None else None
    pvalue = float(binomtest(f, f + g, p=.5, alternative="two-sided").pvalue) if f + g else 1.0
    return {
        "replications": n, "both_reject": e, "first_only": f,
        "second_only": g, "neither_reject": h,
        "first_rate": p1, "second_rate": p2,
        "paired_difference": delta, "mcse": mcse,
        "ci_low": max(-1.0, delta - lower_distance),
        "ci_high": min(1.0, delta + upper_distance),
        "ci_method": "newcombe_1998_method_10",
        "confidence": confidence, "corrected_correlation": correlation,
        "discordance": (f + g) / n,
        "pvalue": pvalue, "pvalue_method": "exact_two_sided_mcnemar",
    }


def holm(pvalues):
    """Holm adjusted p-values in the original order; no silent missing tests."""
    p = _scores(pvalues, allow_empty=True)
    if not len(p):
        return []
    order = np.argsort(p, kind="stable")
    adjusted_sorted = np.minimum(1.0, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]))
    adjusted = np.empty_like(p)
    adjusted[order] = adjusted_sorted
    return adjusted.tolist()


def interaction_summary(decisions, confidence=.95):
    """Difference-in-differences using all four paired rejection indicators.

    For large nondegenerate samples use the prespecified normal Monte Carlo
    interval.  For n<30 or zero sample variance use the bounded [-2,2]
    Hoeffding interval/test, avoiding the misleading [0,0] interval.
    """
    a = _decisions(decisions, columns=4)
    confidence = _probability(confidence, "confidence", open_interval=True)
    scores = a.astype(np.int8) @ np.array([1, -1, -1, 1], dtype=np.int8)
    n = len(scores)
    estimate = float(scores.mean())
    variance = float(scores.var(ddof=1)) if n > 1 else None
    mcse = math.sqrt(variance / n) if variance is not None else None
    if n >= 30 and variance > 0:
        half = float(norm.ppf((1 + confidence) / 2)) * mcse
        pvalue = float(2 * norm.sf(abs(estimate) / mcse))
        ci_method, p_method = "paired_normal_mc_interval", "paired_asymptotic_normal"
    else:
        half = math.sqrt(8 * math.log(2 / (1 - confidence)) / n)
        pvalue = min(1.0, 2 * math.exp(-n * estimate * estimate / 8))
        ci_method, p_method = "bounded_hoeffding", "bounded_hoeffding_upper_bound"
    return {
        "replications": n, "interaction": estimate, "mcse": mcse,
        "ci_low": max(-2.0, estimate - half), "ci_high": min(2.0, estimate + half),
        "ci_method": ci_method, "confidence": confidence,
        "pvalue": pvalue, "pvalue_method": p_method,
        "counts": {str(i): int(np.count_nonzero(scores == i)) for i in range(-2, 3)},
        "definition": "(second_y2-first_y2)-(second_y1-first_y1)",
    }


def factorial_summary(decisions, confidence=.95):
    """Full four-cell summary in CONFIG_ORDER; no bandwidth is selected."""
    a = _decisions(decisions, columns=4)
    pairs = {
        "order_y1": (1, 0), "order_y2": (3, 2),
        "bandwidth_first": (2, 0), "bandwidth_second": (3, 1),
        "main_diagonal": (3, 0), "opposite_diagonal": (1, 2),
    }
    return {
        "config_order": list(CONFIG_ORDER),
        "rates": {name: rate_summary(a[:, i], confidence) for i, name in enumerate(CONFIG_ORDER)},
        "contrasts": {
            name: {"first": CONFIG_ORDER[i], "second": CONFIG_ORDER[j],
                   **paired_summary(a[:, i], a[:, j], confidence)}
            for name, (i, j) in pairs.items()
        },
        "interaction": interaction_summary(a, confidence),
    }
