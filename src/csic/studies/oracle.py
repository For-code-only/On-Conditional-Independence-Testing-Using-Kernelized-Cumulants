"""Conditional feature means and fixed-reference oracle comparisons."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from numbers import Integral

import numpy as np
from scipy.spatial.distance import cdist

from ..inference import matrix_penalty, median_bandwidth
from ..kernels import _checked_eigh, _result, center, mixture_pvalue
from ..simulation import nuisance


def _seed(seed, *parts):
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    data = json.dumps([int(seed), "revision_oracle_v1", *parts], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(data.encode()).digest()[:8], "little")


def _vector(x, name):
    x = np.asarray(x, dtype=float)
    if x.ndim == 2 and x.shape[1] == 1:
        x = x[:, 0]
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError(f"{name} must be a finite scalar-observation vector")
    return x


def _sample(sample, *, null=True):
    x, y = _vector(sample["x"], "X"), _vector(sample["y"], "Y")
    z = np.asarray(sample["z"], dtype=float)
    if z.ndim == 1:
        z = z[:, None]
    if len(x) < 2 or len(y) != len(x) or z.ndim != 2 or z.shape[0] != len(x) or z.shape[1] < 1 or not np.isfinite(z).all():
        raise ValueError("Expected paired finite X,Y,Z and n>=2")
    metadata = sample.get("metadata", {})
    if null and (metadata.get("case", "C0") != "C0" or float(metadata.get("a", 0.0)) != 0.0):
        raise ValueError("These conditional-mean oracles are defined for C0 only")
    if null and float(metadata.get("tau", metadata.get("nuisance_scale", 1.0))) != 1.0:
        raise ValueError("C0 oracle requires the original unscaled nuisance functions")
    return x, y, z


def _independent(test, other, name):
    if test is other:
        raise ValueError(f"{name} must be independent of the evaluation sample")
    test_meta, other_meta = test.get("metadata", {}), other.get("metadata", {})
    a = test_meta.get("data_seed", test_meta.get("seed"))
    b = other_meta.get("data_seed", other_meta.get("seed"))
    if a is not None and b is not None and a == b:
        raise ValueError(f"{name} reuses the evaluation data seed")


def cross_rbf(x, y, sigma):
    """Rectangular Gaussian Gram, using the paper's standard-deviation width."""
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1] or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Rectangular kernel inputs must be finite with matching dimensions")
    return np.exp(-cdist(x, y, "sqeuclidean") / (2.0 * sigma * sigma))


def gaussian_kernel_mean(m0, v0, m1, v1, sigma):
    """E k(V0,V1), for independent normal variables (zero variance allowed)."""
    sigma = float(sigma)
    if sigma <= 0 or not np.isfinite(sigma):
        raise ValueError("sigma must be finite and positive")
    s2 = sigma * sigma
    if s2 <= 0 or not np.isfinite(s2):
        raise ValueError("sigma must be finite and positive")
    m0, v0, m1, v1 = np.broadcast_arrays(m0, v0, m1, v1)
    if np.any(v0 < 0) or np.any(v1 < 0) or not all(np.isfinite(a).all() for a in (m0, v0, m1, v1)):
        raise ValueError("Normal means/variances must be finite; variances nonnegative")
    total = s2 + v0 + v1
    return np.sqrt(s2 / total) * np.exp(-(m0 - m1) ** 2 / (2 * total))


def gaussian_kernel_product(m0, v0, m1, v1, m2, v2, sigma):
    """E[k(V0,V1) k(V0,V2)] for three independent normal variables.

    The two kernel differences share V0, hence their covariance is v0.
    Treating those two differences as independent is an incorrect oracle.
    """
    arrays = np.broadcast_arrays(m0, v0, m1, v1, m2, v2)
    m0, v0, m1, v1, m2, v2 = arrays
    if not all(np.isfinite(a).all() for a in arrays) or any(np.any(v < 0) for v in (v0, v1, v2)):
        raise ValueError("Normal means/variances must be finite; variances nonnegative")
    sigma = float(sigma)
    if sigma <= 0 or not np.isfinite(sigma):
        raise ValueError("sigma must be finite and positive")
    s2 = sigma * sigma
    if not np.isfinite(s2) or s2 <= 0:
        raise ValueError("sigma must be finite and positive")
    a, b, c = s2 + v0 + v1, s2 + v0 + v2, v0
    determinant = a * b - c * c
    d1, d2 = m0 - m1, m0 - m2
    exponent = (b * d1 * d1 + a * d2 * d2 - 2 * c * d1 * d2) / determinant
    return s2 / np.sqrt(determinant) * np.exp(-0.5 * exponent)


@dataclass(frozen=True)
class GaussianReference:
    """A mean embedding of a weighted mixture of Gaussian distributions."""

    means: np.ndarray
    variances: np.ndarray
    weights: np.ndarray
    label: str = "fixed_reference"
    quadrature_mass: float | None = None

    def __post_init__(self):
        means = _vector(self.means, "reference means").copy()
        variances = _vector(self.variances, "reference variances").copy()
        weights = _vector(self.weights, "reference weights").copy()
        if len(means) == 0 or means.shape != variances.shape or means.shape != weights.shape:
            raise ValueError("Reference arrays must have equal nonempty shape")
        if np.any(variances < 0) or np.any(weights < 0) or not np.isclose(weights.sum(), 1.0, rtol=0, atol=1e-12):
            raise ValueError("Reference variances/weights must be nonnegative and weights sum to one")
        weights /= weights.sum()
        for name, value in (("means", means), ("variances", variances), ("weights", weights)):
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @classmethod
    def empirical(cls, y):
        y = _vector(y, "reference Y")
        if len(y) == 0:
            raise ValueError("Reference sample cannot be empty")
        return cls(y, np.zeros(len(y)), np.full(len(y), 1 / len(y)), "independent_empirical_reference")


class CenteredGaussianKernel:
    """Analytic first- and second-order moments for a fixed centred kernel."""

    def __init__(self, sigma, reference):
        self.sigma, self.reference = float(sigma), reference
        self._second_center_cache = {}
        m, v, w = reference.means, reference.variances, reference.weights
        matrix = gaussian_kernel_mean(m[:, None], v[:, None], m[None, :], v[None, :], sigma)
        self.c0 = float(w @ matrix @ w)

    def mean_center(self, means, variances):
        means = _vector(means, "means")
        variances = np.broadcast_to(np.asarray(variances, dtype=float), means.shape)
        r = self.reference
        return gaussian_kernel_mean(means[:, None], variances[:, None], r.means[None, :], r.variances[None, :], self.sigma) @ r.weights

    def second_center(self, means, variances):
        """E[c(V)^2]; independent reference draws share the same V."""
        means = _vector(means, "means")
        variances = np.broadcast_to(np.asarray(variances, dtype=float), means.shape)
        key = (means.tobytes(), variances.tobytes())
        if key in self._second_center_cache:
            return self._second_center_cache[key].copy()
        a = self.mean_center(means, variances)
        out = a * a
        r = self.reference
        for i in np.flatnonzero(variances):
            products = gaussian_kernel_product(means[i], variances[i], r.means[:, None], r.variances[:, None],
                                               r.means[None, :], r.variances[None, :], self.sigma)
            out[i] = r.weights @ products @ r.weights
        self._second_center_cache[key] = out.copy()
        return out

    def _kernel_times_center(self, m0, v0, m1, v1):
        """E[k(V0,V1)c(V0)], evaluated without an n*n*M allocation."""
        result = np.zeros((len(m0), len(m1)))
        r = self.reference
        for mean, variance, weight in zip(r.means, r.variances, r.weights):
            if weight == 0:
                continue
            result += weight * gaussian_kernel_product(m0[:, None], v0[:, None], m1[None, :], v1[None, :],
                                                        mean, variance, self.sigma)
        return result

    def first_moment(self, left, left_var, right, right_var):
        left, right = _vector(left, "left means"), _vector(right, "right means")
        lv = np.broadcast_to(np.asarray(left_var, dtype=float), left.shape)
        rv = np.broadcast_to(np.asarray(right_var, dtype=float), right.shape)
        k = gaussian_kernel_mean(left[:, None], lv[:, None], right[None, :], rv[None, :], self.sigma)
        return k - self.mean_center(left, lv)[:, None] - self.mean_center(right, rv)[None, :] + self.c0

    def second_moment(self, left, left_var, right, right_var):
        """E[kappa(V,W)^2] for independent conditional marginal draws."""
        left, right = _vector(left, "left means"), _vector(right, "right means")
        lv = np.broadcast_to(np.asarray(left_var, dtype=float), left.shape)
        rv = np.broadcast_to(np.asarray(right_var, dtype=float), right.shape)
        k = gaussian_kernel_mean(left[:, None], lv[:, None], right[None, :], rv[None, :], self.sigma)
        k2 = gaussian_kernel_mean(left[:, None], lv[:, None], right[None, :], rv[None, :], self.sigma / np.sqrt(2.0))
        al, ar = self.mean_center(left, lv), self.mean_center(right, rv)
        bl, br = self.second_center(left, lv), self.second_center(right, rv)
        kl = (al[:, None] * k if np.all(lv == 0)
              else self._kernel_times_center(left, lv, right, rv))
        if np.array_equal(left, right) and np.array_equal(lv, rv):
            kr = kl.T
        else:
            kr = (ar[None, :] * k if np.all(rv == 0)
                  else self._kernel_times_center(right, rv, left, lv).T)
        return (k2 + bl[:, None] + br[None, :] + self.c0 * self.c0
                - 2 * kl - 2 * kr + 2 * self.c0 * k
                + 2 * al[:, None] * ar[None, :] - 2 * self.c0 * (al[:, None] + ar[None, :]))

    def residual_grams(self, y, conditional_means):
        """Target, oracle residual and target/oracle cross Gram for b and q."""
        y, g = _vector(y, "Y"), _vector(conditional_means, "conditional means")
        if y.shape != g.shape:
            raise ValueError("Y and conditional means must have equal shape")
        kb = self.first_moment(y, 0.0, y, 0.0)
        mb = self.first_moment(y, 0.0, g, 1.0)
        nb = self.first_moment(g, 1.0, g, 1.0)
        q = kb * kb
        mq = self.second_moment(y, 0.0, g, 1.0)
        nq = self.second_moment(g, 1.0, g, 1.0)
        return {
            "first": {"target": kb, "oracle": _sym(kb - mb - mb.T + nb), "cross": kb - mb},
            "second": {"target": q, "oracle": _sym(q - mq - mq.T + nq), "cross": q - mq},
        }


def _sym(matrix):
    return (matrix + matrix.T) * 0.5


def _product_reference_grams(x, z, rx, rz, sigma_x):
    """A-target, true A residual, and target/oracle cross Gram."""
    f, _ = nuisance(z)
    d = z.shape[1]
    kz = (1 + cross_rbf(z, z, np.sqrt(d))) / 2
    kzr = (1 + cross_rbf(z, rz, np.sqrt(d))) / 2
    kzrr = (1 + cross_rbf(rz, rz, np.sqrt(d))) / 2
    kx = cross_rbf(x, x, sigma_x)
    cross = gaussian_kernel_mean(x[:, None], 0, f[None, :], 1, sigma_x)
    meanmean = gaussian_kernel_mean(f[:, None], 1, f[None, :], 1, sigma_x)
    c = np.mean(cross_rbf(x, rx, sigma_x) * kzr, axis=1)
    c0 = float(np.mean(cross_rbf(rx, rx, sigma_x) * kzrr))
    expected_c = np.mean(gaussian_kernel_mean(f[:, None], 1, rx[None, :], 0, sigma_x) * kzr, axis=1)
    target = kx * kz - c[:, None] - c[None, :] + c0
    oracle = _sym(kz * (kx - cross - cross.T + meanmean))
    target_oracle = kz * (kx - cross) - c[None, :] + expected_c[None, :]
    return {"target": target, "oracle": oracle, "cross": target_oracle, "KZ": kz, "KX": kx}


def _residual_operator(kz, rho):
    values, vectors = _checked_eigh(kz, vectors=True)
    a = (vectors * (rho / (values + rho))) @ vectors.T
    smooth = values / (values + rho)
    return a, float(smooth @ smooth / len(values))


def evaluate_oracle(sample, *, bandwidth_sample, reference_sample, seed, B=4999,
                    reference_size=256, trace_rtol=1e-8):
    """Compare KRR and exact conditional means with a shared fixed reference center."""
    x, y, z = _sample(sample)
    bx, by, bz = _sample(bandwidth_sample)
    rx, ry, rz = _sample(reference_sample)
    for other, name in ((bandwidth_sample, "bandwidth pilot"), (reference_sample, "reference pilot")):
        _independent(sample, other, name)
    _independent(bandwidth_sample, reference_sample, "reference pilot")
    if bz.shape[1] != z.shape[1] or rz.shape[1] != z.shape[1]:
        raise ValueError("Pilot and evaluation conditioning dimensions must agree")
    if isinstance(reference_size, bool) or not isinstance(reference_size, Integral) or not 2 <= reference_size <= len(rx):
        raise ValueError("reference_size must be an integer between 2 and reference pool size")
    rx, ry, rz = rx[:reference_size], ry[:reference_size], rz[:reference_size]
    sx, sy = median_bandwidth(bx)[0], median_bandwidth(by)[0]
    rho = matrix_penalty(.01, len(x))
    product = _product_reference_grams(x, z, rx, rz, sx)
    a, _ = _residual_operator(product["KZ"], rho)
    _, g = nuisance(z)
    feature = CenteredGaussianKernel(2 * sy, GaussianReference.empirical(ry))
    response = feature.residual_grams(y, g)["second"]
    arms = {"fixed_reference": (_sym(a @ product["target"] @ a), _sym(a @ response["target"] @ a)),
            "oracle_both": (product["oracle"], response["oracle"])}
    scores = {}
    for arm, (left, right) in arms.items():
        gram = _sym(left * right)
        fit = _result(np.sum(gram) / len(x), _checked_eigh(center(gram) / len(x)), trace_rtol)
        rng = np.random.default_rng(_seed(seed, arm, "second_y2"))
        scores[arm] = mixture_pvalue(fit["statistic"], fit["weights"], rng, B=B)["pvalue"]
    return scores
