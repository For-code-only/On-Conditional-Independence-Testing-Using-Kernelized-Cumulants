"""Matched SplitKCI adaptation of Pogodin et al. (2024), including the full U statistic."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist

from ..inference import median_bandwidth


CONFIGURATIONS = ("first_m", "second_m", "first_2m", "second_2m")
CONFIGS = CONFIGURATIONS
BETA_GRID = (0.1, 0.2, 0.3, 0.4, 0.5)
EPS_GRID = float(np.finfo(np.float32).eps)


def _square(value):
    a = np.asarray(value, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] < 4:
        raise ValueError("Expected a square matrix with at least four rows")
    if not np.isfinite(a).all():
        raise ValueError("Nonfinite kernel")
    if not np.allclose(a, a.T, rtol=1e-12, atol=1e-13):
        raise ValueError("Expected a symmetric kernel")
    return a


def compute_hsic(K, L, biased=False):
    """Reference full U-HSIC, or V-HSIC for explicit acceptance comparisons."""
    K, L = _square(K), _square(L)
    if K.shape != L.shape:
        raise ValueError("Kernel sizes differ")
    if biased:
        kr, lr = K.mean(axis=0), L.mean(axis=0)
        return float(np.mean(K * L) - 2 * np.mean(kr * lr) + kr.mean() * lr.mean())
    k, l = K.copy(), L.copy()
    np.fill_diagonal(k, 0.0)
    np.fill_diagonal(l, 0.0)
    n = len(k)
    kr, lr = k.sum(axis=1), l.sum(axis=1)
    return float((np.sum(k * l) - 2 * (kr @ lr) / (n - 2)
                  + kr.sum() * lr.sum() / ((n - 1) * (n - 2))) / (n * (n - 3)))


def bootstrap_quadratic_matrix(K, L):
    """M such that q'Mq == full U-HSIC(diag(q) K diag(q), L).

    K alone receives reference Rademacher signs.  This exact rearrangement
    avoids allocating bootstrap arrays of shape B by n by n.
    """
    k, l = _square(K).copy(), _square(L).copy()
    if k.shape != l.shape:
        raise ValueError("Kernel sizes differ")
    np.fill_diagonal(k, 0.0)
    np.fill_diagonal(l, 0.0)
    n = len(k)
    lr = l.sum(axis=1)
    m = k * (l - (lr[:, None] + lr[None, :]) / (n - 2)
             + lr.sum() / ((n - 1) * (n - 2)))
    m /= n * (n - 3)
    return m


def _sign_blocks(seed, B, n, block_size, sign_width=None):
    width = n if sign_width is None else int(sign_width)
    if width < n:
        raise ValueError("sign_width cannot be smaller than evaluation size")
    rng = np.random.default_rng(seed)
    for start in range(0, B, block_size):
        count = min(block_size, B - start)
        # One raw uint64 per sign makes the stream invariant to block size.
        # With width=N, nested test prefixes across beta share matching signs.
        raw = rng.bit_generator.random_raw((count, width))
        signs = 2.0 * (raw[:, :n] & np.uint64(1)).astype(np.float64) - 1.0
        yield signs


def wild_bootstrap_many(pairs, *, seed, B=4999, alpha=0.05,
                        block_size=256, sign_width=None):
    """Paired, streaming, plus-one Rademacher bootstrap for named K/L pairs."""
    if int(B) != B or B < 1 or int(block_size) != block_size or block_size < 1:
        raise ValueError("B and block_size must be positive integers")
    if not pairs:
        return {}
    matrices, results = {}, {}
    n = len(next(iter(pairs.values()))[0])
    for name, (K, L) in pairs.items():
        if len(K) != n:
            raise ValueError("Paired bootstrap requires identical evaluation sizes")
        matrices[name] = bootstrap_quadratic_matrix(K, L)
        results[name] = {"statistic": compute_hsic(K, L), "tail_count": 0,
                         "ties_at_observed": 0, "B": int(B)}
    for signs in _sign_blocks(seed, int(B), n, int(block_size), sign_width):
        for name, matrix in matrices.items():
            draws = np.einsum("ij,ij->i", signs @ matrix, signs)
            result = results[name]
            # A constant sign vector leaves K literally unchanged. Preserve
            # that exact tie across differently associated floating-point sums.
            constant = np.all(signs == signs[:, :1], axis=1)
            draws[constant] = result["statistic"]
            result["tail_count"] += int(np.count_nonzero(draws >= result["statistic"]))
            result["ties_at_observed"] += int(np.count_nonzero(draws == result["statistic"]))
    for result in results.values():
        result["pvalue"] = (1 + result["tail_count"]) / (int(B) + 1)
        result["reject"] = bool(result["pvalue"] <= alpha)
        result["alpha"] = float(alpha)
        result["calibration"] = "full_U_rademacher_plus_one_ge"
    return results


def wild_bootstrap(K, L, *, seed, B=4999, alpha=0.05,
                   block_size=256, sign_width=None):
    return wild_bootstrap_many({"test": (K, L)}, seed=seed, B=B, alpha=alpha,
                               block_size=block_size, sign_width=sign_width)["test"]


def split_indices(n, beta, seed):
    """One permutation, test prefix, then two exactly equal training halves."""
    n_test = int(round(float(beta) * int(n)))
    if not np.isclose(n_test, float(beta) * int(n), rtol=0, atol=1e-10):
        raise ValueError("Frozen design requires integral beta*N")
    if n_test < 4 or n_test >= n or (n - n_test) % 2:
        raise ValueError("Test needs >=4 and training must split into two equal halves")
    order = np.random.default_rng(seed).permutation(n)
    half = (n - n_test) // 2
    return order[:n_test], order[n_test:n_test + half], order[n_test + half:]


def rbf_matrix(values, sigma):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    return np.exp(-cdist(x, x, "sqeuclidean") / (2 * float(sigma) ** 2))


def training_center_kernel(K, train):
    """Gram of phi(y)-mean_training(phi), including out-of-training rows."""
    K = np.asarray(K, dtype=np.float64)
    train = np.asarray(train, dtype=int)
    means = K[:, train].mean(axis=1)
    return K - means[:, None] - means[None, :] + K[np.ix_(train, train)].mean()


@dataclass
class RidgeWorkspace:
    """Exact shared half-training spectral decomposition and inverse grid."""
    kernel: np.ndarray

    def __post_init__(self):
        self.kernel = np.asarray(self.kernel, dtype=np.float64)
        self.s = len(self.kernel)
        self.values, self.vectors = eigh(self.kernel, check_finite=False, driver="evr")
        # No clipping: spectral norm is max absolute eigenvalue, even when
        # tiny roundoff-negative eigenvalues appear in the PSD predictor Gram.
        self.delta = float(np.max(np.abs(self.values))) * EPS_GRID
        self.grid = self.delta * 10.0 ** np.arange(1, 8, dtype=np.float64)
        if self.delta <= 0 or not np.isfinite(self.grid).all():
            raise ValueError("Invalid lambda grid")
        self.inverses = []
        self.inverse_diagonals = []
        for lam in self.grid:
            inverse = (self.vectors / (self.values + self.s * lam)) @ self.vectors.T
            self.inverses.append(inverse)
            self.inverse_diagonals.append(np.diag(inverse).copy())

    def fit(self, target_kernel, predictor_test):
        """LOO feature loss; regularizer remains s*lambda in each LOO identity.

        diag(R T R)/diag(R)^2 is algebraically identical to the official
        (Tii+(H T H)ii-2(H T)ii)/(1-Hii)^2, but avoids cancellation.
        """
        target = np.asarray(target_kernel, dtype=np.float64)
        losses = []
        for inverse, diagonal in zip(self.inverses, self.inverse_diagonals):
            numerator = np.einsum("ij,ij->i", inverse @ target, inverse)
            losses.append(float(np.mean(numerator / diagonal**2)))
        if not np.isfinite(losses).all():
            raise FloatingPointError("Nonfinite LOO loss")
        chosen = int(np.argmin(losses))
        weights = self.inverses[chosen] @ predictor_test
        metadata = {"training_size": self.s, "delta": self.delta,
                    "grid_epsilon_dtype": "float32", "grid_epsilon": EPS_GRID,
                    "lambda_grid": self.grid.tolist(), "loo_losses": losses,
                    "selected_index": chosen, "selected_exponent": chosen + 1,
                    "lambda": float(self.grid[chosen]),
                    "matrix_penalty": float(self.s * self.grid[chosen]),
                    "lower_boundary": chosen == 0, "upper_boundary": chosen == 6}
        return weights, metadata


def cross_residual_kernel(target, test, one, two, weights_one, weights_two):
    """Symmetric reference split-regression kernel, without PSD projection."""
    test, one, two = map(np.asarray, (test, one, two))
    cross = (-target[np.ix_(test, one)] @ weights_one
             -target[np.ix_(test, two)] @ weights_two
             + weights_one.T @ target[np.ix_(one, two)] @ weights_two)
    return target[np.ix_(test, test)] + (cross + cross.T) / 2


def _fit_response(target, zkernel, test, one, two, workspaces):
    w1, m1 = workspaces[0].fit(target[np.ix_(one, one)], zkernel[np.ix_(one, test)])
    w2, m2 = workspaces[1].fit(target[np.ix_(two, two)], zkernel[np.ix_(two, test)])
    return cross_residual_kernel(target, test, one, two, w1, w2), {"half_one": m1, "half_two": m2}


def build_split_kernels(x, y, z, *, beta, split_seed, configurations=CONFIGURATIONS):
    """Construct all requested response kernels with shared predictor work."""
    x, y, z = (np.asarray(v, dtype=np.float64) for v in (x, y, z))
    if z.ndim != 2 or len(x) != len(y) or len(x) != len(z):
        raise ValueError("Need equal-length X/Y and matrix Z")
    if not all(np.isfinite(v).all() for v in (x, y, z)):
        raise ValueError("Nonfinite observations")
    if any(c not in CONFIGURATIONS for c in configurations):
        raise ValueError("Unknown configuration")
    test, one, two = split_indices(len(x), beta, split_seed)
    train = np.concatenate((one, two))
    sx, fx = median_bandwidth(x[train])
    sy, fy = median_bandwidth(y[train])
    kz = 0.5 * (1 + rbf_matrix(z, np.sqrt(z.shape[1])))
    workspaces = (RidgeWorkspace(kz[np.ix_(one, one)]), RidgeWorkspace(kz[np.ix_(two, two)]))
    xres, xmeta = _fit_response(rbf_matrix(x, sx), kz, test, one, two, workspaces)
    split_hash = hashlib.sha256(np.concatenate((test, one, two)).astype("<i8").tobytes()).hexdigest()
    common = {"beta": float(beta), "N": len(x), "n_test": len(test),
              "n_train": len(train), "n_half": len(one), "split_sha256": split_hash,
              "sigma_x": sx, "median_y": sy, "median_zero_fallback_x": bool(fx),
              "median_zero_fallback_y": bool(fy), "sigma_z": float(np.sqrt(z.shape[1])),
              "bandwidth_source": "full_training", "y_center_source": "full_training",
              "x_target": "phi_X", "x_lambda": xmeta,
              "z_factor_placement": "second_hsic_input_Y_residual"}
    result, centered = {}, {}
    for config in configurations:
        multiplier = 2.0 if config.endswith("2m") else 1.0
        if multiplier not in centered:
            centered[multiplier] = training_center_kernel(rbf_matrix(y, sy * multiplier), train)
        target = centered[multiplier]
        second = config.startswith("second")
        if second:
            target = target * target  # tensorization BEFORE conditional regression
        yres, ymeta = _fit_response(target, kz, test, one, two, workspaces)
        result[config] = {"X": xres, "Y": yres, "Z": kz[np.ix_(test, test)],
                          "metadata": dict(common, y_multiplier=multiplier,
                                           sigma_y=sy * multiplier,
                                           y_target="centered_phi_Y_tensor2" if second else "centered_phi_Y",
                                           y_lambda=ymeta)}
    return result


def evaluate_split(x, y, z, beta, split_seed, bootstrap_seed, B=4999,
                   diagnostic=False, *, configurations=CONFIGURATIONS,
                   block_size=256, alpha=0.05):
    """Evaluate one paired split. Development uses test-only residual-Z tests."""
    kernels = build_split_kernels(x, y, z, beta=beta, split_seed=split_seed,
                                  configurations=configurations)
    if diagnostic:
        first = next(iter(kernels.values()))
        pairs = {"x": (first["X"], first["Z"])}
        pairs.update({c: (k["Y"], k["Z"]) for c, k in kernels.items()})
        tests = wild_bootstrap_many(pairs, seed=bootstrap_seed, B=B, alpha=alpha,
                                    block_size=block_size, sign_width=len(x))
        return {c: {"x_pvalue": tests["x"]["pvalue"], "y_pvalue": tests[c]["pvalue"],
                    "both_reject": bool(tests["x"]["reject"] and tests[c]["reject"]),
                    "x_diagnostic": tests["x"], "y_diagnostic": tests[c],
                    "metadata": k["metadata"]} for c, k in kernels.items()}
    pairs = {c: (k["X"], k["Y"] * k["Z"]) for c, k in kernels.items()}
    results = wild_bootstrap_many(pairs, seed=bootstrap_seed, B=B, alpha=alpha,
                                  block_size=block_size, sign_width=len(x))
    for c in results:
        results[c]["metadata"] = kernels[c]["metadata"]
    return results


def evaluate_configurations(x, y, z, beta_by_config: Mapping[str, float],
                            split_seed, bootstrap_seed, B=4999, diagnostic=False,
                            *, block_size=256, alpha=0.05):
    """Common data/permutation/sign streams; different frozen beta permitted."""
    if set(beta_by_config) != set(CONFIGURATIONS):
        raise ValueError("Require exactly the four paired configurations")
    output = {}
    for beta in sorted(set(float(v) for v in beta_by_config.values())):
        configs = tuple(c for c in CONFIGURATIONS if float(beta_by_config[c]) == beta)
        output.update(evaluate_split(x, y, z, beta, split_seed, bootstrap_seed, B,
                                     diagnostic, configurations=configs,
                                     block_size=block_size, alpha=alpha))
    return {c: output[c] for c in CONFIGURATIONS}


def choose_beta(diagnostic_rows, *, beta_grid=BETA_GRID, threshold=0.05,
                expected_diagnostics=100):
    """Largest ACTUALLY passing beta, else flagged smallest-beta fallback.

    diagnostic_rows maps each beta to a sequence of both-reject booleans.
    It represents repeated split diagnostics, not independent data sets.
    """
    rates = {}
    for beta in beta_grid:
        values = np.asarray(diagnostic_rows[beta], dtype=bool)
        if values.size != expected_diagnostics:
            raise ValueError("Incomplete development diagnostics")
        rates[str(beta)] = {"both_reject_count": int(values.sum()),
                            "diagnostics": int(values.size), "rate": float(values.mean()),
                            "pass": bool(values.mean() <= threshold)}
    passing = [float(b) for b in beta_grid if rates[str(b)]["pass"]]
    return {"beta": max(passing) if passing else min(beta_grid),
            "fallback": not bool(passing), "passed": bool(passing),
            "threshold": float(threshold), "diagnostics": rates}
