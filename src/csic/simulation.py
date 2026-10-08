"""Data generators and random streams for the paper's three main experiments."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import io
import json
from numbers import Integral
from pathlib import Path

import numpy as np
from scipy.integrate import quad


CASES = ("U0", "U1", "U2", "C0", "C1", "C2")
METHODS = ("HSIC", "CSIC", "KCI", "CSIC_CI")
STUDIES = ("main_unconditional", "main_conditional", "seoul")
PAPER_NAMESPACE = "confirmation_y2_v1"
SEOUL_SHA256 = "65ba6823721e6136cf9c33e39909a6650f150f1ebe32d433bcef96908196e83c"
Q_SD = (2.0 / np.pi) ** 0.25


def _sample_size(n):
    if isinstance(n, bool) or not isinstance(n, Integral) or n < 2:
        raise ValueError("n must be an integer >= 2")
    return int(n)


@lru_cache(maxsize=1)
def h_sd():
    """Population SD under N(0,1), from deterministic Gaussian quadrature."""
    integral, error = quad(
        lambda r: np.tanh(1.4 * r) ** 2 * np.exp(-r * r / 2) / np.sqrt(2 * np.pi),
        -np.inf, np.inf, epsabs=1e-12, epsrel=1e-12, limit=200)
    if integral <= 0 or error > 1e-10:
        raise ArithmeticError("Population normalization did not converge")
    return float(np.sqrt(integral))


def h(r):
    """Odd, population-centered and unit-variance mean direction."""
    return np.tanh(1.4 * np.asarray(r, dtype=float)) / h_sd()


def q(r):
    """Odd, population-centered and unit-variance scale direction."""
    r = np.asarray(r, dtype=float)
    return np.sign(r) * np.sqrt(np.abs(r)) / Q_SD


def nuisance(z):
    """The fixed conditional mean functions f and g used in C0--C2."""
    z = np.asarray(z, dtype=float)
    if z.ndim != 2 or z.shape[1] == 0 or not np.all(np.isfinite(z)):
        raise ValueError("Z must be a finite n-by-D array with D >= 1")
    scale = np.sqrt(z.shape[1])
    return (np.sin(z).sum(axis=1) / scale,
            (0.5 * z + 0.25 * (z * z - 1)).sum(axis=1) / scale)


def generate(case, n, seed, *, d=2, c0=8.0, return_latent=False):
    """Generate one Gaussian-covariate or unconditional dataset.

    Null cases use a=0; alternatives use a=c0/sqrt(n), with population-
    normalized directions h and q and no sample standardization.
    """
    if case not in CASES:
        raise ValueError(f"case must be one of {CASES}")
    n = _sample_size(n)
    c0 = float(c0)
    if not np.isfinite(c0) or c0 < 0:
        raise ValueError("c0 must be finite and nonnegative")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    a = 0.0 if case in ("U0", "C0") else c0 / np.sqrt(n)
    if case.startswith("U"):
        x, epsilon = rng.normal(size=(2, n))
        z = None
        if case == "U1":
            mean, sd = 2 * a * (x > 0.5), np.ones(n)
        elif case == "U2":
            mean, sd = np.zeros(n), np.sqrt(1 + 2.5 * a * np.abs(x))
        else:
            mean, sd = np.zeros(n), np.ones(n)
        y = mean + sd * epsilon
        latent = {"epsilon_y": epsilon, "conditional_mean": mean,
                  "conditional_sd": sd}
        dimension = 0
    else:
        if isinstance(d, bool) or not isinstance(d, Integral) or d < 1:
            raise ValueError("d must be an integer >= 1")
        dimension = int(d)
        z = rng.normal(size=(n, dimension))
        r, epsilon = rng.normal(size=(2, n))
        f, g = nuisance(z)
        x = f + r
        mean = g + a * h(r) if case == "C1" else g
        # exp(a*q/2) is a standard deviation; the variance is exp(a*q).
        with np.errstate(over="raise", under="raise", invalid="raise"):
            sd = np.exp(a * q(r) / 2) if case == "C2" else np.ones(n)
        y = mean + sd * epsilon
        latent = {"r": r, "epsilon_y": epsilon, "f": f, "g": g,
                  "conditional_mean": mean, "conditional_sd": sd}
    if not np.all(np.isfinite(y)) or not np.all(sd > 0):
        raise ArithmeticError("Nonfinite observations or nonpositive variance")
    sample = {"x": x, "y": y, "z": z,
              "metadata": {"case": case, "n": n, "d": dimension,
                           "c0": 0.0 if a == 0 else c0, "a": float(a),
                           "seed": int(seed), "role": "null" if a == 0 else "alternative",
                           "null_id": "U0" if z is None else "C0"}}
    if return_latent:
        sample["latent"] = latent
    return sample


def _job_values(job):
    study, case = job["study"], job["case"]
    if study not in STUDIES or case not in CASES:
        raise ValueError("Unknown study or case")
    n = _sample_size(job["n"])
    d, rep = job["d"], job["rep"]
    if isinstance(d, bool) or not isinstance(d, Integral) or d < 0:
        raise ValueError("d must be a nonnegative integer")
    if isinstance(rep, bool) or not isinstance(rep, Integral) or rep < 0:
        raise ValueError("rep must be a nonnegative integer")
    if study == "main_unconditional":
        if not case.startswith("U") or d != 0:
            raise ValueError("Unconditional jobs require a U case and d=0")
    elif not case.startswith("C") or d < 1:
        raise ValueError("Conditional jobs require a C case and d>=1")
    if study == "seoul" and d != 5:
        raise ValueError("Seoul jobs use the five-column weather pool")
    c0 = float(job["c0"])
    if not np.isfinite(c0) or c0 < 0:
        raise ValueError("c0 must be finite and nonnegative")
    return study, case, n, int(d), c0, int(rep)


def paper_seed(job, role, method=None, root=20261008):
    """Return the paper's SHA256-derived 64-bit stream seed.

    Main null jobs retain c0=8.0 in the seed even though their a is zero.
    Data streams omit method; mixture streams include it.
    """
    if role not in ("data", "mixture"):
        raise ValueError("role must be data or mixture")
    if (role == "data" and method is not None) or (role != "data" and method not in METHODS):
        raise ValueError("Data streams omit method; other streams require a known method")
    if isinstance(root, bool) or not isinstance(root, Integral) or root < 0:
        raise ValueError("root must be a nonnegative integer")
    study, case, n, d, c0, rep = _job_values(job)
    value = [int(root), PAPER_NAMESPACE, role, study, case, n, d, c0, rep]
    if role != "data":
        value.append(method)
    raw = json.dumps(value, separators=(",", ":"), ensure_ascii=True,
                     allow_nan=False).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little")


def load_seoul(path):
    """Load and verify the paper's 8760-by-5 standardized weather pool."""
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != SEOUL_SHA256:
        raise ValueError("Seoul pool SHA256 does not match the paper archive")
    pool = np.load(io.BytesIO(raw), allow_pickle=False)
    if pool.shape != (8760, 5) or not np.all(np.isfinite(pool)):
        raise ValueError("Seoul pool must contain 8760 finite five-dimensional rows")
    pool.setflags(write=False)
    return pool


def generate_job(job, seed_root=20261008, seoul_pool=None):
    """Generate one dataset; Seoul requires a pool from load_seoul."""
    study, case, n, d, c0, _ = _job_values(job)
    seed = paper_seed(job, "data", root=seed_root)
    if study != "seoul":
        sample = generate(case, n, seed, d=d, c0=c0)
        sample["metadata"].update(namespace=PAPER_NAMESPACE, study=study)
        return sample
    if seoul_pool is None:
        raise ValueError("Seoul generation requires the fixed weather pool")
    pool = np.asarray(seoul_pool, dtype=float)
    if pool.shape != (8760, 5) or not np.all(np.isfinite(pool)):
        raise ValueError("Seoul pool must contain 8760 finite five-dimensional rows")
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, 8760, size=n, dtype=np.int64)
    z = pool[ix]
    r, epsilon = rng.normal(size=(2, n))
    f, g = nuisance(z)
    a = 0. if case == "C0" else c0 / np.sqrt(n)
    x = f + r
    mean = g + a * h(r) if case == "C1" else g
    with np.errstate(over="raise", under="raise", invalid="raise"):
        sd = np.exp(a * q(r) / 2) if case == "C2" else np.ones(n)
        y = mean + sd * epsilon
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.all(sd > 0):
        raise ArithmeticError("Nonfinite observations or nonpositive variance")
    return dict(x=x, y=y, z=z, row_indices=ix.astype(np.int16),
                metadata=dict(case=case, n=n, d=5, a=float(a),
                              c0=0. if a == 0 else c0, seed=seed,
                              population="Seoul fixed weather archive",
                              namespace=PAPER_NAMESPACE, study=study))
