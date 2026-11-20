"""Bernoulli response laws and paired base draws at n=400, D=5."""
from functools import lru_cache
from typing import Any

import numpy as np
from scipy.integrate import quad

from .engine import derive_seed

PROTOCOL_ID = "binary_response_minimal_v1"
MASTER_SEED = 2026111501
N, D = 400, 5
STAGES = {"null_calibration": 999, "null_validation": 1000, "alternative": 500}
P_VALUES = {"p025": .25, "p049": .49, "p050": .50}
CONFIGS = ("first_y1", "second_y2")


def spectral_seed(stage, rep, pid, config):
    return derive_seed(MASTER_SEED, PROTOCOL_ID, "spectral", stage, rep, pid, config)


@lru_cache(maxsize=1)
def h_sd() -> float:
    """Population normalization; called on demand, never at module import."""
    integral, error = quad(
        lambda r: np.tanh(1.4 * r) ** 2 * np.exp(-r * r / 2) / np.sqrt(2 * np.pi),
        -np.inf, np.inf, epsabs=1e-12, epsrel=1e-12, limit=200)
    if not np.isfinite(integral) or integral <= 0 or not np.isfinite(error) or error > 1e-10:
        raise ArithmeticError("Population normalization did not converge")
    return float(np.sqrt(integral))


def nuisance(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2 or z.shape[1] != D or not np.isfinite(z).all():
        raise ValueError(f"Z must be a finite n-by-{D} array")
    return (np.sin(z).sum(axis=1) / np.sqrt(D),
            (0.5 * z + 0.25 * (z * z - 1)).sum(axis=1) / np.sqrt(D))


def binary_probability(z: np.ndarray, r: np.ndarray, p_id: str,
                       stage: str) -> tuple[np.ndarray, dict[str, Any]]:
    if p_id not in P_VALUES or stage not in STAGES:
        raise ValueError("Unknown p_id or stage")
    z, r = np.asarray(z, dtype=np.float64), np.asarray(r, dtype=np.float64)
    if r.shape != (len(z),) or not np.isfinite(r).all():
        raise ValueError("r must be a finite vector matching Z")
    _, g = nuisance(z)
    _, g_negative = nuisance(-z)
    v = (np.tanh(g) - np.tanh(g_negative)) / 2
    p = P_VALUES[p_id]
    pi = p + 0.05 * v
    delta = .2 if stage == "alternative" else 0.0
    norm = h_sd()
    h = np.tanh(1.4 * r) / norm
    probability = pi + delta * np.sqrt(pi * (1 - pi)) * h
    if not np.isfinite(probability).all() or np.any(probability <= 0) or np.any(probability >= 1):
        raise ArithmeticError("DGP probability is not strictly inside (0,1); no clipping is permitted")
    return probability, {"p_id": p_id, "p": float(p), "delta": float(delta),
                         "h_sd": norm, "probability_min": float(probability.min()),
                         "probability_max": float(probability.max()),
                         "pi_min": float(pi.min()), "pi_max": float(pi.max()),
                         "response_nuisance": "odd bounded transform of original g"}


def generate_block(stage, rep):
    """Share Z, r and U across the two original prevalences."""
    if stage not in STAGES or not 0 <= rep < STAGES[stage]:
        raise ValueError("Unknown stage or replication")
    def rng(component):
        seed = derive_seed(MASTER_SEED, PROTOCOL_ID, "data", stage, rep, component)
        return np.random.Generator(np.random.PCG64(seed))
    z = rng("Z").standard_normal((N, D))
    r = rng("r").standard_normal(N)
    u = rng("U").random(N)
    f, _ = nuisance(z)
    return {"x": f + r, "z": z, "r": r, "u": u}


def sample_from_block(block, pid, stage):
    probability, _ = binary_probability(block["z"], block["r"], pid, stage)
    return {"x": block["x"], "z": block["z"], "y": (block["u"] < probability).astype(float)}
