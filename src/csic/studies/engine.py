"""Paired feature-order and bandwidth configurations."""
from __future__ import annotations

import hashlib
import json
from numbers import Integral

import numpy as np
from scipy.linalg import solve
from scipy.spatial.distance import cdist

from ..inference import matrix_penalty, median_bandwidth
from ..kernels import _checked_eigh, _matrix, _result, center, mixture_pvalue


CONFIG_IDS = ("first_y1", "second_y1", "first_y2", "second_y2")


def derive_seed(seed, *tokens):
    """Stable independent 64-bit stream; excludes B so draw prefixes persist."""
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    raw = json.dumps([int(seed), "csic_revision_engine_v1", *tokens],
                     sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little")


def normalize_variant(variant=None):
    """Return the complete, unambiguous estimator configuration."""
    value = {} if variant is None else dict(variant)
    unknown = set(value) - {"kind", "rho400", "train_fraction"}
    if unknown:
        raise ValueError(f"Unknown variant fields: {sorted(unknown)}")
    kind = value.get("kind", "same_sample")
    if kind not in ("same_sample", "holdout"):
        raise ValueError("variant kind must be same_sample or holdout")
    rho = float(value.get("rho400", .01))
    if not np.isfinite(rho) or rho <= 0:
        raise ValueError("rho400 must be finite and positive")
    result = {"kind": kind, "rho400": rho}
    if kind == "holdout":
        fraction = float(value.get("train_fraction", .5))
        if fraction not in (.5, .75):
            raise ValueError("The frozen holdout fractions are 0.5 and 0.75")
        result["train_fraction"] = fraction
    elif "train_fraction" in value:
        raise ValueError("same_sample does not use a train_fraction")
    return result


def calibration_seed(seed, config_id, variant=None):
    if config_id not in CONFIG_IDS:
        raise ValueError("Unknown factorial configuration")
    return derive_seed(seed, "spectral", normalize_variant(variant), config_id)


def _observations(value, label):
    value = np.asarray(value, dtype=float)
    if value.ndim == 1:
        value = value[:, None]
    if value.ndim != 2 or value.shape[1] == 0 or not np.all(np.isfinite(value)):
        raise ValueError(f"{label} must be finite observations in rows")
    return value


def prepare_sample(sample, *, decompose_z=True):
    """Cache distances and Z eigendecomposition for all same-sample variants."""
    x = _observations(sample["x"], "X")
    y = _observations(sample["y"], "Y")
    z = None if sample.get("z") is None else _observations(sample["z"], "Z")
    n = len(x)
    if n < 2 or len(y) != n or (z is not None and len(z) != n):
        raise ValueError("X, Y, Z must have matching n >= 2")
    sx, fallback_x = median_bandwidth(x)
    sy, fallback_y = median_bandwidth(y)
    dx = cdist(x, x, metric="sqeuclidean")
    dy = cdist(y, y, metric="sqeuclidean")
    kx = np.exp(-dx / (2 * sx * sx))
    data = dict(x=x, y=y, z=z, n=n, dx=dx, dy=dy, KX=kx,
                median_x=sx, median_y=sy, fallback_x=fallback_x,
                fallback_y=fallback_y, source_metadata=dict(sample.get("metadata", {})))
    if z is None:
        data["KXc"] = center(kx)
        data["x_spectrum"] = _checked_eigh(data["KXc"] / n)
    else:
        kz = (1 + np.exp(-cdist(z, z, metric="sqeuclidean") / (2 * z.shape[1]))) / 2
        data.update(KZ=kz, KAc=center(kx * kz))
        if decompose_z:
            values, vectors = _checked_eigh(kz, vectors=True)
            data.update(z_values=values, z_vectors=vectors)
    # Y-centering and quadratic features do not depend on ridge.
    data["responses"] = {}
    for mult in (1, 2):
        ky = np.exp(-dy / (2 * (sy * mult) ** 2))
        kyc = center(ky)
        data["responses"][mult] = {"KY": ky, "first": kyc, "second": kyc * kyc}
    return data


def reference_center(gram, reference_indices):
    """Center a Gram matrix about a specified empirical feature mean.

    No evaluation observation is used when reference_indices are training rows.
    """
    gram = _matrix(gram)
    indices = np.asarray(reference_indices)
    if indices.ndim != 1 or indices.size == 0 or indices.dtype.kind not in "iu":
        raise ValueError("Reference indices must be a nonempty integer vector")
    if np.any(indices < 0) or np.any(indices >= len(gram)):
        raise ValueError("Reference index outside Gram matrix")
    row_mean = gram[:, indices].mean(axis=1)
    return _matrix(gram - row_mean[:, None] - row_mean[None, :]
                   + float(gram[np.ix_(indices, indices)].mean()))


def _metadata(data, variant, mult, n_eval, **extra):
    return dict(variant=dict(variant), n=data["n"], n_total=data["n"], n_eval=n_eval,
                sigma_x=data["median_x"], sigma_y=data["median_y"] * mult,
                sigma_y_multiplier=float(mult), sample_standardization=False,
                median_zero_fallback_x=bool(data["fallback_x"]),
                median_zero_fallback_y=bool(data["fallback_y"]),
                source_sample=data["source_metadata"], **extra)


def prepare_four(sample, *, variant=None, split_seed=0, prepared_sample=None):
    """Prepare the four statistics; exposes residual Grams for direct audits.

    prepared_sample from prepare_sample can be shared across ridge variants.
    The supplied sample is never regenerated: in holdout, its signal remains
    a_N, while ridge uses n_train and statistic/spectrum use n_eval.
    """
    variant = normalize_variant(variant)
    data = (prepare_sample(sample, decompose_z=variant["kind"] == "same_sample")
            if prepared_sample is None else prepared_sample)
    n = data["n"]
    output = {"configs": {}, "variant": variant, "n_total": n}
    if data["z"] is None:
        if variant["kind"] != "same_sample":
            raise ValueError("Holdout residual regression requires conditioning variables")
        for mult in (1, 2):
            for order in ("first", "second"):
                cid = f"{order}_y{mult}"
                lr = data["responses"][mult][order]
                ry = _checked_eigh((lr if order == "first" else center(lr)) / n)
                output["configs"][cid] = dict(
                    method="HSIC" if order == "first" else "CSIC",
                    LA=data["KXc"], LR=lr, n_eval=n, conditional=False,
                    spectrum_factors=(data["x_spectrum"], ry),
                    metadata=_metadata(data, variant, mult, n, rho400=None,
                                       ridge_matrix=None, lambda_population=None))
        return output

    if variant["kind"] == "same_sample":
        rho = matrix_penalty(variant["rho400"], n)
        d, v = data["z_values"], data["z_vectors"]
        a = (v * (rho / (d + rho))) @ v.T
        la = _matrix(a @ data["KAc"] @ a)
        smooth = d / (d + rho)
        output.update(A=a, KZ=data["KZ"])
        for mult in (1, 2):
            for order in ("first", "second"):
                cid = f"{order}_y{mult}"
                lr = _matrix(a @ data["responses"][mult][order] @ a)
                output["configs"][cid] = dict(
                    method="KCI" if order == "first" else "CSIC_CI",
                    LA=la, LR=lr, n_eval=n, conditional=True,
                    metadata=_metadata(data, variant, mult, n, n_train=n,
                                       rho400=variant["rho400"], ridge_matrix=rho,
                                       lambda_population=rho / n,
                                       trace_S2_over_n=float(smooth @ smooth / n)))
        return output

    # Same random permutation is shared by all four cells and ridge choices.
    permutation = np.random.default_rng(derive_seed(split_seed, "holdout_split")).permutation(n)
    nt = int(np.floor(n * variant["train_fraction"]))
    ne = n - nt
    if nt < 2 or ne < 2:
        raise ValueError("Holdout requires at least two training and evaluation observations")
    train, evaluation = permutation[:nt], permutation[nt:]
    sx, fallback_x = median_bandwidth(data["x"][train])
    sy, fallback_y = median_bandwidth(data["y"][train])
    # The original row order is retained; columns R index that order.
    kz = data["KZ"]
    rho = matrix_penalty(variant["rho400"], nt)
    ktt = kz[np.ix_(train, train)]
    ket = kz[np.ix_(evaluation, train)]
    prediction = solve(ktt + rho * np.eye(nt), ket.T, assume_a="pos", check_finite=False).T
    residual = np.zeros((ne, n))
    residual[:, train] = -prediction
    residual[np.arange(ne), evaluation] = 1
    kx = np.exp(-data["dx"] / (2 * sx * sx))
    ka = reference_center(kx * kz, train)
    la = _matrix(residual @ ka @ residual.T)
    d = _checked_eigh(ktt)
    smooth = d / (d + rho)
    local = dict(data, median_x=sx, median_y=sy,
                 fallback_x=fallback_x, fallback_y=fallback_y)
    output.update(R=residual, train_indices=train, evaluation_indices=evaluation, KZ=kz)
    for mult in (1, 2):
        ky = np.exp(-data["dy"] / (2 * (sy * mult) ** 2))
        kycenter = reference_center(ky, train)
        for order in ("first", "second"):
            response = kycenter if order == "first" else kycenter * kycenter
            lr = _matrix(residual @ response @ residual.T)
            cid = f"{order}_y{mult}"
            output["configs"][cid] = dict(
                method="KCI" if order == "first" else "CSIC_CI",
                LA=la, LR=lr, n_eval=ne, conditional=True,
                metadata=_metadata(local, variant, mult, ne, n_train=nt,
                                   rho400=variant["rho400"], ridge_matrix=rho,
                                   lambda_population=rho / nt,
                                   trace_S2_over_n=float(smooth @ smooth / nt),
                                   split_seed=int(split_seed),
                                   split_indices_sha256=hashlib.sha256(permutation.astype("<i8").tobytes()).hexdigest(),
                                   bandwidth_source="training", centering_source="training"))
    return output


def evaluate_prepared(prepared, *, seed, B=4999, trace_rtol=1e-8, return_spectra=True):
    """Apply unchanged spectral calibration to prepared residual features."""
    results = {}
    for cid in CONFIG_IDS:
        item = prepared["configs"][cid]
        n = item["n_eval"]
        statistic = float(np.sum(item["LA"] * item["LR"]) / n)
        if item["conditional"]:
            values = _checked_eigh(center(item["LA"] * item["LR"]) / n)
        else:
            values = np.multiply.outer(*item["spectrum_factors"])
        estimate = _result(statistic, values, trace_rtol)
        cseed = calibration_seed(seed, cid, prepared["variant"])
        cal = mixture_pvalue(estimate["statistic"], estimate["weights"],
                             np.random.default_rng(cseed), B=B, alpha=.05)
        record = dict(method=item["method"], statistic=estimate["statistic"],
                      pvalue=cal["pvalue"], calibration=cal,
                      metadata={**item["metadata"], "calibration_seed": cseed,
                                "B_requested": int(B), "trace_rtol": float(trace_rtol)},
                      spectrum={k: value for k, value in estimate.items()
                                if k not in ("weights", "statistic")})
        if return_spectra:
            if item["conditional"]:
                record["weights"] = estimate["weights"]
            else:
                # Store two factors, not the potentially n^2 products. Reapply
                # the frozen trimming rule when reconstructing the mixture.
                record["spectrum_factors"] = tuple(a.copy() for a in item["spectrum_factors"])
        results[cid] = record
    return results


def evaluate_four(sample, *, variant=None, seed=0, B=4999, trace_rtol=1e-8,
                  return_spectra=True):
    prepared = prepare_four(sample, variant=variant, split_seed=derive_seed(seed, "split"))
    return evaluate_prepared(prepared, seed=seed, B=B, trace_rtol=trace_rtol,
                             return_spectra=return_spectra)


def evaluate_variants(sample, variants, *, seed=0, B=4999, trace_rtol=1e-8,
                      return_spectra=True):
    """Evaluate a mapping of candidate IDs with shared exact kernel caches."""
    variants = {key: normalize_variant(value) for key, value in variants.items()}
    shared = prepare_sample(sample, decompose_z=any(v["kind"] == "same_sample"
                                                   for v in variants.values()))
    results = {}
    for variant_id, variant in variants.items():
        prepared = prepare_four(sample, variant=variant, split_seed=derive_seed(seed, "split"),
                                prepared_sample=shared)
        results[variant_id] = evaluate_prepared(prepared, seed=seed, B=B,
                                               trace_rtol=trace_rtol,
                                               return_spectra=return_spectra)
    return results
