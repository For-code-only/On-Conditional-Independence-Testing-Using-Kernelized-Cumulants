"""Independent p=.50 experiment with common first/mixed spectral draws."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from . import binary_kernel as core, binary_law as protocol
from .mixed import SECOND_WEIGHT

DESIGN = {"seed_root": 202611204207, "seed_namespace": "mixed-response-independent-binary-p050-20261120-v1",
          "delta_alternative": .2, "ridge": .01}
N, D, B = 400, 5, 4999


def seed(stage, rep, *tokens):
    return protocol.derive_seed(DESIGN["seed_root"], DESIGN["seed_namespace"], stage, rep, *tokens)


def sample(stage, rep):
    z = np.random.default_rng(seed(stage, rep, "Z")).standard_normal((N, D))
    r = np.random.default_rng(seed(stage, rep, "r")).standard_normal(N)
    u = np.random.default_rng(seed(stage, rep, "U")).random(N)
    f, _ = protocol.nuisance(z)
    probability, meta = protocol.binary_probability(z, r, "p050", stage)
    if meta["delta"] != (DESIGN["delta_alternative"] if stage == "alternative" else 0):
        raise ArithmeticError("Unexpected alternative strength in the frozen binary protocol")
    return {"x": f + r, "y": (u < probability).astype(float), "z": z}


def spectral_fit(product):
    # Both the trace statistic and the null weights are on the n*T scale.
    t = max(0.0, float(np.sum(product) / len(product)))
    eigs, _ = core.checked_eigh(core.center_matrix(product) / len(product))
    weights, _ = core.trim_weights(eigs, trace_rtol=1e-8)
    return t, weights


def shared_spectral_pvalues(fits, spectral_seed):
    # Share draws only for comparing two marginal rules. Each one has its own spectrum.
    if len(fits[0][1]) != len(fits[1][1]) or any(len(w) < 2 or np.all(w == w[0]) for _, w in fits):
        return [core.mixture_pvalue(t, w, spectral_seed, B)["pvalue"] for t, w in fits]
    rng = np.random.default_rng(spectral_seed)
    weights = np.column_stack([fit[1] for fit in fits])
    statistics = np.array([fit[0] for fit in fits])
    tails = np.zeros(2, dtype=int)
    for start in range(0, B, 256):
        draws = rng.standard_normal((min(B - start, 256), len(weights)))
        np.square(draws, out=draws)
        tails += np.count_nonzero(draws @ weights >= statistics, axis=0)
    return ((1 + tails) / (1 + B)).tolist()


def run_one(stage, rep):
    data = sample(stage, rep)
    x, y, z = (data[key] for key in ("x", "y", "z"))
    b = core.binary_parameters(y)
    count = b["k"]
    if b["constant"]:
        return stage, rep, count, 1., 1., 1., 0., 0., 0., 0
    mx, _, _ = core.median_bandwidth(x)
    my, _, _ = core.median_bandwidth(y)
    kz = (1 + core.rbf(z, np.sqrt(D))) / 2
    kx = core.rbf(x, mx)
    rho = DESIGN["ridge"]
    identity = np.eye(N)
    A = core.matrix(cho_solve(cho_factor(kz + rho * identity, lower=True,
                                        check_finite=False), rho * identity, check_finite=False))
    LA = core.matrix(A @ core.center_matrix(kx * kz) @ A)
    first_distance = core.feature_distance(my)
    second_distance = core.feature_distance(2 * my)

    # The first and original quadratic comparisons follow E.6's exact binary factors.
    v1 = A @ b["e"]
    v2 = b["alpha"] * v1 + b["beta"] * A.sum(axis=1)
    first_fit = spectral_fit(LA * np.outer(v1, v1) * first_distance)
    # Normalize the original quadratic residual before diagonalizing, preserving accuracy
    # for equal labels counts where only the small ridge constant component survives.
    amp = float(np.max(np.abs(v2)))
    if amp == 0:
        second_fit = (0., np.array([]))
    else:
        second_fit = spectral_fit(LA * np.outer(v2 / amp, v2 / amp))

    # Independently form the actual mixed Gram on every sample, without using the identity.
    C1 = core.center_matrix(core.rbf(y, my))
    C2 = core.center_matrix(core.rbf(y, 2 * my))
    M = C1 + SECOND_WEIGHT * core.center_matrix(C2 ** 2)
    mixed_fit = spectral_fit(LA * core.matrix(A @ M @ A))
    first_seed = seed(stage, rep, "spectral", "first_y1")
    second_seed = seed(stage, rep, "spectral", "original_second_y2")
    first_p, mixed_p = shared_spectral_pvalues([first_fit, mixed_fit], first_seed)
    second_p = core.mixture_pvalue(*second_fit, second_seed, B)["pvalue"]

    scalar = 1 + SECOND_WEIGHT * second_distance ** 2 * b["alpha"] ** 2 / first_distance
    gram_error = float(np.max(np.abs(M - scalar * C1)))
    stat_error = abs(mixed_fit[0] / (scalar * first_fit[0]) - 1)
    if len(mixed_fit[1]) != len(first_fit[1]):
        raise ArithmeticError(f"Different trimmed ranks: {stage}, {rep}")
    spectral_error = float(np.abs(mixed_fit[1] - scalar * first_fit[1]).sum() /
                           (scalar * np.sum(first_fit[1])))
    if max(gram_error, stat_error, spectral_error) > 1e-8:
        raise ArithmeticError(f"Failed binary Gram identity: {stage}, {rep}")

    audited = rep < 3 or (b["balanced"] and rep % 17 == 0)
    if audited:
        prepared = core.prepare_sample(data)
        for config, expected_p, stream in (("first_y1", first_p, first_seed),
                                            ("second_y2", second_p, second_seed)):
            baseline = core.evaluate_prepared(prepared, config, stream, B=B)
            if baseline["calibration"]["pvalue"] != expected_p:
                raise ArithmeticError(f"Disagreement with original E.6 {config}: {stage}, {rep}")
        if core.mixture_pvalue(*mixed_fit, first_seed, B)["pvalue"] != mixed_p:
            raise ArithmeticError(f"Mixed MC implementation disagreement: {stage}, {rep}")
    return stage, rep, count, first_p, second_p, mixed_p, gram_error, stat_error, spectral_error, int(audited)
