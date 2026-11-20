"""Continuous mixed-response study with shared observations and spectral draws."""
import math
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from ..inference import matrix_penalty
from ..kernels import center, _matrix, _checked_eigh, _result, mixture_pvalue
from .engine import prepare_sample, prepare_four
from .streams import generate_sample, stable_seed, seoul_pool, PAPER_PLAN
from .statistics import fit_null_calibration, rate_summary, paired_summary

CONFIG = {
    "seed_root": 2026112027, "D": 5, "rho400": .01, "trace_rtol": 1e-8,
    "cohorts": {
        "calibrate": {"case": "C0", "strength": 0., "reps": 5999},
        "validate": {"case": "C0", "strength": 0., "reps": 6000},
        "weak_mean": {"case": "C1", "strength": 2., "reps": 2000},
        "strong_mean": {"case": "C1", "strength": 8., "reps": 2000},
        "variance": {"case": "C2", "strength": 8., "reps": 4000},
    },
}
METHODS = ("first", "second", "mixed")
ALPHA = .05
B = 4999
SECOND_WEIGHT = 7.438530300238965


def spectral_scores(statistics, weights, seed, audit):
    """Value each marginal chi-square mixture on a common Gaussian stream."""
    p = np.empty(len(weights))
    sizes = {len(w) for w in weights}
    for size in sizes:
        columns = [k for k, w in enumerate(weights) if len(w) == size]
        ordinary = [k for k in columns if size > 0 and not np.all(weights[k] == weights[k][0])]
        for k in set(columns) - set(ordinary):
            p[k] = mixture_pvalue(statistics[k], weights[k], np.random.default_rng(seed), B=B)["pvalue"]
        if not ordinary:
            continue
        rng = np.random.default_rng(seed)
        matrix = np.column_stack([weights[k] for k in ordinary])
        tails = np.zeros(len(ordinary), dtype=int)
        for start in range(0, B, 256):
            gaussian = rng.standard_normal((min(B - start, 256), size))
            np.square(gaussian, out=gaussian)
            tails += np.count_nonzero(gaussian @ matrix >= statistics[ordinary], axis=0)
        p[ordinary] = (1 + tails) / (B + 1)
    if audit:
        for t, spectrum, score in zip(statistics, weights, p):
            ref = mixture_pvalue(t, spectrum, np.random.default_rng(seed), B=B)["pvalue"]
            if ref != score:
                raise ArithmeticError("Monte Carlo disagrees with the original implementation")
    return p


def one_rep(study, n, cohort, rep):
    spec = CONFIG["cohorts"][cohort]
    job = dict(study=study, n=n, d=CONFIG["D"], role=cohort, case=spec["case"],
               c0=spec["strength"], rep=rep)
    sample = generate_sample(CONFIG, job, pool=seoul_pool() if study == "seoul" else None)
    data = prepare_sample(sample, decompose_z=False)
    rho = matrix_penalty(CONFIG["rho400"], n)
    system = data["KZ"].copy()
    system.flat[::n + 1] += rho
    A = _matrix(cho_solve(cho_factor(system, lower=True, check_finite=False),
                          rho * np.eye(n), check_finite=False))
    left = _matrix(A @ data["KAc"] @ A)
    C1 = data["responses"][1]["first"]
    Q2 = data["responses"][2]["second"]
    # The standalone quadratic comparator retains the manuscript's original Q2.
    grams = (C1, Q2, C1 + SECOND_WEIGHT * center(Q2))
    statistics, spectra = [], []
    audit = rep == 0
    if audit:
        ref = prepare_four(sample)
        np.testing.assert_allclose(left, ref["configs"]["first_y1"]["LA"], rtol=1e-8, atol=1e-10)
    for j, gram in enumerate(grams):
        right = _matrix(A @ gram @ A)
        if audit and j < 2:
            name = ("first_y1", "second_y2")[j]
            np.testing.assert_allclose(right, ref["configs"][name]["LR"], rtol=1e-8, atol=1e-10)
        product = left * right
        t = float(product.sum() / n)
        fit = _result(t, _checked_eigh(center(product) / n), CONFIG["trace_rtol"])
        spectra.append(fit["weights"])
        statistics.append(fit["statistic"])
    seed = stable_seed(CONFIG["seed_root"], "shared_mixture", study, n, cohort, rep)
    p = spectral_scores(np.asarray(statistics), spectra, seed, audit)
    return (study, n, cohort, rep, str(sample["metadata"]["data_seed"]), *map(float, p))

def covariance_energies(sample):
    grams = prepare_sample(sample, decompose_z=False)
    n = grams["n"]
    identity = np.eye(n)
    ridge = 0.01
    factor = cho_factor(grams["KZ"] + ridge * identity, check_finite=False)
    residual = cho_solve(factor, ridge * identity, check_finite=False)
    covariate = residual @ grams["KAc"] @ residual
    first = grams["responses"][1]["first"]
    second = center(grams["responses"][2]["second"])
    energies = []
    for response in (first, second):
        residual_response = residual @ response @ residual
        covariance = center(covariate * residual_response) / n
        energies.append(np.sum(covariance ** 2))
    return energies


def operating_point(null, alpha=ALPHA):
    n = len(null)
    target = n * alpha
    rank = math.ceil(target)
    threshold = np.partition(null, rank - 1, axis=0)[rank - 1]
    lower = np.count_nonzero(null < threshold, axis=0)
    ties = np.count_nonzero(null == threshold, axis=0)
    fraction = (target - lower) / ties
    if np.any((fraction < 0) | (fraction > 1)):
        raise ArithmeticError("Tie fractions outside [0,1]")
    return threshold, fraction, lower, ties


def rejection_rates(values, threshold, fraction):
    lower = np.count_nonzero(values < threshold, axis=0)
    ties = np.count_nonzero(values == threshold, axis=0)
    return (lower + fraction * ties) / len(values)


def roc_comparison(null, alternative_groups, design, design_index, n, draw_count):
    thresholds, fractions, below, tied = operating_point(null)
    np.testing.assert_allclose(rejection_rates(null, thresholds, fractions), ALPHA, atol=1e-14)
    cohorts = list(alternative_groups)
    alternatives = list(alternative_groups.values())
    rates = np.stack([rejection_rates(values, thresholds, fractions) for values in alternatives])
    draws = np.empty((draw_count, len(cohorts), len(METHODS)))
    source = np.random.SeedSequence([2026112047, design_index, n])
    null_rng, *alt_rngs = [np.random.default_rng(child) for child in source.spawn(len(cohorts) + 1)]
    for b in range(draw_count):
        sampled_null = null[null_rng.integers(0, len(null), size=len(null))]
        cuts, ties, _, _ = operating_point(sampled_null)
        for j, (values, rng) in enumerate(zip(alternatives, alt_rngs)):
            sampled_alt = values[rng.integers(0, len(values), size=len(values))]
            draws[b, j] = rejection_rates(sampled_alt, cuts, ties)
    lower, upper = np.percentile(100 * draws, (2.5, 97.5), axis=0)
    rate_rows, contrast_rows = [], []
    for j, cohort in enumerate(cohorts):
        common = dict(design=design, n=n, cohort=cohort, pooled_C0=len(null),
                      N_alt=len(alternatives[j]), target_FPR_pct=100 * ALPHA)
        for i, method in enumerate(METHODS):
            rate_rows.append(dict(**common, method=method, cutoff=thresholds[i],
                                  C0_below=int(below[i]), C0_tied=int(tied[i]),
                                  tie_fraction=fractions[i], tpr_pct=100*rates[j, i],
                                  low95_pct=lower[j, i], high95_pct=upper[j, i]))
        mixed = METHODS.index("mixed")
        for other in ("first", "second"):
            k = METHODS.index(other)
            differences = 100 * (draws[:, j, mixed] - draws[:, j, k])
            lo, hi = np.percentile(differences, [2.5, 97.5])
            contrast_rows.append(dict(**common, method="mixed", comparator=other,
                                      diff_pp=100*(rates[j, mixed] - rates[j, k]),
                                      low95_pp=lo, high95_pp=hi, draws=draw_count))
    return rate_rows, contrast_rows


def pilot(ctx):
    def one(index):
        job = dict(study="main_conditional", n=400, d=5, role="calibration", case="C0",
                   c0=0., rep=9999+index)
        return covariance_energies(generate_sample(PAPER_PLAN, job))
    energies = ctx.scores("pilot", 3 if ctx.quick else 256, one)
    sums = energies.sum(axis=0)
    ctx.table("weight", [dict(datasets=len(energies), first_energy=sums[0],
                              second_energy=sums[1], weight=np.sqrt(sums[0]/sums[1]))])


def run(ctx):
    rates, pairs, roc_rows, roc_pairs = [], [], [], []
    for design_index, design in enumerate(("main_conditional", "seoul")):
        for n in (200, 400):
            cohorts = {}
            for cohort, spec in CONFIG["cohorts"].items():
                small = 19 if cohort == "calibrate" else 3
                cohorts[cohort] = ctx.scores(f"{design}_{n}_{cohort}", small if ctx.quick else spec["reps"],
                                             lambda rep: one_rep(design, n, cohort, rep)[5:])
            bank = cohorts.pop("calibrate")
            cutoffs = np.array([fit_null_calibration(col)["cutoff"] for col in bank.T])
            for cohort, values in cohorts.items():
                raw, outer = values <= ALPHA, values < cutoffs
                key = dict(design=design, n=n, cohort=cohort)
                for j, method in enumerate(METHODS):
                    rates.append(dict(**key, method=method, raw_rate=float(raw[:, j].mean()),
                                      cutoff=cutoffs[j], **rate_summary(outer[:, j])))
                if cohort == "validate":
                    for left, right in ((2, 0), (2, 1), (0, 1)):
                        pairs.append(dict(**key, left=METHODS[left], right=METHODS[right],
                                          **paired_summary(outer[:, left], outer[:, right])))
            pooled_null = np.vstack([bank, cohorts.pop("validate")])
            rows, contrasts = roc_comparison(pooled_null, cohorts, design, design_index, n,
                                             20 if ctx.quick else 2000)
            roc_rows.extend(rows)
            roc_pairs.extend(contrasts)
    ctx.table("mixed_rates", rates)
    ctx.table("mixed_null_contrasts", pairs)
    ctx.table("roc_rates", roc_rows)
    ctx.table("roc_contrasts", roc_pairs)
