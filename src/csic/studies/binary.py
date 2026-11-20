"""Paired prevalence experiment and unit-weight mixed-kernel comparison."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from . import binary_kernel as core
from . import binary_law as law
from .statistics import fit_null_calibration, rate_summary, paired_summary


def mixed_score(x, y, z, seed, second_weight=1.):
    n = len(y)
    sigma_x, _, _ = core.median_bandwidth(x)
    sigma_y, _, _ = core.median_bandwidth(y)
    kx = core.rbf(x, sigma_x)
    kz = (1.0 + core.rbf(z, np.sqrt(z.shape[1]))) / 2
    ridge = .01
    identity = np.eye(n)
    ridge_factor = cho_factor(kz + ridge * identity, lower=True)
    residual = core.matrix(ridge * cho_solve(ridge_factor, identity))
    left = core.matrix(residual @ core.center_matrix(kx * kz) @ residual)

    first_gram = core.center_matrix(core.rbf(y, sigma_y))
    second_base = core.center_matrix(core.rbf(y, 2 * sigma_y))
    second_gram = core.center_matrix(second_base ** 2)
    mixed_gram = first_gram + second_weight * second_gram
    right = core.matrix(residual @ mixed_gram @ residual)
    statistic = float(np.sum(left * right) / n)
    eigenvalues, _ = core.checked_eigh(core.center_matrix(left * right) / n)
    weights, _ = core.trim_weights(eigenvalues, trace_rtol=1e-8)
    return core.mixture_pvalue(statistic, weights, seed, 4999)["pvalue"]


def paired_block(stage, rep):
    block = law.generate_block(stage, rep)
    row = []
    for pid in ("p025", "p049"):
        sample = law.sample_from_block(block, pid, stage)
        prepared = core.prepare_sample(sample)
        row.append(int(sample["y"].sum()))
        for config in law.CONFIGS:
            seed = law.spectral_seed(stage, rep, pid, config)
            result = core.evaluate_prepared(prepared, config, seed=seed, B=4999)
            row.append(result["calibration"]["pvalue"])
        if pid == "p049":
            mixed = mixed_score(sample["x"], sample["y"], sample["z"],
                                law.spectral_seed(stage, rep, pid, "first_y1"))
            if mixed != row[-2]:
                raise ArithmeticError("Unit-weight mixed score disagrees with the first-order score")
            row.append(mixed)
    return row


def run(ctx):
    stages = {}
    for stage, full_count in law.STAGES.items():
        small = 19 if stage == "null_calibration" else 3
        stages[stage] = ctx.scores(stage, small if ctx.quick else full_count,
                                   lambda rep: paired_block(stage, rep))
    labels = ("p025_first", "p025_second", "p049_first", "p049_second", "p049_mixed")
    cols = [1, 2, 4, 5, 6]
    banks = [fit_null_calibration(stages["null_calibration"][:, col]) for col in cols]
    cutoffs = np.asarray([b["cutoff"] for b in banks])
    thresholds, rates, paired, by_count = [], [], [], []
    for j, method in enumerate(labels):
        stability = banks[j]["stability"]
        thresholds.append(dict(method=method, cutoff=cutoffs[j], rank=banks[j]["rank_one_based"],
                               **stability))
    for stage in ("null_validation", "alternative"):
        scores = stages[stage][:, cols]
        for scale, decisions in (("raw", scores <= .05), ("outer", scores < cutoffs)):
            for j, method in enumerate(labels):
                rates.append(dict(stage=stage, method=method, scale=scale, **rate_summary(decisions[:, j])))
            for left, right in ((1, 0), (3, 2), (2, 0), (3, 1)):
                paired.append(dict(stage=stage, scale=scale, left=labels[left], right=labels[right],
                                   **paired_summary(decisions[:, left], decisions[:, right])))
        for j, method in enumerate(labels):
            for endpoint in ("lower_score", "upper_score"):
                value = banks[j]["stability"][endpoint]
                if value is not None:
                    rates.append(dict(stage=stage, method=method, scale=endpoint,
                                      **rate_summary(scores[:, j] < value)))
        for j in (2, 3, 4):
            for balance in (False, True):
                mask = (stages[stage][:, 3] == 200) == balance
                by_count.append(dict(stage=stage, balanced=balance, method=labels[j], n=int(mask.sum()),
                                     rejections=int(np.sum(scores[mask, j] < cutoffs[j]))))
    ctx.table("binary_thresholds", thresholds)
    ctx.table("binary_rates", rates)
    ctx.table("binary_contrasts", paired)
    ctx.table("binary_counts", by_count)


def balanced(ctx):
    from .balanced import run_one
    rows = []
    for stage, size in [("null_calibration", 1999), ("null_validation", 2000), ("alternative", 2000)]:
        small = 19 if stage == "null_calibration" else 3
        values = ctx.scores(stage, small if ctx.quick else size, lambda rep: run_one(stage, rep)[2:6])
        scores = values[:, 1:]
        if not np.array_equal(scores[:, 0], scores[:, 2]):
            raise ArithmeticError("Shared first and mixed spectral scores differ")
        if stage == "null_calibration":
            cutoffs = np.array([fit_null_calibration(col)["cutoff"] for col in scores.T])
            continue
        for subgroup, mask in (("all", np.ones(len(scores), bool)), ("balanced", values[:, 0] == 200),
                               ("unbalanced", values[:, 0] != 200)):
            for j, method in enumerate(("first", "second", "mixed")):
                rows.append(dict(stage=stage, subgroup=subgroup, method=method, n=int(mask.sum()),
                                 cutoff=cutoffs[j], raw=int(np.sum(scores[mask, j] <= .05)),
                                 calibrated=int(np.sum(scores[mask, j] < cutoffs[j]))))
    ctx.table("balanced_counts", rows)
