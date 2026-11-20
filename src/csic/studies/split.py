"""Independent split development and frozen-split evaluation."""
import hashlib

import numpy as np

from ..simulation import generate, nuisance
from .streams import PAPER_PLAN, canonical, generate_sample, seoul_pool
from .split_kernel import CONFIGURATIONS, BETA_GRID, choose_beta, evaluate_configurations, evaluate_split
from .statistics import paired_summary, rate_summary, holm

DEVELOPMENT_NAMESPACE = "splitkci_supplement_v1_development"
DEVELOPMENT_SEED_ROOT = 20261113


def seed(*parts):
    digest = hashlib.sha256(canonical([20261113, "splitkci_supplement_v1", *parts]))
    return int(digest.hexdigest()[:16], 16)


def development_sample(family, n, rep):
    """One of 60 independent standard-C0 development datasets, shared by configs."""
    if family not in ("gaussian", "seoul") or n not in (200, 400, 800):
        raise ValueError("Unknown development setting")
    if type(rep) is not int or not 0 <= rep < 10:
        raise ValueError("Exactly ten development datasets per setting are prescribed")
    tokens = [DEVELOPMENT_SEED_ROOT, DEVELOPMENT_NAMESPACE, "data", family, n, rep]
    seed = int.from_bytes(hashlib.sha256(canonical(tokens)).digest()[:8], "little")
    if family == "gaussian":
        sample = generate("C0", n, seed, d=5, c0=0.)
    else:
        rng = np.random.default_rng(seed)
        ix = rng.integers(0, 8760, size=n, dtype=np.int64)
        z = seoul_pool()[ix]
        r, epsilon = rng.normal(size=(2, n))
        f, g = nuisance(z)
        sample = dict(x=f+r, y=g+epsilon, z=z, row_indices=ix.astype(np.int16),
                      metadata=dict(case="C0", n=n, d=5, a=0., c0=0.))
    return sample


def develop(ctx, *, sizes=None, diagnostics=None):
    """Select fractions from the independent development stream."""
    if sizes is None:
        sizes = (200,) if ctx.quick else (200, 400, 800)
    if diagnostics is None:
        diagnostics = 2 if ctx.quick else 100
    rows, fractions = [], {}
    for family in ("gaussian", "seoul"):
        for n in sizes:
            flags = {}
            for beta in BETA_GRID:
                def replicate(rep):
                    dataset, split = divmod(rep, 10)
                    sample = development_sample(family, n, dataset)
                    result = evaluate_split(sample["x"], sample["y"], sample["z"], beta,
                                            seed("development", "split", family, n, dataset, split),
                                            seed("development", "sign", family, n, dataset, split),
                                            B=999, diagnostic=True)
                    return [result[c]["both_reject"] for c in CONFIGURATIONS]
                flags[beta] = ctx.scores(f"development_{family}_{n}_{beta}", diagnostics, replicate)
            for j, config in enumerate(CONFIGURATIONS):
                chosen = choose_beta({b: v[:, j] for b, v in flags.items()},
                                     expected_diagnostics=diagnostics)
                fractions[family, n, config] = chosen["beta"]
                diagnostic = chosen["diagnostics"][str(chosen["beta"])]
                rows.append(dict(family=family, n=n, config=config, beta=chosen["beta"],
                                 flags=diagnostic["both_reject_count"], diagnostics=diagnostic["diagnostics"]))
    ctx.table("split_fractions", rows)
    return fractions


def evaluate(ctx):
    sizes = (400,) if ctx.quick else (200, 400, 800)
    # Preserve the selected fractions for each evaluated sample size even in a short replay.
    fractions = develop(ctx, sizes=sizes, diagnostics=100)
    rates, comparisons = [], []
    for family in ("gaussian", "seoul"):
        study = "main_conditional" if family == "gaussian" else "seoul"
        for n in sizes:
            betas = {c: fractions[family, n, c] for c in CONFIGURATIONS}
            cells = [("C0", "validation", "C0", 0., 1., 20000), ("C2", "power", "C2", 8., 1., 10000)]
            if n == 400:
                cells += [("weakC1", "power", "C1", 2., 1., 5000), ("stressC0", "stress", "C0", 0., 2., 20000)]
            for name, role, case, strength, tau, count in cells:
                scenario = f"{family}_n{n}_{name}"
                def replicate(rep):
                    job = dict(study=study, n=n, d=5, role=role, case=case, c0=strength, tau=tau, rep=rep)
                    sample = generate_sample(PAPER_PLAN, job)
                    result = evaluate_configurations(sample["x"], sample["y"], sample["z"], betas,
                                                     seed("evaluation", "split", scenario, rep),
                                                     seed("evaluation", "sign", scenario, rep), B=4999)
                    return [result[c]["pvalue"] for c in CONFIGURATIONS]
                decisions = ctx.scores(scenario, 2 if ctx.quick else count, replicate) <= .05
                for j, config in enumerate(CONFIGURATIONS):
                    rates.append(dict(scenario=scenario, config=config, **rate_summary(decisions[:, j])))
                contrasts = [(1, 0), (3, 2), (3, 0)]
                if name == "weakC1":
                    contrasts = contrasts[:2]
                for left, right in contrasts:
                    comparisons.append(dict(scenario=scenario, cohort=name, first=CONFIGURATIONS[left],
                                            second=CONFIGURATIONS[right], pvalue_holm="",
                                            **paired_summary(decisions[:, left], decisions[:, right])))
    if not ctx.quick:
        for name in ("C2", "weakC1"):
            family = [r for r in comparisons if r["cohort"] == name]
            for row, adjusted in zip(family, holm([r["pvalue"] for r in family])):
                row["pvalue_holm"] = adjusted
    ctx.table("split_rates", rates)
    ctx.table("split_contrasts", comparisons)
