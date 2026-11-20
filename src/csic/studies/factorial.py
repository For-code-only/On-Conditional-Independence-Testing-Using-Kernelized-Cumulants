"""Four-configuration calibration, common-ridge selection and oracle diagnostics."""
from functools import partial
import json

import numpy as np

from . import selection
from .engine import CONFIG_IDS, evaluate_variants
from .oracle import evaluate_oracle
from .statistics import (apply_null_calibration, factorial_summary, fit_null_calibration,
                         holm, rate_summary)
from .streams import PAPER_PLAN, generate_sample, stable_seed, data_seed, pilot_sample


def settings(quick, conditional=False):
    designs = [("main_unconditional", 0), ("main_conditional", 2),
               ("main_conditional", 5), ("seoul", 5)]
    if quick:
        return [("main_conditional", 200, 5)]
    return [(study, n, d) for study, d in designs if d or not conditional for n in (200, 400, 800)]


def sample_job(setting, role, case, c0, rep, tau=1.):
    study, n, d = setting
    job = dict(study=study, n=n, d=d, role=role, case=case, c0=float(c0), rep=rep, tau=tau)
    sample = generate_sample(PAPER_PLAN, job)
    seed = stable_seed(PAPER_PLAN["seed_root"], "evaluation", data_seed(PAPER_PLAN, job))
    return sample, seed


def evaluate(rep, *, setting, role, case, c0, variants, tau):
    sample, seed = sample_job(setting, role, case, c0, rep, tau)
    values = evaluate_variants(sample, variants, seed=seed, B=4999, return_spectra=False)
    return [[values[v][config]["pvalue"] for config in CONFIG_IDS] for v in variants]


def group(run, stage, setting, role, case, c0, count, variants, tau=1.):
    key = "_".join(map(str, (stage, *setting, role, case, c0, tau)))
    fn = partial(evaluate, setting=setting, role=role, case=case, c0=c0, variants=variants, tau=tau)
    return run.scores(key, count, fn)


def row_key(setting, case, c0):
    study, n, d = setting
    return dict(study=study, n=n, d=d, case=case, c0=c0)


def run(ctx):
    rates, pairs, interactions, families = [], [], [], {"order": [], "interaction": []}
    variants = {"original": {"rho400": .01}}
    for setting in settings(ctx.quick):
        prefix = "C" if setting[2] else "U"
        bank = group(ctx, "factorial", setting, "calibration", prefix+"0", 0,
                     19 if ctx.quick else 49999, variants)[:, 0]
        banks = [fit_null_calibration(column) for column in bank.T]
        for role, suffix, c0, size in [("validation", "0", 0, 50000), ("power", "1", 2, 5000),
                                       ("power", "1", 8, 5000), ("power", "2", 8, 20000)]:
            case = prefix + suffix
            scores = group(ctx, "factorial", setting, role, case, c0, 3 if ctx.quick else size, variants)[:, 0]
            outer = np.column_stack([apply_null_calibration(scores[:, j], b)["reject"]
                                     for j, b in enumerate(banks)])
            for scale, decisions in (("raw", scores <= .05), ("outer", outer)):
                key = dict(**row_key(setting, case, c0), scale=scale)
                result = factorial_summary(decisions)
                for config, rate in result["rates"].items():
                    rates.append(dict(**key, config=config, **rate))
                for contrast, data in result["contrasts"].items():
                    row = dict(**key, contrast=contrast, **data, pvalue_holm="")
                    pairs.append(row)
                    if suffix == "2" and scale == "outer" and contrast in ("order_y1", "order_y2"):
                        families["order"].append(row)
                row = dict(**key, **result["interaction"], pvalue_holm="")
                row.pop("counts")
                interactions.append(row)
                if suffix == "2" and scale == "outer":
                    families["interaction"].append(row)
    if not ctx.quick:
        for family in families.values():
            for row, p in zip(family, holm([r["pvalue"] for r in family])):
                row["pvalue_holm"] = p
    ctx.table("rates", rates)
    ctx.table("paired_differences", pairs)
    ctx.table("interactions", interactions)


def development(ctx, stage, variants, grid):
    rows = []
    confirm = stage == "confirm"
    prefix = "development_confirm" if confirm else "development_screen"
    for setting in grid:
        cells = [("null", "C0", 0, 2000 if confirm else 500)]
        if setting[1] == (32 if ctx.quick else 400):
            cells.extend([("power", "C1", 2, 1000 if confirm else 500),
                          ("power", "C2", 8, 1000 if confirm else 500)])
        for role, case, c0, count in cells:
            role = prefix + "_" + role
            scores = group(ctx, stage, setting, role, case, c0, 2 if ctx.quick else count, variants)
            for v, variant in enumerate(variants):
                for j, config in enumerate(CONFIG_IDS):
                    reject = scores[:, v, j] <= .05
                    rows.append(dict(**row_key(setting, case, c0), role=role, config=config,
                                     variant=variant, replications=len(reject),
                                     rejections=int(reject.sum()), rate=float(reject.mean())))
    return rows


def ridge(ctx):
    grid = [(s, {200: 24, 400: 32, 800: 40}[n] if ctx.quick else n, d)
            for s, n, d in settings(False, conditional=True)]
    expected = [dict(study=s, n=n, d=d) for s,n,d in grid]
    anchors = [.0001, .001, .01, .1, 1.]
    screen = development(ctx, "screen", selection.variant_specs(anchors), grid)
    refined = selection.select_refinement(screen, anchors, expected_settings=expected)
    extra = refined["new_rhos"]
    screen.extend(development(ctx, "refine", selection.variant_specs(extra), grid))
    candidates = selection.rank_candidates(screen, anchors+extra, expected_settings=expected)
    variants = {"original": {"kind": "same_sample", "rho400": .01},
                **candidates["retained_variants"], **candidates["holdout_variants"]}
    confirmation = development(ctx, "confirm", variants, grid)
    chosen = selection.freeze_final(confirmation, variants, expected_settings=expected, baseline_id="original")
    ctx.table("development_rates", screen + confirmation)
    (ctx.path / "selection.json").write_text(json.dumps(chosen, indent=2)+"\n")
    if chosen["winner"] is None:
        return
    comparison = {"original": variants["original"], chosen["winner"]: chosen["winner_spec"]}
    rates = []
    for setting in grid:
        if setting[2] != 5:
            continue
        cases = [("validation", 1., 50000)]
        if setting[1] == 400:
            cases.append(("stress", 2., 5000))
        for role, tau, size in cases:
            scores = group(ctx, "comparison", setting, role, "C0", 0, 3 if ctx.quick else size, comparison, tau)
            for v, variant in enumerate(comparison):
                rates.append(dict(**row_key(setting, "C0", 0), tau=tau, variant=variant,
                                  config="second_y2", **rate_summary(scores[:, v, 3] <= .05)))
    ctx.table("ridge_rates", rates)


def oracle(ctx):
    study = "main_conditional"
    bandwidth = pilot_sample(PAPER_PLAN, study, 5, "oracle_bandwidth", 1024)
    # The fixed 256-point center is the prefix of a 512-observation reference stream.
    reference = pilot_sample(PAPER_PLAN, study, 5, "oracle_reference", 512)
    rates = []
    for n in ((200,) if ctx.quick else (200, 400, 800)):
        def one(rep):
            data, seed = sample_job((study, n, 5), "oracle_null", "C0", 0, rep)
            result = evaluate_oracle(data, bandwidth_sample=bandwidth, reference_sample=reference,
                                     reference_size=256, seed=seed)
            return [result[arm] for arm in ("fixed_reference", "oracle_both")]
        scores = ctx.scores(f"oracle_n{n}", 2 if ctx.quick else (2000 if n == 400 else 1000), one)
        for j, arm in enumerate(("fixed_reference", "oracle_both")):
            rates.append(dict(n=n, arm=arm, config="second_y2", **rate_summary(scores[:, j] <= .05)))
    ctx.table("oracle_rates", rates)
