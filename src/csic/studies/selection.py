"""Deterministic development decisions, independent of formal evaluation.

Inputs are per-configuration rejection-rate records with keys ``variant``,
``study``, ``n``, ``d``, ``case``, ``c0``, ``config`` and ``rate``.  ``role``
is optional, but mixed null roles and duplicate records are rejected.  These
functions perform no I/O and never pool or average power over configurations.
The runner is responsible for finishing the frozen development-stage inventory
before calling them; all supplied power records are retained for reporting.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral

from .engine import CONFIG_IDS, normalize_variant

BASELINE_RHO = .01
NEAR_TIE = .005
_EPS = 1e-12


def _number_id(value):
    value = float(value)
    return str(int(value)) if value.is_integer() else repr(value)


def rho_id(rho):
    """Round-trip stable ID for a common ridge anchor (e.g. rho_0.01)."""
    spec = normalize_variant({"rho400": rho})
    return "rho_" + _number_id(spec["rho400"])


def variant_id(spec):
    spec = normalize_variant(spec)
    result = rho_id(spec["rho400"])
    if spec["kind"] == "holdout":
        result = "holdout_" + _number_id(spec["train_fraction"]) + "_" + result
    return result


def variant_specs(rhos, *, holdout=False, fractions=(.5, .75)):
    """Return an ID-to-engine-spec mapping; repeated anchors are merged."""
    result = {}
    for rho in rhos:
        if holdout:
            specs = [dict(kind="holdout", rho400=rho, train_fraction=f)
                     for f in fractions]
        else:
            specs = [dict(kind="same_sample", rho400=rho)]
        for spec in specs:
            spec = normalize_variant(spec)
            result[variant_id(spec)] = spec
    return result


def _setting(value):
    if isinstance(value, Mapping):
        study, n, d = value["study"], value["n"], value["d"]
    else:
        study, n, d = value
    if (not isinstance(study, str) or isinstance(n, bool)
            or not isinstance(n, Integral) or n < 2 or isinstance(d, bool)
            or not isinstance(d, Integral)):
        raise ValueError("Settings require study, integer n >= 2 and integer d")
    return study, int(n), int(d)


def _expected_grid(rows, expected_settings):
    values = ([_setting(row) for row in rows if row["case"] == "C0"]
              if expected_settings is None
              else [_setting(s) for s in expected_settings])
    expected = set(values)
    sizes = {s[1] for s in expected}
    correct = {(study, n, d) for n in sizes for study, d in
               (("main_conditional", 2), ("main_conditional", 5), ("seoul", 5))}
    if len(sizes) != 3 or len(expected) != 9 or expected != correct:
        raise ValueError("Development requires all nine conditional settings: "
                         "three sample sizes by Gaussian D=2,5 and Seoul D=5")
    if expected_settings is not None and len(values) != 9:
        raise ValueError("expected_settings must contain nine unique settings")
    return expected


def _records(rate_rows, variants):
    rows = []
    for record in rate_rows:
        if not isinstance(record, Mapping):
            raise TypeError("rate_rows must contain mapping records")
        if record.get("variant") not in variants:
            continue
        row = dict(record)
        for name in ("study", "n", "d", "case", "c0", "config", "rate"):
            if name not in row:
                raise ValueError(f"Missing field {name} in rate record")
        row["study"], row["n"], row["d"] = _setting(row)
        row["rate"], row["c0"] = float(row["rate"]), float(row["c0"])
        if not math.isfinite(row["rate"]) or not 0 <= row["rate"] <= 1:
            raise ValueError("Rates must be finite probabilities; failures are not nonrejections")
        if not math.isfinite(row["c0"]):
            raise ValueError("Signal strength must be finite")
        if row["config"] not in CONFIG_IDS:
            raise ValueError(f"Unknown factorial configuration: {row['config']}")
        if row["case"] == "C0" and row["c0"] != 0:
            raise ValueError("C0 selection records must have c0=0")
        if row.get("tau", 1.) != 1.:
            raise ValueError("Stress-test records cannot be used for development selection")
        rows.append(row)
    return rows


def _row_key(row):
    return (row["variant"], row["study"], row["n"], row["d"],
            row["case"], row["c0"], row["config"], str(row.get("role", "")))


def _summaries(rate_rows, variants, expected_settings):
    if not variants:
        raise ValueError("At least one variant is required")
    variants = {str(key): normalize_variant(spec) for key, spec in variants.items()}
    rows = _records(rate_rows, variants)
    expected = _expected_grid(rows, expected_settings)
    expected_keys = {(s, config) for s in expected for config in CONFIG_IDS}
    roles = {row.get("role") for row in rows if row["case"] == "C0"}
    if len(roles) != 1:
        raise ValueError("Use a single development null role; do not mix screening and confirmation")
    summaries = []
    for key, spec in variants.items():
        null = [row for row in rows if row["variant"] == key and row["case"] == "C0"]
        seen = set()
        for row in null:
            cell = (_setting(row), row["config"])
            if cell in seen:
                raise ValueError(f"Duplicate null cell for {key}: {cell}")
            seen.add(cell)
        if seen != expected_keys:
            raise ValueError(f"Incomplete null grid for {key}: "
                             f"{len(expected_keys-seen)} missing, {len(seen-expected_keys)} unexpected cells")
        rates = [row["rate"] for row in null]
        bias = max(abs(rate-.05) for rate in rates)
        summaries.append(dict(variant=key, spec=spec, worst_null_bias=bias,
                              min_null_rate=min(rates), max_null_rate=max(rates),
                              within_screen_bounds=all(.025-_EPS <= r <= .075+_EPS for r in rates),
                              null_cells=len(null), null_rows=sorted(null, key=_row_key)))
    power = sorted([row for row in rows if row["case"] != "C0"], key=_row_key)
    return summaries, power, sorted(expected)


def _preference(summary):
    spec = summary["spec"]
    return (spec["kind"] != "same_sample",
            round(abs(math.log(spec["rho400"]/BASELINE_RHO)), 14), spec["rho400"],
            spec.get("train_fraction", 0.), summary["variant"])


def _rank(summaries):
    """At each rank, prefer simpler candidates within .005 of the best bias."""
    remaining = list(summaries)
    result = []
    while remaining:
        best_bias = min(row["worst_null_bias"] for row in remaining)
        near = [row for row in remaining if row["worst_null_bias"] <= best_bias+NEAR_TIE+_EPS]
        chosen = min(near, key=_preference)
        result.append(dict(chosen, rank=len(result)+1,
                           preference_band_best_bias=best_bias))
        remaining.remove(chosen)
    return result


def select_refinement(rate_rows, coarse_grid, *, expected_settings=None):
    """Choose one coarse anchor and at most two adjacent geometric midpoints."""
    variants = variant_specs(coarse_grid)
    if len(variants) < 2:
        raise ValueError("Refinement requires at least two distinct coarse anchors")
    summaries, power, expected = _summaries(rate_rows, variants, expected_settings)
    ranking = _rank(summaries)
    best = ranking[0]
    grid = sorted(spec["rho400"] for spec in variants.values())
    index = grid.index(best["spec"]["rho400"])
    neighbors = ([grid[index-1]] if index else []) + ([grid[index+1]] if index+1 < len(grid) else [])
    # Log-space midpoint avoids overflowing the product of finite anchors.
    new_rhos = sorted(math.exp((math.log(best["spec"]["rho400"])+math.log(other))/2)
                      for other in neighbors)
    minimum_bias = min(row["worst_null_bias"] for row in summaries)
    minimum_bias_rhos = sorted(row["spec"]["rho400"] for row in summaries
                              if abs(row["worst_null_bias"]-minimum_bias) <= _EPS)
    boundary_rhos = sorted(({best["spec"]["rho400"]} | set(minimum_bias_rhos))
                           & {grid[0],grid[-1]})
    boundary = bool(boundary_rhos)
    return dict(best=best["variant"], best_rho=best["spec"]["rho400"],
                new_rhos=new_rhos, new_variants=variant_specs(new_rhos),
                boundary_unresolved=boundary, boundary_rhos=boundary_rhos,
                minimum_bias_rhos=minimum_bias_rhos, ranking=ranking, power_rows=power,
                expected_settings=[dict(study=s,n=n,d=d) for s,n,d in expected],
                reason=("A selected or minimum-bias anchor is on the search boundary; optimality is unresolved"
                        if boundary else "Refine only the two adjacent coarse intervals"),
                near_tie_tolerance=NEAR_TIE)


def rank_candidates(rate_rows, rhos, *, expected_settings=None):
    """Retain at most two common anchors and define the finite holdout search."""
    variants = variant_specs(rhos)
    summaries, power, expected = _summaries(rate_rows, variants, expected_settings)
    ranking = _rank(summaries)
    best = ranking[0]
    retained = [row["variant"] for row in ranking[:2]]
    return dict(best=best["variant"], best_rho=best["spec"]["rho400"],
                retained=retained, retained_variants={key: variants[key] for key in retained},
                holdout_variants=variant_specs([BASELINE_RHO,best["spec"]["rho400"]],holdout=True),
                ranking=ranking, power_rows=power,
                expected_settings=[dict(study=s,n=n,d=d) for s,n,d in expected],
                near_tie_tolerance=NEAR_TIE,
                reason="Retain at most two common anchors; report each supplied power cell")


def freeze_final(rate_rows, variants, *, expected_settings=None, baseline_id=None):
    """Freeze zero or one practical improvement using independent confirmation.

    Eligibility requires every null point estimate in [.025,.075] and a
    worst-bias improvement of at least .005 over the original same-sample .01
    configuration. Power is retained configuration by configuration, never used
    as an average noninferiority gate. These are development rules, not formal
    level-validity claims.
    """
    variants = {str(key): normalize_variant(spec) for key, spec in variants.items()}
    baseline_spec = normalize_variant({"rho400": BASELINE_RHO})
    original_ids = [key for key,spec in variants.items() if spec == baseline_spec]
    if baseline_id is None:
        if not original_ids:
            raise ValueError("Independent confirmation must include the original .01 configuration")
        baseline_id = min(original_ids, key=lambda key: (key != "original", key != rho_id(.01), key))
    if baseline_id not in variants or variants[baseline_id] != baseline_spec:
        raise ValueError("baseline_id must identify the original same-sample .01 configuration")
    summaries, power, expected = _summaries(rate_rows, variants, expected_settings)
    baseline = next(row for row in summaries if row["variant"] == baseline_id)
    role = baseline["null_rows"][0].get("role")
    if role is not None and role != "development_confirm_null":
        raise ValueError("Final freezing requires independent development_confirm_null records")
    eligible = []
    for row in summaries:
        improvement = baseline["worst_null_bias"]-row["worst_null_bias"]
        reasons = []
        if row["spec"] == baseline_spec:
            reasons.append("original_configuration")
        if not row["within_screen_bounds"]:
            reasons.append("null_rate_outside_0.025_0.075")
        if improvement < NEAR_TIE-_EPS:
            reasons.append("worst_bias_improvement_below_0.005")
        row.update(improvement_over_baseline=improvement, eligible=not reasons,
                   ineligibility_reasons=reasons)
        if not reasons:
            eligible.append(row)
    eligible_ranking = _rank(eligible)
    winner = eligible_ranking[0] if eligible_ranking else None
    return dict(winner=None if winner is None else winner["variant"],
                winner_spec=None if winner is None else winner["spec"],
                baseline=baseline_id, baseline_worst_bias=baseline["worst_null_bias"],
                ranking=_rank(summaries), eligible_ranking=eligible_ranking,
                power_rows=power,
                expected_settings=[dict(study=s,n=n,d=d) for s,n,d in expected],
                near_tie_tolerance=NEAR_TIE,
                reason=("No candidate meets every development size bound and the minimum worst-bias improvement"
                        if winner is None else "One practical candidate frozen after independent confirmation"),
                interpretation="Development selection only; independent formal evaluation determines conclusions")
