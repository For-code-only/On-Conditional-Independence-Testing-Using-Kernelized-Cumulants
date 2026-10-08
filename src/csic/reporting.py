"""Summarize paired main-study records and plot pointwise Wilson intervals.

CSV values are proportions; figures use percentages.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

Z = 1.959963984540054
CONFIGS = ("baseline", "proposed")
MAIN_STUDIES = ("main_unconditional", "main_conditional", "seoul")
KEY_FIELDS = ("study", "n", "d", "cell", "case")
RATE_FIELDS = (
    *KEY_FIELDS, "config", "method", "y_multiplier", "rho400", "replications",
    "rejections", "rate", "ci_low", "ci_high", "mcse", "failures",
)
PAIRED_FIELDS = (
    *KEY_FIELDS, "config", "reference", "replications", "first_only",
    "reference_only", "paired_difference", "mcse", "ci_low", "ci_high",
)


def wilson(k: int, n: int) -> tuple[float, float]:
    """Pointwise 95% Wilson interval, including exact boundary endpoints."""
    if type(n) is not int or type(k) is not int or n < 1 or not 0 <= k <= n:
        raise ValueError("Wilson counts require integers 0 <= k <= n and n > 0")
    p = k / n
    den = 1 + Z * Z / n
    center = (p + Z * Z / (2 * n)) / den
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return (0.0 if k == 0 else max(0.0, center - half),
            1.0 if k == n else min(1.0, center + half))


def paired(first: Sequence[bool], reference: Sequence[bool]) -> dict[str, Any]:
    """Paired rejection difference with sample-variance MCSE and normal CI.

    One pair gives None for MCSE and interval endpoints (blank CSV cells).
    """
    n = len(first)
    if n == 0 or len(reference) != n:
        raise ValueError("Paired samples must have the same positive length")
    if any(type(value) is not bool for value in (*first, *reference)):
        raise ValueError("Paired decisions must be booleans")
    plus = sum(a and not b for a, b in zip(first, reference))
    minus = sum(b and not a for a, b in zip(first, reference))
    delta = (plus - minus) / n
    se = math.sqrt(max(0.0, (plus + minus - n * delta * delta) / (n - 1)) / n) if n > 1 else None
    return dict(replications=n, first_only=plus, reference_only=minus,
                paired_difference=delta, mcse=se,
                ci_low=max(-1.0, delta - Z * se) if se is not None else None,
                ci_high=min(1.0, delta + Z * se) if se is not None else None)


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _result_info(result: Mapping[str, Any], config: str, study: str) -> tuple[str, float, float | None]:
    expected = ("HSIC", "CSIC") if study == "main_unconditional" else ("KCI", "CSIC_CI")
    method = result["method"]
    if method != expected[CONFIGS.index(config)]:
        raise ValueError(f"Unexpected method {method!r} for {study}/{config}")
    _number(result["statistic"], "statistic")
    calibration = _mapping(result["calibration"], "calibration")
    if type(calibration["reject"]) is not bool:
        raise ValueError("calibration.reject must be a boolean")
    pvalue = _number(calibration["pvalue"], "calibration.pvalue")
    if not 0 <= pvalue <= 1:
        raise ValueError("calibration.pvalue must be in [0, 1]")
    metadata = _mapping(result["metadata"], "metadata")
    multiplier = result.get("y_multiplier", metadata.get("sigma_y_multiplier"))
    multiplier = _number(multiplier, "Y bandwidth multiplier")
    if multiplier <= 0:
        raise ValueError("Y bandwidth multiplier must be positive")
    if "sigma_y_multiplier" in metadata:
        if _number(metadata["sigma_y_multiplier"], "metadata.sigma_y_multiplier") != multiplier:
            raise ValueError("Result and metadata Y bandwidth multipliers disagree")
    rho = metadata.get("rho400")
    if study == "main_unconditional":
        if rho is not None:
            raise ValueError("Unconditional records must not specify a conditional rho400")
    else:
        rho = _number(rho, "metadata.rho400")
        if rho <= 0:
            raise ValueError("metadata.rho400 must be positive")
    return method, multiplier, rho


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    """Write a CSV atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def summarize(output_dir: str | Path) -> dict[str, Any]:
    """Summarize valid saved pairs and plot rates using their actual counts."""
    output_dir = Path(output_dir)
    records_dir = output_dir / "records"
    if not records_dir.is_dir():
        raise FileNotFoundError(f"Records directory does not exist: {records_dir}")
    groups: dict[tuple[Any, ...], dict[int, dict[str, bool]]] = defaultdict(dict)
    settings: dict[tuple[Any, ...], tuple[Any, ...]] = {}
    amplitudes: dict[tuple[Any, ...], float] = {}
    ids: set[str] = set()
    identities: set[tuple[Any, ...]] = set()
    configuration_identities: set[Any] = set()
    for path in sorted(records_dir.glob("*.json")):
        try:
            record = _mapping(json.loads(path.read_text(encoding="utf-8")), "record")
            job = _mapping(record["job"], "job")
            study = job["study"]
            if study not in MAIN_STUDIES:
                raise ValueError(f"Unknown study {study!r}")
            job_id = job["id"]
            if not isinstance(job_id, str) or not job_id:
                raise ValueError("job.id must be a nonempty string")
            n = _integer(job["n"], "job.n", 1)
            d = _integer(job["d"], "job.d")
            rep = _integer(job["rep"], "job.rep")
            if (study == "main_unconditional") != (d == 0):
                raise ValueError("Unconditional studies require d=0; conditional studies require d>0")
            case = job["case"]
            prefix = "U" if study == "main_unconditional" else "C"
            if case not in (prefix + "0", prefix + "1", prefix + "2"):
                raise ValueError(f"Invalid case {case!r} for {study}")
            cell = job["cell"]
            if not isinstance(cell, str) or not cell:
                raise ValueError("job.cell must be a nonempty string")
            key = (study, n, d, cell, case)
            identity = (*key, rep)
            if job_id in ids or identity in identities:
                raise ValueError(f"Duplicate job ID or replication identity: {job_id}")
            ids.add(job_id)
            identities.add(identity)
            config_hash = record.get("configuration_sha256")
            if config_hash is not None and (not isinstance(config_hash, str) or not config_hash):
                raise ValueError("configuration_sha256 must be a nonempty string when present")
            configuration_identities.add(config_hash)
            if len(configuration_identities) > 1:
                raise ValueError("Records have mixed or partially missing configuration identities")
            c0 = _number(job["c0"], "job.c0")
            if key in amplitudes and amplitudes[key] != c0:
                raise ValueError("A cell contains different c0 values")
            amplitudes[key] = c0
            results = _mapping(record["results"], "results")
            if set(results) != set(CONFIGS):
                raise ValueError("Each record must contain exactly baseline and proposed results")
            decisions = {}
            for config in CONFIGS:
                result = _mapping(results[config], f"results.{config}")
                setting = _result_info(result, config, study)
                setting_key = (*key, config)
                if setting_key in settings and settings[setting_key] != setting:
                    raise ValueError(f"Inconsistent method settings within {key}/{config}")
                settings[setting_key] = setting
                decisions[config] = result["calibration"]["reject"]
            groups[key][rep] = decisions
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid record {path}: {error}") from error
    unresolved = set()
    for path in sorted((output_dir / "failures").glob("*.json")):
        try:
            failure = _mapping(json.loads(path.read_text(encoding="utf-8")), "failure")
            failed_job = _mapping(failure["job"], "failure.job")
            failed_id = failed_job["id"]
            if not isinstance(failed_id, str) or not failed_id:
                raise ValueError("failure.job.id must be a nonempty string")
            if failed_id not in ids:
                unresolved.add(failed_id)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid failure log {path}: {error}") from error
    if unresolved:
        raise ValueError(f"Cannot summarize: unresolved failed jobs: {', '.join(sorted(unresolved))}")
    if not groups:
        raise ValueError(f"No main-study records found in {records_dir}")
    rates, pairs = [], []
    for key, rep_results in sorted(groups.items()):
        info = dict(zip(KEY_FIELDS, key))
        decisions = {config: [rep_results[rep][config] for rep in sorted(rep_results)] for config in CONFIGS}
        for config in CONFIGS:
            nrep = len(decisions[config])
            count = sum(decisions[config])
            lo, hi = wilson(count, nrep)
            method, multiplier, rho = settings[(*key, config)]
            rate = count / nrep
            rates.append(dict(**info, config=config, method=method, y_multiplier=multiplier, rho400=rho,
                              replications=nrep, rejections=count, rate=rate, ci_low=lo, ci_high=hi,
                              mcse=math.sqrt(rate * (1 - rate) / nrep), failures=0))
        pairs.append(dict(**info, config="proposed", reference="baseline",
                          **paired(decisions["proposed"], decisions["baseline"])))
    summary_dir = output_dir / "summary"
    rates_path, paired_path = summary_dir / "rates.csv", summary_dir / "paired_differences.csv"
    _write_csv(rates_path, RATE_FIELDS, rates)
    _write_csv(paired_path, PAIRED_FIELDS, pairs)
    figures = plot(rates_path, summary_dir)
    return {"rates": rates_path, "paired_differences": paired_path, "figures": figures,
            "datasets": len(ids)}


def plot(rates_path: str | Path, output_dir: str | Path) -> list[Path]:
    """Plot an existing rates CSV; skip studies not present and use actual n values."""
    rates_path, output_dir = Path(rates_path), Path(output_dir)
    rows: list[dict[str, Any]] = []
    identities: set[tuple[Any, ...]] = set()
    with rates_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {*KEY_FIELDS, "config", "rate", "ci_low", "ci_high"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"Rates CSV is missing columns: {sorted(required - set(reader.fieldnames or []))}")
        for line, raw in enumerate(reader, start=2):
            try:
                row = dict(raw)
                if row["study"] not in MAIN_STUDIES or row["config"] not in CONFIGS:
                    raise ValueError("Unknown study or configuration")
                row["n"], row["d"] = int(row["n"]), int(row["d"])
                if row["n"] < 1 or row["d"] < 0:
                    raise ValueError("Invalid n or d")
                if (row["study"] == "main_unconditional") != (row["d"] == 0):
                    raise ValueError("Study and d do not agree")
                prefix = "U" if row["study"] == "main_unconditional" else "C"
                if row["case"] not in (prefix + "0", prefix + "1", prefix + "2"):
                    raise ValueError("Invalid case")
                for field in ("rate", "ci_low", "ci_high"):
                    row[field] = float(row[field])
                    if not math.isfinite(row[field]) or not 0 <= row[field] <= 1:
                        raise ValueError(f"{field} must be a finite proportion")
                if not row["ci_low"] <= row["rate"] <= row["ci_high"]:
                    raise ValueError("Confidence interval must contain the rate")
                # More than one cell at the same plotted coordinate would merge
                # different experiments into one line; require separate CSVs.
                identity = tuple(row[k] for k in ("study", "n", "d", "case", "config"))
                if identity in identities:
                    raise ValueError("Duplicate plotted coordinate (possibly different cells)")
                identities.add(identity)
                rows.append(row)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid rates CSV {rates_path}, line {line}: {error}") from error
    if not rows:
        raise ValueError(f"No main-study rates in {rates_path}")
    # No global pyplot state or GUI backend is needed for batch commands.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for study, name in zip(MAIN_STUDIES, ("unconditional", "conditional", "seoul")):
        study_rows = [r for r in rows if r["study"] == study]
        if not study_rows:
            continue
        dimensions = sorted({r["d"] for r in study_rows})
        if study == "main_conditional":
            dimensions = sorted(set(dimensions) | {2, 5})
        figure = Figure(figsize=(12, 3.4 * len(dimensions) + 0.6))
        FigureCanvasAgg(figure)
        axes = figure.subplots(len(dimensions), 3, squeeze=False)
        handles: dict[str, Any] = {}
        for i, d in enumerate(dimensions):
            dimension_rows = [r for r in study_rows if r["d"] == d]
            ns = sorted({r["n"] for r in dimension_rows})
            for j, title in enumerate(("Null", "Mean", "Variance")):
                ax = axes[i, j]
                case = ("U" if study == "main_unconditional" else "C") + str(j)
                present = False
                for config, color, marker in (("baseline", "#1f77b4", "o"), ("proposed", "#ff7f0e", "s")):
                    subset = sorted((r for r in dimension_rows if r["case"] == case and r["config"] == config), key=lambda r: r["n"])
                    if not subset:
                        continue
                    present = True
                    label = (("HSIC" if config == "baseline" else "CSIC") if study == "main_unconditional"
                             else ("KCI-type" if config == "baseline" else "CSIC–CI"))
                    values = [100 * r["rate"] for r in subset]
                    errors = [[100 * (r["rate"] - r["ci_low"]) for r in subset],
                              [100 * (r["ci_high"] - r["rate"]) for r in subset]]
                    handle = ax.errorbar([r["n"] for r in subset], values, yerr=errors,
                                         label=label, color=color, marker=marker, linewidth=1.7,
                                         capsize=3, markersize=4)
                    handles.setdefault(label, handle)
                heading = "Unconditional" if study == "main_unconditional" else f"{'Gaussian' if study == 'main_conditional' else 'Seoul'} D={d}"
                ax.set_title(f"{heading}: {title}")
                ax.set_xlabel("Sample size n")
                ax.set_ylabel("Rejection rate (%)")
                if ns:
                    ax.set_xticks(ns)
                ax.grid(alpha=0.2)
                if j == 0:
                    ax.axhline(5, color="#555555", linestyle="--", linewidth=1)
                    ax.set_ylim(0, min(103, max(12, ax.get_ylim()[1])))
                else:
                    ax.set_ylim(0, 103)
                if not present:
                    ax.text(0.5, 0.5, "No observations", ha="center", va="center", transform=ax.transAxes)
        figure.legend(list(handles.values()), list(handles), loc="upper center", ncol=2, frameon=False)
        figure.text(0.5, 0.015, "Error bars: pointwise 95% Wilson intervals", ha="center", fontsize=9)
        figure.tight_layout(rect=(0, 0.045, 1, 0.94))
        try:
            for extension in ("png", "pdf"):
                target = output_dir / f"{name}.{extension}"
                figure.savefig(target, dpi=180)
                outputs.append(target)
        finally:
            figure.clear()
    return outputs


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Summarize paired CSIC main-study records.")
    parser.add_argument("output_dir", type=Path, help="Run directory containing records/*.json")
    args = parser.parse_args(argv)
    result = summarize(args.output_dir)
    print(f"Summarized {result['datasets']} paired datasets into {result['rates'].parent}")


def plot_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot CSIC rate CSV with pointwise Wilson intervals.")
    parser.add_argument("--rates", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    for path in plot(args.rates, args.output):
        print(path)


if __name__ == "__main__":
    main()
