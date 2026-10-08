"""Paired, resumable main experiments. Run with ``csic-run --help``."""
from __future__ import annotations

# Set before importing NumPy/SciPy, including in spawned workers.
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import platform
import traceback

import numpy as np
import scipy
from .inference import evaluate_sample
from .simulation import generate_job, load_seoul, paper_seed

STUDIES = ("main_unconditional", "main_conditional", "seoul")
_POOL = None


def _json(value):
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _bytes(value):
    return (json.dumps(_json(value), sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _save(path, value):
    """A successful record is written once; never replace a previous result."""
    path = Path(path)
    data = _bytes(value)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"Refusing to overwrite {path}")
        return
    temp = path.with_suffix(f".{os.getpid()}.tmp")
    with temp.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)


@contextmanager
def _lock(directory):
    """OS-managed lock releases even if a run is interrupted."""
    path = directory / ".run.lock"
    with path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            if path.stat().st_size == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("This output directory is already in use") from exc
        else:
            import fcntl
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("This output directory is already in use") from exc
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def make_jobs(studies, ns, ds, null_reps, alt_reps):
    jobs = []
    for study in studies:
        for n in ns:
            dimensions = [0] if study == "main_unconditional" else ds if study == "main_conditional" else [5]
            for d in dimensions:
                for case in (("U0", "U1", "U2") if d == 0 else ("C0", "C1", "C2")):
                    for rep in range(null_reps if case.endswith("0") else alt_reps):
                        jobs.append(dict(id=f"{study}_{case}_n{n}_d{d}_r{rep:04d}",
                                         study=study, case=case, cell=case,
                                         n=n, d=d, c0=8.0, rep=rep))
    return jobs


def input_digest(sample):
    digest = hashlib.sha256()
    for name in ("x", "y", "z", "row_indices"):
        if sample.get(name) is not None:
            array = np.ascontiguousarray(sample[name])
            digest.update(name.encode())
            digest.update(str(array.shape).encode())
            digest.update(array.dtype.str.encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


def _initialize(pool_path):
    global _POOL
    _POOL = load_seoul(pool_path) if pool_path else None


def evaluate_job(job, config, config_hash):
    sample = generate_job(job, seed_root=config["seed_root"], seoul_pool=_POOL)
    results = {}
    for cid in ("baseline", "proposed"):
        method = ("HSIC" if job["d"] == 0 else "KCI") if cid == "baseline" else ("CSIC" if job["d"] == 0 else "CSIC_CI")
        multiplier = 1.0 if cid == "baseline" else 2.0
        result = evaluate_sample(sample, method, seed=paper_seed(job, "mixture", method, root=config["seed_root"]),
                                 rho400=0.01 if job["d"] else None, B=config["B"], alpha=0.05,
                                 trace_rtol=1e-8, sigma_y_multiplier=multiplier)
        spectrum = dict(result["estimate"])
        spectrum.pop("weights")
        results[cid] = dict(method=method, statistic=spectrum["statistic"], spectrum=spectrum,
                            calibration=result["calibration"], metadata=result["metadata"],
                            y_multiplier=multiplier)
    return dict(job=job, configuration_sha256=config_hash,
                data_seed=paper_seed(job, "data", root=config["seed_root"]),
                input_sha256=input_digest(sample), results=results)


def _check_record(record, job, config, config_hash):
    if record["job"] != job or record["configuration_sha256"] != config_hash:
        raise RuntimeError(f"Mismatched checkpoint: {job['id']}")
    if record["data_seed"] != paper_seed(job, "data", root=config["seed_root"]):
        raise RuntimeError(f"Mismatched data seed: {job['id']}")
    if set(record["results"]) != {"baseline", "proposed"}:
        raise RuntimeError(f"Incomplete paired record: {job['id']}")
    fingerprint = record["input_sha256"]
    if not isinstance(fingerprint, str) or len(fingerprint) != 64 or any(c not in "0123456789abcdef" for c in fingerprint):
        raise RuntimeError(f"Invalid input fingerprint: {job['id']}")
    for cid, result in record["results"].items():
        method = ("HSIC" if job["d"] == 0 else "KCI") if cid == "baseline" else ("CSIC" if job["d"] == 0 else "CSIC_CI")
        cal, meta = result["calibration"], result["metadata"]
        if (result["method"] != method or meta["calibration_seed"] != paper_seed(job, "mixture", method, root=config["seed_root"])
                or meta["B_requested"] != config["B"] or not 0 <= cal["pvalue"] <= 1
                or cal["reject"] != (cal["pvalue"] <= 0.05)
                or meta["sigma_y_multiplier"] != (1.0 if cid == "baseline" else 2.0)
                or meta["rho400"] != (0.01 if job["d"] else None)):
            raise RuntimeError(f"Invalid method record: {job['id']}/{cid}")
        if (not np.isfinite(result["statistic"]) or result["statistic"] < 0
                or result["spectrum"]["statistic"] != result["statistic"]
                or meta["n"] != job["n"] or cal["alpha"] != 0.05
                or cal["B_requested"] != config["B"]
                or meta["source_sample"]["seed"] != record["data_seed"]):
            raise RuntimeError(f"Inconsistent statistic or metadata: {job['id']}/{cid}")
        branch = cal["calibration"]
        if branch == "spectral_mc_plus_one":
            tail = cal["tail_count"]
            if (cal["B"] != config["B"] or type(tail) is not int or not 0 <= tail <= cal["B"]
                    or cal["pvalue"] != (1 + tail) / (cal["B"] + 1)):
                raise RuntimeError(f"Inconsistent Monte Carlo calibration: {job['id']}/{cid}")
        elif branch in ("exact_degenerate", "exact_equal_weights_chi2"):
            if cal["B"] != 0 or cal["tail_count"] is not None:
                raise RuntimeError(f"Inconsistent exact calibration: {job['id']}/{cid}")
        else:
            raise RuntimeError(f"Unknown calibration: {job['id']}/{cid}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=("all", "unconditional", "conditional", "seoul"), default="all")
    parser.add_argument("--output", type=Path, default=Path("results/paper"))
    parser.add_argument("--n", nargs="+", type=int, default=[200, 400, 800])
    parser.add_argument("--d", nargs="+", type=int, default=[2, 5], help="Gaussian conditional dimensions")
    parser.add_argument("--null-reps", type=int, default=1000)
    parser.add_argument("--alt-reps", type=int, default=500)
    parser.add_argument("--B", type=int, default=4999, help="spectral Monte Carlo draws")
    parser.add_argument("--seed-root", type=int, default=20261008)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seoul-pool", type=Path, default=Path("data/seoul_weather.npy"))
    parser.add_argument("--quick", action="store_true", help="smoke run: n=40, D=2, two repeats per case, B=99")
    parser.add_argument("--retry-failed", action="store_true", help="retry incomplete jobs; retain all previous failure logs")
    args = parser.parse_args(argv)
    if args.quick:
        args.n, args.d, args.null_reps, args.alt_reps, args.B = [40], [2], 2, 2, 99
    if (min(args.n) < 2 or min(args.d) < 1 or min(args.null_reps, args.alt_reps, args.B, args.workers) < 1
            or args.B < 19 or args.seed_root < 0 or len(set(args.n)) != len(args.n) or len(set(args.d)) != len(args.d)):
        parser.error("Use positive counts/dimensions, B>=19, distinct n>=2 and a nonnegative seed root")
    studies = list(STUDIES) if args.study == "all" else [{"unconditional": STUDIES[0], "conditional": STUDIES[1], "seoul": STUDIES[2]}[args.study]]
    pool_path = args.seoul_pool.resolve() if "seoul" in studies else None
    if pool_path:
        load_seoul(pool_path)  # Verify before creating any run files.
    source_hashes = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                     for name in ("kernels.py", "inference.py", "simulation.py")}
    config = dict(schema=1, studies=studies, n=args.n, d=args.d, null_reps=args.null_reps,
                  alt_reps=args.alt_reps, B=args.B, seed_root=args.seed_root,
                  seed_namespace="confirmation_y2_v1", c0=8.0, alpha=0.05,
                  y_multipliers={"baseline": 1.0, "proposed": 2.0}, rho400=0.01,
                  trace_rtol=1e-8, quick=args.quick, source_sha256=source_hashes,
                  environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                                   system=platform.system(), machine=platform.machine()),
                  seoul_pool_sha256=hashlib.sha256(pool_path.read_bytes()).hexdigest() if pool_path else None)
    config_hash = hashlib.sha256(_bytes(config)).hexdigest()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    jobs = make_jobs(studies, args.n, args.d, args.null_reps, args.alt_reps)
    with _lock(output):
        _save(output / "configuration.json", config)
        records = output / "records"
        records.mkdir(exist_ok=True)
        failures = output / "failures"
        if failures.exists() and any(failures.iterdir()) and not args.retry_failed:
            raise RuntimeError("Previous failures are retained in failures/. Inspect them, then use --retry-failed if resolved.")
        expected = {job["id"] for job in jobs}
        unexpected = {p.stem for p in records.glob("*.json")} - expected
        if unexpected:
            raise RuntimeError("Output contains records outside the selected configuration")
        pending = []
        for job in jobs:
            path = records / (job["id"] + ".json")
            if path.exists():
                _check_record(json.loads(path.read_text()), job, config, config_hash)
            else:
                pending.append(job)
        done = len(jobs) - len(pending)
        print(f"{len(jobs)} paired datasets; {done} already complete; {len(pending)} remaining", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_initialize, initargs=(str(pool_path) if pool_path else None,)) as executor:
            iterator = iter(pending)
            active = {}
            def submit():
                job = next(iterator, None)
                if job is not None:
                    active[executor.submit(evaluate_job, job, config, config_hash)] = job
            for _ in range(min(args.workers * 2, len(pending))):
                submit()
            while active:
                finished, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in finished:
                    job = active.pop(future)
                    try:
                        record = future.result()
                        _check_record(record, job, config, config_hash)
                        _save(records / (job["id"] + ".json"), record)
                    except Exception:
                        failures.mkdir(exist_ok=True)
                        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                        _save(failures / f"{job['id']}_{stamp}.json", dict(job=job, traceback=traceback.format_exc(), configuration_sha256=config_hash))
                        for remaining in active:
                            remaining.cancel()
                        raise
                    done += 1
                    if done % 100 == 0 or done == len(jobs):
                        print(f"Completed {done}/{len(jobs)} paired datasets", flush=True)
                    submit()
        from .reporting import summarize
        summarize(output)
        _save(output / "completion.json", dict(computed=True, summarized=True, datasets=len(jobs),
                                               method_evaluations=2 * len(jobs), configuration_sha256=config_hash))
        print(f"Results and figures: {output / 'summary'}", flush=True)


if __name__ == "__main__":
    main()
