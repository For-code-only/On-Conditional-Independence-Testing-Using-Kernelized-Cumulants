"""Reproduce one supplementary study from the paper."""

import argparse
import csv
import hashlib
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Each worker performs its own dense linear algebra.
for variable in ("OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import scipy


def write_csv(path, rows):
    if not rows:
        return
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(stream.name, path)


class Run:
    """Store consecutive numeric rows in atomic batches for restartable simulations."""

    def __init__(self, path, quick, executor):
        self.path = path
        self.quick = quick
        self.executor = executor

    def scores(self, key, count, evaluate):
        directory = self.path / "batches" / key
        directory.mkdir(parents=True, exist_ok=True)
        chunks = []
        for start in range(0, count, 32):
            stop = min(start + 32, count)
            path = directory / f"{start:06d}.npy"
            if path.exists():
                rows = np.load(path, allow_pickle=False)
            else:
                rows = np.asarray(list(self.executor.map(evaluate, range(start, stop))), dtype=float)
                if not np.isfinite(rows).all():
                    raise ArithmeticError(f"Nonfinite result in {key} at {start}")
                with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
                    np.save(stream, rows, allow_pickle=False)
                os.replace(stream.name, path)
            if len(rows) != stop - start or not np.isfinite(rows).all():
                raise ValueError(f"Invalid checkpoint {path}")
            chunks.append(rows)
        return np.concatenate(chunks)

    def table(self, name, rows):
        write_csv(self.path / f"{name}.csv", rows)


def main(argv=None):
    from . import factorial, split, binary, mixed
    studies = {"factorial": factorial.run, "ridge": factorial.ridge, "oracle": factorial.oracle,
               "split-development": split.develop, "split": split.evaluate, "binary": binary.run,
               "balanced": binary.balanced, "weight": mixed.pilot, "mixed": mixed.run}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=studies)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--quick", action="store_true", help="Run a reduced example")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    path = args.output.resolve()
    path.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[3]
    sources = sorted((root / "src/csic").rglob("*.py"))
    stamp = dict(study=args.study, quick=args.quick, numpy=np.__version__, scipy=scipy.__version__,
                 files={p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sources})
    # A hard link creates the manifest atomically even for overlapping invocations.
    manifest = path / "run.json"
    with tempfile.NamedTemporaryFile("w", dir=path, delete=False) as stream:
        json.dump(stamp, stream, indent=2)
    try:
        try:
            os.link(stream.name, manifest)
        except FileExistsError:
            if json.loads(manifest.read_text()) != stamp:
                raise ValueError("Use a separate output directory for different code or settings")
    finally:
        Path(stream.name).unlink()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        studies[args.study](Run(path, args.quick, executor))
    print(f"{args.study} {'example' if args.quick else 'study'} finished in {path}")
