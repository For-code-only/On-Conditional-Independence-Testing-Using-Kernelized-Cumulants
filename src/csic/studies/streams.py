"""Paired simulation streams. The namespace is part of the seed definition."""
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..simulation import generate, nuisance, h, q, load_seoul

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_VERSION = "csic_revision_v1"
PAPER_PLAN = {"seed_root": 20261031}


@lru_cache(maxsize=1)
def seoul_pool():
    return load_seoul(ROOT / "data/seoul_weather.npy")


def canonical(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def stable_seed(root, *parts):
    return int.from_bytes(hashlib.sha256(canonical(
        [int(root), PROTOCOL_VERSION, *parts])).digest()[:8], "little")


def data_seed(plan, job):
    """Each role has its own stream, paired across method configurations."""
    return stable_seed(plan["seed_root"], "data", job["role"], job["study"],
                       job["n"], job["d"], job["case"], job["c0"],
                       job.get("tau",1.), job["rep"])


def generate_sample(plan, job, pool=None):
    seed = data_seed(plan, job)
    if job["study"] != "seoul":
        sample = generate(job["case"],job["n"],seed,d=max(1,job["d"]),c0=job["c0"],return_latent=True)
    else:
        if pool is None:
            pool = seoul_pool()
        rng = np.random.default_rng(seed)
        ix = rng.integers(0,len(pool),size=job["n"],dtype=np.int64)
        z = pool[ix]
        r,epsilon = rng.normal(size=(2,job["n"]))
        f,g = nuisance(z)
        a = 0. if job["case"] == "C0" else job["c0"]/math.sqrt(job["n"])
        mean = g+a*h(r) if job["case"] == "C1" else g
        sd = np.exp(a*q(r)/2) if job["case"] == "C2" else np.ones(job["n"])
        sample = dict(x=f+r,y=mean+sd*epsilon,z=z,row_indices=ix.astype(np.int16),
                      latent=dict(r=r,epsilon_y=epsilon,f=f,g=g),
                      metadata=dict(case=job["case"],n=job["n"],d=5,a=a,c0=job["c0"]))
    if job.get("tau",1.) != 1.:
        if job["case"] != "C0":
            raise ValueError("Nuisance-strength stress is null-only")
        f,g = nuisance(sample["z"])
        sample["x"] = sample["x"]+(job["tau"]-1)*f
        sample["y"] = sample["y"]+(job["tau"]-1)*g
    if not all(np.isfinite(sample[key]).all() for key in ("x","y")):
        raise ArithmeticError("Nonfinite generated observations")
    sample["metadata"].update(data_seed=seed, namespace=PROTOCOL_VERSION,
                              role=job["role"],study=job["study"],tau=job.get("tau",1.))
    return sample


def pilot_sample(plan, study, d, role, size, reference_seed=0, pool=None):
    job = dict(study=study,n=size,d=d,role=role,case="C0",c0=0.,rep=reference_seed)
    return generate_sample(plan,job,pool)
