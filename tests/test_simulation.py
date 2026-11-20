"""Small engineering checks; no paper-grid datasets are generated."""

import tempfile
from pathlib import Path
import unittest

import numpy as np

from csic.simulation import generate, generate_job, load_seoul, paper_seed


class SimulationTests(unittest.TestCase):
    def test_archived_seed_vectors(self):
        # Constants retained from the paper's saved job manifest, not recomputed
        # by a second copy of the seed implementation. Only seeds are evaluated.
        job = dict(study="main_unconditional", case="U0", n=200, d=0, c0=8., rep=0)
        self.assertEqual(paper_seed(job, "data"), 1523935372057159294)
        self.assertEqual(paper_seed(job, "mixture", "HSIC"), 10184335772988207312)
        self.assertEqual(paper_seed(job, "mixture", "CSIC"), 3689991391600367394)
        job = dict(study="main_conditional", case="C0", n=400, d=5, c0=8., rep=0)
        self.assertEqual(paper_seed(job, "data"), 16514845375549730671)
        self.assertEqual(paper_seed(job, "mixture", "KCI"), 1101786146186624445)
        self.assertEqual(paper_seed(job, "mixture", "CSIC_CI"), 3133640349832951632)
        self.assertNotEqual(paper_seed(job, "data"), paper_seed(dict(job, c0=0.), "data"))
        job = dict(study="seoul", case="C0", n=200, d=5, c0=8., rep=0)
        self.assertEqual(paper_seed(job, "data"), 3298085029925429987)

    def test_roles_and_methods_have_separate_streams(self):
        job = dict(study="main_unconditional", case="U2", n=12, d=0, c0=8., rep=0)
        seeds = [paper_seed(job, "data")]
        seeds += [paper_seed(job, "mixture", method) for method in ("HSIC", "CSIC")]
        self.assertEqual(len(set(seeds)), 3)
        self.assertNotEqual(seeds[0], paper_seed(job, "data", root=1234))
        with self.assertRaises(ValueError):
            paper_seed(job, "data", "HSIC")
        with self.assertRaises(ValueError):
            paper_seed(job, "mixture")

    def test_small_gaussian_jobs_are_deterministic(self):
        for case in ("U0", "U1", "U2", "C0", "C1", "C2"):
            unconditional = case.startswith("U")
            job = dict(study="main_unconditional" if unconditional else "main_conditional",
                       case=case, n=12, d=0 if unconditional else 2, c0=8., rep=0)
            a, b = generate_job(job, seed_root=1234), generate_job(job, seed_root=1234)
            for key in ("x", "y"):
                np.testing.assert_array_equal(a[key], b[key])
                self.assertEqual(a[key].shape, (12,))
                self.assertTrue(np.all(np.isfinite(a[key])))
            if unconditional:
                self.assertIsNone(a["z"])
            else:
                np.testing.assert_array_equal(a["z"], b["z"])
                self.assertEqual(a["z"].shape, (12, 2))
            self.assertEqual(a["metadata"], b["metadata"])
            self.assertEqual(a["metadata"]["c0"], 0. if case.endswith("0") else 8.)

    def test_zero_effect_alternatives_match_the_null(self):
        for prefix in ("U", "C"):
            null = generate(prefix + "0", 12, 17, d=2, c0=8.)
            for suffix in ("1", "2"):
                alt = generate(prefix + suffix, 12, 17, d=2, c0=0.)
                np.testing.assert_array_equal(null["x"], alt["x"])
                np.testing.assert_array_equal(null["y"], alt["y"])

    def test_seoul_row_draws_precede_noise(self):
        # A synthetic pool tests indexing and draw order without a download.
        pool = np.arange(8760 * 5, dtype=float).reshape(8760, 5) / 8760
        job = dict(study="seoul", case="C0", n=12, d=5, c0=8., rep=0)
        sample = generate_job(job, seed_root=1234, seoul_pool=pool)
        rng = np.random.default_rng(paper_seed(job, "data", root=1234))
        indices = rng.integers(0, 8760, size=12, dtype=np.int64)
        r, epsilon = rng.normal(size=(2, 12))
        z = pool[indices]
        f = np.sin(z).sum(axis=1) / np.sqrt(5)
        g = (0.5 * z + 0.25 * (z * z - 1)).sum(axis=1) / np.sqrt(5)
        np.testing.assert_array_equal(sample["row_indices"], indices)
        np.testing.assert_array_equal(sample["z"], z)
        np.testing.assert_array_equal(sample["x"], f + r)
        np.testing.assert_array_equal(sample["y"], g + epsilon)
        self.assertEqual(sample["row_indices"].dtype, np.dtype("int16"))
        self.assertEqual(sample["metadata"]["c0"], 0.)
        with self.assertRaises(ValueError):
            generate_job(job, seoul_pool=np.zeros((8760, 4)))

    def test_seoul_loader_rejects_unverified_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pool.npy"
            np.save(path, np.zeros((8760, 5)))
            with self.assertRaisesRegex(ValueError, "SHA256"):
                load_seoul(path)


if __name__ == "__main__":
    unittest.main()
