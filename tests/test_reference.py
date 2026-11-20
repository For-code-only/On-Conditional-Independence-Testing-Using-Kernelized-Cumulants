"""Replay six method results recorded by the original main experiment."""
import json
from pathlib import Path
import unittest

import numpy as np
from csic.inference import evaluate_sample
from csic.simulation import generate_job, load_seoul, paper_seed

ROOT = Path(__file__).resolve().parents[1]


class ReferenceTests(unittest.TestCase):
    def test_saved_results(self):
        fixtures = json.loads((ROOT / "tests/fixtures/reference_cases.json").read_text())
        pool = load_seoul(ROOT / "data/seoul_weather.npy")
        for fixture in fixtures:
            job = fixture["job"]
            with self.subTest(job=job["id"]):
                self.assertEqual(paper_seed(job, "data"), fixture["data_seed"])
                sample = generate_job(job, seoul_pool=pool)
                np.testing.assert_allclose(sample["x"][:4], fixture["x_head"], rtol=2e-14, atol=2e-14)
                np.testing.assert_allclose(sample["y"][:4], fixture["y_head"], rtol=2e-14, atol=2e-14)
                for expected in fixture["methods"]:
                    method = expected["method"]
                    self.assertEqual(paper_seed(job, "mixture", method), expected["seed"])
                    result = evaluate_sample(sample, method, seed=expected["seed"])
                    np.testing.assert_allclose(result["estimate"]["statistic"], expected["statistic"], rtol=1e-9, atol=1e-12)
                    np.testing.assert_allclose(result["estimate"]["weights"], expected["weights"], rtol=1e-7, atol=1e-12)
                    # Allow at most one Monte Carlo draw at a numerically tied threshold across BLAS builds.
                    self.assertLessEqual(abs(result["calibration"]["pvalue"]-expected["pvalue"]), 1/5000 + 1e-14)
                    self.assertEqual(result["calibration"]["reject"], expected["reject"])


if __name__ == "__main__":
    unittest.main()
