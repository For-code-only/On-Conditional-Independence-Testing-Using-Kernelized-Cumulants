"""Runner contracts and small checkpoint checks; no paper-grid run is started."""

from collections import Counter
from copy import deepcopy
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from csic import benchmark
from csic.simulation import paper_seed


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Two n=8 engineering fixtures exercise real result schemas without
        # generating any n=200/400/800 paper sample.
        cls.config = dict(seed_root=1234, B=19, alpha=0.05, trace_rtol=1e-8,
                          rho400=0.01, y_multipliers={"baseline": 1.0, "proposed": 2.0})
        cls.config_hash = hashlib.sha256(benchmark._bytes(cls.config)).hexdigest()
        cls.fixtures = []
        for study, d in (("main_unconditional", 0), ("main_conditional", 2)):
            job = benchmark.make_jobs([study], [8], [d], 1, 1)[0]
            record = benchmark.evaluate_job(job, cls.config, cls.config_hash)
            cls.fixtures.append((job, record))

    def test_main_budget_and_replication_identity(self):
        jobs = benchmark.make_jobs(benchmark.STUDIES, [200, 400, 800], [2, 5], 1000, 500)
        self.assertEqual(len(jobs), 24000)
        self.assertEqual(2 * len(jobs), 48000)
        self.assertEqual(len({job["id"] for job in jobs}), 24000)
        self.assertEqual(Counter(job["study"] for job in jobs),
                         {"main_unconditional": 6000, "main_conditional": 12000, "seoul": 6000})
        cells = Counter((job["study"], job["n"], job["d"], job["case"]) for job in jobs)
        self.assertEqual(len(cells), 36)
        for (study, n, d, case), count in cells.items():
            self.assertEqual(count, 1000 if case.endswith("0") else 500)
            self.assertIn(n, (200, 400, 800))
            self.assertEqual(d, 0 if study == "main_unconditional" else 5 if study == "seoul" else d)
            if study == "main_conditional":
                self.assertIn(d, (2, 5))
        for job in jobs:
            self.assertEqual(job["c0"], 8.0)
            self.assertEqual(job["cell"], job["case"])
            self.assertLess(job["rep"], 1000 if job["case"].endswith("0") else 500)
        self.assertNotIn("diagnostic", {job["study"] for job in jobs})

    def test_main_null_keeps_c0_in_seed_serialization(self):
        job = benchmark.make_jobs(["main_unconditional"], [200], [2, 5], 1, 1)[0]
        self.assertEqual((job["case"], job["c0"], job["rep"]), ("U0", 8.0, 0))
        self.assertEqual(paper_seed(job, "data"), 1523935372057159294)
        self.assertNotEqual(paper_seed(job, "data"), paper_seed(dict(job, c0=0.0), "data"))

    def test_seoul_dimension_does_not_follow_gaussian_option(self):
        jobs = benchmark.make_jobs(["seoul"], [12], [2], 2, 1)
        self.assertEqual(len(jobs), 4)
        self.assertTrue(all(job["d"] == 5 for job in jobs))
        self.assertEqual(Counter(job["case"] for job in jobs), {"C0": 2, "C1": 1, "C2": 1})

    def test_save_is_idempotent_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            benchmark._save(path, {"complete": True, "value": 1})
            original = path.read_bytes()
            benchmark._save(path, {"value": 1, "complete": True})
            self.assertEqual(path.read_bytes(), original)
            with self.assertRaisesRegex(RuntimeError, "overwrite"):
                benchmark._save(path, {"complete": True, "value": 2})
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_valid_small_checkpoints_are_accepted(self):
        for job, record in self.fixtures:
            with self.subTest(study=job["study"]):
                benchmark._check_record(record, job, self.config, self.config_hash)

    def test_checkpoint_rejects_wrong_job_or_configuration(self):
        job, original = self.fixtures[0]
        for problem in ("job", "configuration"):
            with self.subTest(problem=problem):
                record = deepcopy(original)
                if problem == "job":
                    record["job"]["rep"] += 1
                else:
                    record["configuration_sha256"] = "0" * 64
                with self.assertRaises(RuntimeError):
                    benchmark._check_record(record, job, self.config, self.config_hash)

    def test_checkpoint_rejects_forged_data_or_mixture_seed(self):
        for job, original in self.fixtures:
            for stream in ("data", "baseline", "proposed"):
                with self.subTest(study=job["study"], stream=stream):
                    record = deepcopy(original)
                    if stream == "data":
                        record["data_seed"] += 1
                    else:
                        record["results"][stream]["metadata"]["calibration_seed"] += 1
                    with self.assertRaises(RuntimeError):
                        benchmark._check_record(record, job, self.config, self.config_hash)

    def test_checkpoint_rejects_wrong_method_bandwidth_penalty_or_B(self):
        job, original = self.fixtures[1]
        for field, value in (("method", "HSIC"), ("sigma_y_multiplier", 1.0),
                             ("rho400", 0.0001), ("B_requested", 99)):
            with self.subTest(field=field):
                record = deepcopy(original)
                result = record["results"]["proposed"]
                if field == "method":
                    result[field] = value
                else:
                    result["metadata"][field] = value
                with self.assertRaises(RuntimeError):
                    benchmark._check_record(record, job, self.config, self.config_hash)

    def test_checkpoint_rejects_missing_half_of_pair(self):
        job, original = self.fixtures[0]
        record = deepcopy(original)
        del record["results"]["proposed"]
        with self.assertRaises(RuntimeError):
            benchmark._check_record(record, job, self.config, self.config_hash)

    def test_checkpoint_rejects_contradictory_calibration(self):
        job, original = self.fixtures[1]
        self.assertEqual(original["results"]["proposed"]["calibration"]["calibration"],
                         "spectral_mc_plus_one")
        for problem in ("alpha", "B_requested", "B", "tail", "tail_type", "formula", "exact", "unknown"):
            with self.subTest(problem=problem):
                record = deepcopy(original)
                cal = record["results"]["proposed"]["calibration"]
                if problem == "alpha":
                    cal["alpha"] = 0.1
                elif problem in ("B_requested", "B"):
                    cal[problem] = 99
                elif problem == "tail":
                    cal["tail_count"] = self.config["B"] + 1
                elif problem == "tail_type":
                    cal["tail_count"] = True
                elif problem == "formula":
                    cal["pvalue"] = 0.123
                    cal["reject"] = False
                elif problem == "exact":
                    # Exact branches must report zero actual MC draws and no tail.
                    cal["calibration"] = "exact_degenerate"
                else:
                    cal["calibration"] = "unrecognized"
                with self.assertRaises(RuntimeError):
                    benchmark._check_record(record, job, self.config, self.config_hash)

    def test_checkpoint_rejects_bad_statistics_or_input_metadata(self):
        job, original = self.fixtures[0]
        for problem in ("nan", "negative", "spectrum", "n", "source_seed", "fingerprint"):
            with self.subTest(problem=problem):
                record = deepcopy(original)
                result = record["results"]["baseline"]
                if problem == "nan":
                    result["statistic"] = float("nan")
                elif problem == "negative":
                    result["statistic"] = -1.0
                elif problem == "spectrum":
                    result["spectrum"]["statistic"] += 1
                elif problem == "n":
                    result["metadata"]["n"] += 1
                elif problem == "source_seed":
                    result["metadata"]["source_sample"]["seed"] += 1
                else:
                    record["input_sha256"] = "not-a-sha256"
                with self.assertRaises(RuntimeError):
                    benchmark._check_record(record, job, self.config, self.config_hash)

    def test_cli_rejects_unattainable_level_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with patch("sys.stderr", new_callable=io.StringIO):
                with self.assertRaises(SystemExit) as error:
                    benchmark.main(["--study", "unconditional", "--B", "18", "--output", str(output)])
            self.assertEqual(error.exception.code, 2)
            self.assertFalse(output.exists())

    def test_lock_is_exclusive_and_released(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            with benchmark._lock(directory):
                with self.assertRaisesRegex(RuntimeError, "already in use"):
                    with benchmark._lock(directory):
                        self.fail("A second lock unexpectedly succeeded")
            with benchmark._lock(directory):
                pass
            with self.assertRaisesRegex(ValueError, "interruption"):
                with benchmark._lock(directory):
                    raise ValueError("interruption")
            with benchmark._lock(directory):
                pass


if __name__ == "__main__":
    unittest.main()
