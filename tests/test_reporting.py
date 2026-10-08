"""Small deterministic reporting checks; no scientific simulations are run."""
import csv
import itertools
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from csic import reporting


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        (self.directory / "records").mkdir()

    def record(self, rep, baseline, proposed, *, study="main_unconditional", n=17, d=0, case="U2", **overrides):
        methods = ("HSIC", "CSIC") if d == 0 else ("KCI", "CSIC_CI")
        job = dict(id=f"{study}_{case}_n{n}_d{d}_r{rep}", study=study, case=case,
                   cell=case, n=n, d=d, c0=8.0, rep=rep)
        job.update(overrides)
        results = {}
        for config, method, decision, multiplier in zip(reporting.CONFIGS, methods, (baseline, proposed), (1.0, 2.0)):
            results[config] = dict(method=method, statistic=0.125,
                                   calibration=dict(reject=decision, pvalue=0.01 if decision else 0.5),
                                   y_multiplier=multiplier,
                                   metadata=dict(sigma_y_multiplier=multiplier, rho400=None if d == 0 else 0.01))
        return dict(job=job, results=results, configuration_sha256="test-configuration")

    def save(self, record, name=None):
        path = self.directory / "records" / (name or f"{record['job']['id']}.json")
        path.write_text(json.dumps(record), encoding="utf-8")
        return path

    @staticmethod
    def read_csv(path):
        with Path(path).open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    def test_wilson_known_value_and_boundaries(self):
        low, high = reporting.wilson(3, 10)
        self.assertAlmostEqual(low, 0.10779126740630099)
        self.assertAlmostEqual(high, 0.6032218525388546)
        self.assertEqual(reporting.wilson(0, 10)[0], 0.0)
        self.assertEqual(reporting.wilson(10, 10)[1], 1.0)
        for k, n in [(-1, 10), (11, 10), (0, 0), (True, 3)]:
            with self.assertRaises(ValueError):
                reporting.wilson(k, n)

    def test_paired_matches_direct_sample_variance(self):
        for n in range(2, 5):
            for first in itertools.product((False, True), repeat=n):
                for reference in itertools.product((False, True), repeat=n):
                    result = reporting.paired(first, reference)
                    differences = [int(a) - int(b) for a, b in zip(first, reference)]
                    mean = sum(differences) / n
                    expected = math.sqrt(sum((d - mean) ** 2 for d in differences) / (n * (n - 1)))
                    self.assertAlmostEqual(result["paired_difference"], mean)
                    self.assertAlmostEqual(result["mcse"], expected)
                    self.assertGreaterEqual(result["ci_low"], -1)
                    self.assertLessEqual(result["ci_high"], 1)
        self.assertEqual(reporting.paired([True, True], [False, False])["ci_high"], 1)
        self.assertEqual(reporting.paired([False, False], [True, True])["ci_low"], -1)

    def test_paired_one_rep_and_invalid_lengths(self):
        result = reporting.paired([True], [False])
        self.assertEqual(result["paired_difference"], 1)
        self.assertIsNone(result["mcse"])
        self.assertIsNone(result["ci_low"])
        for first, second in [([], []), ([True], []), ([1], [False])]:
            with self.assertRaises(ValueError):
                reporting.paired(first, second)

    def test_summary_is_paired_and_counts_actual_saved_records(self):
        # Deliberately nonconsecutive, lexically unordered replication IDs.
        for rep, first, second in [(9, True, True), (1, True, False), (5, False, True), (3, False, True)]:
            self.save(self.record(rep, first, second))
        with patch.object(reporting, "plot", return_value=[]):
            result = reporting.summarize(self.directory)
        self.assertEqual(result["datasets"], 4)
        rates = self.read_csv(result["rates"])
        self.assertEqual(tuple(rates[0]), reporting.RATE_FIELDS)
        self.assertEqual([float(row["rate"]) for row in rates], [0.5, 0.75])
        self.assertEqual([row["rho400"] for row in rates], ["", ""])
        self.assertEqual([int(row["replications"]) for row in rates], [4, 4])
        pair = self.read_csv(result["paired_differences"])[0]
        self.assertEqual(tuple(pair), reporting.PAIRED_FIELDS)
        self.assertEqual(int(pair["first_only"]), 2)
        self.assertEqual(int(pair["reference_only"]), 1)
        self.assertAlmostEqual(float(pair["paired_difference"]), 0.25)
        self.assertAlmostEqual(float(pair["mcse"]), math.sqrt(2.75 / 12))

    def test_one_rep_summary_leaves_unestimable_uncertainty_blank(self):
        self.save(self.record(0, False, True))
        with patch.object(reporting, "plot", return_value=[]):
            result = reporting.summarize(self.directory)
        pair = self.read_csv(result["paired_differences"])[0]
        self.assertEqual([pair[key] for key in ("mcse", "ci_low", "ci_high")], ["", "", ""])
        self.assertEqual(pair["paired_difference"], "1.0")

    def test_duplicate_replication_and_id_rejected_before_output(self):
        for change_id in (False, True):
            with self.subTest(change_id=change_id):
                for path in (self.directory / "records").glob("*.json"):
                    path.unlink()
                original = self.record(0, False, True)
                self.save(original)
                if change_id:
                    original["job"]["id"] = "different-id-same-rep"
                self.save(original, "duplicate.json")
                with self.assertRaisesRegex(ValueError, "Duplicate"):
                    reporting.summarize(self.directory)
                self.assertFalse((self.directory / "summary").exists())

    def test_missing_result_or_invalid_calibration_is_not_silently_skipped(self):
        for problem in ("missing", "not-boolean", "nonfinite", "wrong-method", "multiplier"):
            with self.subTest(problem=problem):
                rec = self.record(0, False, True)
                if problem == "missing":
                    del rec["results"]["proposed"]
                elif problem == "not-boolean":
                    rec["results"]["baseline"]["calibration"]["reject"] = "false"
                elif problem == "nonfinite":
                    rec["results"]["baseline"]["calibration"]["pvalue"] = float("nan")
                elif problem == "wrong-method":
                    rec["results"]["baseline"]["method"] = "KCI"
                else:
                    rec["results"]["baseline"]["metadata"]["sigma_y_multiplier"] = 3
                self.save(rec)
                with self.assertRaises(ValueError):
                    reporting.summarize(self.directory)
                self.assertFalse((self.directory / "summary").exists())

    def test_mixed_amplitude_or_configuration_rejected(self):
        self.save(self.record(0, False, True))
        self.save(self.record(1, True, True, c0=4.0))
        with self.assertRaisesRegex(ValueError, "c0"):
            reporting.summarize(self.directory)
        second = self.record(1, True, True)
        second["configuration_sha256"] = "different"
        self.save(second)
        with self.assertRaisesRegex(ValueError, "configuration identities"):
            reporting.summarize(self.directory)

    def test_metadata_multiplier_without_convenience_field(self):
        record = self.record(0, False, True, study="main_conditional", n=23, d=2, case="C0")
        for result in record["results"].values():
            del result["y_multiplier"]
        self.save(record)
        with patch.object(reporting, "plot", return_value=[]):
            result = reporting.summarize(self.directory)
        self.assertEqual([row["rho400"] for row in self.read_csv(result["rates"])], ["0.01", "0.01"])

    def test_empty_records_fail(self):
        with self.assertRaisesRegex(ValueError, "No main-study records"):
            reporting.summarize(self.directory)

    def test_unresolved_failure_prevents_summary_write(self):
        self.save(self.record(0, False, True))
        failed_job = self.record(1, False, True)["job"]
        failures = self.directory / "failures"
        failures.mkdir()
        (failures / "attempt.json").write_text(json.dumps({"job": failed_job}))
        summary = self.directory / "summary"
        summary.mkdir()
        previous_rates = summary / "rates.csv"
        previous_rates.write_text("previous summary remains unchanged\n")
        with patch.object(reporting, "plot") as plot:
            with self.assertRaisesRegex(ValueError, "unresolved failed jobs") as caught:
                reporting.summarize(self.directory)
            plot.assert_not_called()
        self.assertIn(failed_job["id"], str(caught.exception))
        self.assertEqual(previous_rates.read_text(), "previous summary remains unchanged\n")
        self.assertFalse((summary / "paired_differences.csv").exists())

    def test_successful_retry_allows_retained_failure_logs(self):
        record = self.record(0, False, True)
        self.save(record)
        failures = self.directory / "failures"
        failures.mkdir()
        logs = []
        for attempt in range(2):
            path = failures / f"attempt-{attempt}.json"
            path.write_text(json.dumps({"job": record["job"], "traceback": "historical failure"}))
            logs.append((path, path.read_bytes()))
        with patch.object(reporting, "plot", return_value=[]):
            result = reporting.summarize(self.directory)
        self.assertEqual(result["datasets"], 1)
        self.assertTrue(all(row["failures"] == "0" for row in self.read_csv(result["rates"])))
        for path, original in logs:
            self.assertEqual(path.read_bytes(), original)

    def test_plot_nonstandard_n_all_studies_and_percentages(self):
        for study, d, prefix in [("main_unconditional", 0, "U"), ("main_conditional", 2, "C"),
                                 ("main_conditional", 5, "C"), ("seoul", 5, "C")]:
            for n in (17, 23):
                for case in (prefix + "0", prefix + "1", prefix + "2"):
                    for rep in range(2):
                        self.save(self.record(rep, rep == 0, True, study=study, n=n, d=d, case=case))
        from matplotlib.axes import Axes
        original = Axes.errorbar
        seen = []

        def capture(axis, x, y, *args, **kwargs):
            seen.append((list(x), list(y), kwargs["color"]))
            return original(axis, x, y, *args, **kwargs)

        with patch.object(Axes, "errorbar", capture):
            result = reporting.summarize(self.directory)
        self.assertEqual(len(result["figures"]), 6)
        self.assertEqual({p.name for p in result["figures"]},
                         {f"{stem}.{ext}" for stem in ("unconditional", "conditional", "seoul") for ext in ("png", "pdf")})
        for path in result["figures"]:
            self.assertGreater(path.stat().st_size, 1000)
        self.assertTrue(seen)
        self.assertTrue(all(x == [17, 23] for x, _, _ in seen))
        self.assertTrue(all(y in ([50.0, 50.0], [100.0, 100.0]) for _, y, _ in seen))
        self.assertEqual({color for _, _, color in seen}, {"#1f77b4", "#ff7f0e"})

    def test_standalone_plot_skips_absent_studies(self):
        self.save(self.record(0, False, True))
        with patch.object(reporting, "plot", return_value=[]):
            result = reporting.summarize(self.directory)
        paths = reporting.plot(result["rates"], self.directory / "plots")
        self.assertEqual([p.name for p in paths], ["unconditional.png", "unconditional.pdf"])

    def test_plot_rejects_invalid_rate_and_duplicate_coordinate(self):
        self.save(self.record(0, False, True))
        with patch.object(reporting, "plot", return_value=[]):
            result = reporting.summarize(self.directory)
        rates = self.read_csv(result["rates"])
        for invalid in (rates + [rates[0]], [dict(rates[0], rate="50")]):
            reporting._write_csv(result["rates"], reporting.RATE_FIELDS, invalid)
            with self.assertRaises(ValueError):
                reporting.plot(result["rates"], self.directory / "plots")

    def test_entrypoints(self):
        with patch.object(reporting, "summarize", return_value={"datasets": 2, "rates": self.directory / "summary/rates.csv"}) as summarize:
            with patch("builtins.print"):
                reporting.main([str(self.directory)])
            summarize.assert_called_once_with(self.directory)
        rates = self.directory / "rates.csv"
        with patch.object(reporting, "plot", return_value=[]) as plot:
            reporting.plot_main(["--rates", str(rates), "--output", str(self.directory)])
            plot.assert_called_once_with(rates, self.directory)


if __name__ == "__main__":
    unittest.main()
