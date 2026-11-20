"""Small regression cases from the supplementary simulations."""
import unittest
import numpy as np

from csic.studies.balanced import run_one
from csic.studies.factorial import evaluate
from csic.studies.mixed import operating_point, rejection_rates


class SupplementaryStudyTests(unittest.TestCase):
    def test_paired_variance_stream_and_configuration_order(self):
        result = evaluate(0, setting=("main_conditional", 200, 5), role="power",
                          case="C2", c0=8., tau=1., variants={"original": {"rho400": .01}})
        np.testing.assert_array_equal(result, [[.0054, .0966, .0460, .0086]])

    def test_equal_class_counts_retain_ridge_intercept(self):
        row = run_one("alternative", 1)
        self.assertEqual(tuple(row[2:6]), (200, .0094, .7972, .0094))

    def test_fractional_roc_ties(self):
        null = np.array([[.1], [.1], [.2], [.3]])
        cut, fraction, _, _ = operating_point(null, alpha=.25)
        np.testing.assert_array_equal(cut, [.1])
        np.testing.assert_array_equal(fraction, [.5])
        np.testing.assert_array_equal(rejection_rates(null, cut, fraction), [.25])
