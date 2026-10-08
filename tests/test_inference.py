"""Deterministic small-matrix checks for the public inference API."""
import inspect
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import chi2

from csic.inference import build_matrices, evaluate_sample, matrix_penalty, median_bandwidth
from csic.inference import test as csic_test
from csic.kernels import center, conditional, mixture_pvalue, _mc_summary, rbf, unconditional


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.x = np.array([-1.4, -0.2, 0.3, 0.7, 1.5, 2.1])
        self.y = np.array([0.4, 1.2, -0.8, 1.7, 0.1, -1.1])
        self.z = np.array([[0.1, 1.0], [0.4, -0.2], [-0.7, 0.3],
                           [1.5, 0.1], [0.6, 0.9], [-0.2, -1.0]])
        self.n = len(self.x)
        self.H = np.eye(self.n) - np.ones((self.n, self.n)) / self.n

    def test_center_matches_projection(self):
        K = rbf(self.x, sigma=1.3)
        centered = center(K)
        assert_allclose(centered, self.H @ K @ self.H, atol=5e-16)
        assert_allclose(centered.sum(axis=0), 0.0, atol=2e-15)
        assert_allclose(centered.sum(axis=1), 0.0, atol=2e-15)
        asymmetric = K.copy()
        asymmetric[0, 1] += 0.1
        assert_allclose(center(asymmetric),
                        self.H @ ((asymmetric + asymmetric.T) / 2) @ self.H,
                        atol=5e-16)

    def test_unconditional_statistics_and_quadratic_spectrum(self):
        KX, KY = rbf(self.x, 1.3), rbf(self.y, 0.9)
        KXc, KYc = self.H @ KX @ self.H, self.H @ KY @ self.H
        Q = KYc * KYc
        out = unconditional(KX, KY, trace_rtol=0)
        self.assertAlmostEqual(out['HSIC']['statistic'], np.trace(KXc @ KYc) / self.n, places=14)
        self.assertAlmostEqual(out['CSIC']['statistic'], np.trace(KXc @ Q) / self.n, places=14)
        # Quadratic-feature centering is essential for the null spectrum.
        Cx, Cq = KXc / self.n, self.H @ Q @ self.H / self.n
        weights = out['CSIC']['weights']
        self.assertAlmostEqual(weights.sum(), np.trace(Cx) * np.trace(Cq), places=14)
        self.assertAlmostEqual(weights @ weights,
                               np.trace(Cx @ Cx) * np.trace(Cq @ Cq), places=14)
        self.assertGreater(abs(np.trace(Cq) - np.trace(Q / self.n)), 1e-3)

    def test_conditional_square_precedes_residualization(self):
        matrices = build_matrices(self.x, self.y, self.z, method='CSIC_CI')
        KX, KY, KZ = (matrices[k] for k in ('KX', 'KY', 'KZ'))
        ridge = matrix_penalty(0.1, self.n)
        out = conditional(KX, KY, KZ, ridge, trace_rtol=0, return_matrices=True)
        # Dense inverse/projection formula independent of implementation's
        # eigendecomposition and implicit centering.
        A = ridge * np.linalg.solve(KZ + ridge * np.eye(self.n), np.eye(self.n))
        KAc, KYc = self.H @ (KX * KZ) @ self.H, self.H @ KY @ self.H
        LA, LB, LQ = A @ KAc @ A, A @ KYc @ A, A @ (KYc * KYc) @ A
        for name, expected in (('A', A), ('LA', LA), ('LB', LB), ('LQ', LQ)):
            assert_allclose(out['matrices'][name], expected, rtol=5e-12, atol=2e-14)
        self.assertGreater(np.linalg.norm(LQ - LB * LB), 1e-4)
        for method, response in (('KCI', LB), ('CSIC_CI', LQ)):
            self.assertAlmostEqual(out[method]['statistic'],
                                   np.trace(LA @ response) / self.n, places=13)
            score_covariance = self.H @ (LA * response) @ self.H / self.n
            weights = out[method]['weights']
            self.assertAlmostEqual(weights.sum(), np.trace(score_covariance), places=13)
            self.assertAlmostEqual(weights @ weights,
                                   np.trace(score_covariance @ score_covariance), places=13)

    def test_paper_defaults_and_result_metadata(self):
        self.assertEqual(inspect.signature(csic_test).parameters['B'].default, 4999)
        self.assertEqual(inspect.signature(mixture_pvalue).parameters['B'].default, 4999)
        for method, multiplier in (('HSIC', 1), ('CSIC', 2), ('KCI', 1), ('CSIC_CI', 2)):
            with self.subTest(method=method):
                z = self.z if method in ('KCI', 'CSIC_CI') else None
                out = csic_test(self.x, self.y, z, method=method, seed=72, B=39)
                meta = out['metadata']
                self.assertEqual(meta['sigma_y_multiplier'], multiplier)
                self.assertEqual(meta['sigma_y'], multiplier * median_bandwidth(self.y)[0])
                self.assertEqual(meta['sigma_x'], median_bandwidth(self.x)[0])
                self.assertFalse(meta['sample_standardization'])
                self.assertEqual(meta['calibration_seed'], 72)
                self.assertEqual(meta['B_requested'], 39)
                self.assertIsInstance(out['estimate']['weights'], np.ndarray)
                self.assertEqual(out['calibration']['alpha'], 0.05)
                if z is None:
                    self.assertIsNone(meta['rho400'])
                    self.assertIsNone(meta['ridge_matrix'])
                else:
                    self.assertEqual(meta['rho400'], 0.01)
                    self.assertEqual(meta['sigma_z'], np.sqrt(2))
                    self.assertEqual(meta['z_kernel'], '(1+RBF)/2')
                    self.assertEqual(meta['ridge_matrix'], 0.01 * np.log(self.n) / np.log(400))
                    self.assertEqual(meta['lambda_population'], meta['ridge_matrix'] / self.n)

    def test_explicit_parameters_and_runner_entry_point(self):
        sample = {'x': self.x, 'y': self.y, 'z': self.z, 'metadata': {'fixture': True}}
        direct = evaluate_sample(sample, 'CSIC_CI', seed=np.int64(98), rho400=0.02,
                                 sigma_y_multiplier=1, B=79, alpha=0.1, trace_rtol=1e-7)
        public = csic_test(self.x, self.y, self.z, method='CSIC_CI', seed=98,
                           rho400=0.02, sigma_y_multiplier=1, B=79,
                           alpha=0.1, trace_rtol=1e-7)
        self.assertEqual(direct['estimate']['statistic'], public['estimate']['statistic'])
        assert_array_equal(direct['estimate']['weights'], public['estimate']['weights'])
        self.assertEqual(direct['calibration']['pvalue'], public['calibration']['pvalue'])
        self.assertEqual(direct['metadata']['source_sample'], {'fixture': True})
        self.assertEqual(direct['metadata']['sigma_y_multiplier'], 1)
        self.assertEqual(direct['metadata']['rho400'], 0.02)
        self.assertEqual(matrix_penalty(0.02, 400), 0.02)
        assert_allclose(build_matrices(self.x, self.y, self.z, method='KCI')['KZ'],
                        (1 + rbf(self.z, np.sqrt(2))) / 2, rtol=0, atol=0)

    def test_bandwidth_ties_and_multivariate_observations(self):
        self.assertEqual(median_bandwidth([0, 0, 2]), (2.0, False))
        self.assertEqual(median_bandwidth([3, 3, 3]), (1.0, True))
        self.assertEqual(median_bandwidth([[0, 0], [3, 4], [0, 4]]), (4.0, False))
        out = csic_test(np.ones(4), np.ones(4), seed=1, B=39)
        self.assertEqual(out['calibration']['calibration'], 'exact_degenerate')
        self.assertEqual(out['calibration']['pvalue'], 1.0)
        self.assertFalse(out['calibration']['reject'])

    def test_required_seed_and_parameter_validation(self):
        with self.assertRaises(TypeError):
            csic_test(self.x, self.y)
        with self.assertRaises(TypeError):
            csic_test(self.x, self.y, None, 'CSIC', 2)
        for bad_seed in (None, True, -1, 1.5, '2'):
            with self.subTest(seed=bad_seed), self.assertRaises(ValueError):
                csic_test(self.x, self.y, seed=bad_seed, B=3)
        for args in ({'method': 'other'}, {'method': 'KCI'}, {'rho400': 0.01},
                     {'sigma_y_multiplier': 0}, {'sigma_y_multiplier': np.nan},
                     {'B': 0}, {'B': 1.5}, {'alpha': 0}, {'alpha': 1},
                     {'trace_rtol': -1}, {'trace_rtol': 1}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                csic_test(self.x, self.y, seed=1, **args)
        with self.assertRaises(ValueError):
            csic_test(self.x, self.y, self.z, seed=1)
        with self.assertRaises(ValueError):
            csic_test(self.x, self.y, self.z, method='KCI', seed=1, rho400=0)
        for x, y in (([1], [2]), ([1, 2], [2]), ([1, np.nan], [2, 3]),
                     (np.empty((2, 0)), [2, 3])):
            with self.subTest(x=x), self.assertRaises(ValueError):
                csic_test(x, y, seed=1)
        for n in (1, True, 3.5):
            with self.subTest(n=n), self.assertRaises(ValueError):
                matrix_penalty(0.01, n)

    def test_exact_spectral_branches(self):
        for statistic, expected_p in ((0, 1), (1, 0)):
            out = mixture_pvalue(statistic, [], np.random.default_rng(1))
            self.assertEqual(out['calibration'], 'exact_degenerate')
            self.assertEqual(out['pvalue'], expected_p)
            self.assertEqual(out['B'], 0)
            self.assertEqual(out['B_requested'], 4999)
        out = mixture_pvalue(4.0, [0, 2.0, 2.0, 2.0], np.random.default_rng(1))
        self.assertEqual(out['calibration'], 'exact_equal_weights_chi2')
        self.assertEqual(out['pvalue'], chi2.sf(2.0, 3))
        self.assertEqual(out['critical'], 2 * chi2.isf(0.05, 3))
        self.assertEqual(out['B'], 0)
        self.assertEqual(out['chi2_df'], 3)
        self.assertEqual(out['chi2_scale'], 2.0)

    def test_spectral_mc_plus_one_against_direct_draws(self):
        seed, B, alpha, statistic = 19, 39, 0.05, 4.5
        weights = np.array([1.5, 0.6, 0.1])
        draws = (np.random.default_rng(seed).standard_normal((B, 3)) ** 2 * weights).sum(axis=1)
        out = mixture_pvalue(statistic, weights, np.random.default_rng(seed), B=B, alpha=alpha)
        tail = int((draws >= statistic).sum())
        self.assertEqual(out['calibration'], 'spectral_mc_plus_one')
        self.assertEqual(out['tail_count'], tail)
        self.assertEqual(out['pvalue'], (tail + 1) / (B + 1))
        self.assertEqual(out['critical_index_one_based'], 38)
        self.assertEqual(out['critical'], np.sort(draws)[37])
        self.assertEqual(out['reject'], statistic > out['critical'])

    def test_rank_rule_ties_and_unattainable_alpha(self):
        # Ties at the critical value contribute to the right tail.
        draws = np.array([0, 1, 2, 3, 4, 5, 6, 8, 8], dtype=float)
        at_tie = _mc_summary(8.0, draws, alpha=0.2)
        self.assertEqual(at_tie['tail_count'], 2)
        self.assertEqual(at_tie['ties_at_observed'], 2)
        self.assertEqual(at_tie['pvalue'], 0.3)
        self.assertEqual(at_tie['critical'], 8.0)
        self.assertFalse(at_tie['reject'])
        self.assertTrue(_mc_summary(8.1, draws, alpha=0.2)['reject'])
        unattainable = _mc_summary(100, draws, alpha=0.01)
        self.assertEqual(unattainable['pvalue'], 0.1)
        self.assertEqual(unattainable['critical'], float('inf'))
        self.assertFalse(unattainable['reject'])


if __name__ == '__main__':
    unittest.main()
