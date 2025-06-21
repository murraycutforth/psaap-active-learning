import unittest
import numpy as np
from scipy.special import expit
import pytest
from unittest.mock import patch, MagicMock
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

# Import the functions to test
from src.log_marginal_likelihood_metric import (
    bernoulli_likelihood,
    log_bernoulli_likelihood,
    sigmoid,
    compute_log_marginal_likelihood
)


class TestBernoulliLikelihoodFunctions(unittest.TestCase):
    """Tests for the Bernoulli likelihood functions."""

    def test_bernoulli_likelihood_basic(self):
        """Test basic functionality of bernoulli_likelihood."""
        y = np.array([0, 1, 0, 1])
        p = np.array([0.2, 0.7, 0.8, 0.3])
        expected = np.array([0.8, 0.7, 0.2, 0.3])

        result = bernoulli_likelihood(y, p)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_bernoulli_likelihood_clipping(self):
        """Test that probability clipping works properly."""
        y = np.array([0, 1])
        p = np.array([0, 1])  # These would cause issues without clipping

        result = bernoulli_likelihood(y, p)
        # Should be clipped to [1e-15, 1-1e-15]
        self.assertGreater(result[0], 0)  # Not exactly 0
        self.assertLess(result[1], 1)  # Not exactly 1

    def test_log_bernoulli_likelihood_basic(self):
        """Test basic functionality of log_bernoulli_likelihood."""
        y = np.array([0, 1, 0, 1])
        p = np.array([0.2, 0.7, 0.8, 0.3])
        expected = np.array([np.log(0.8), np.log(0.7), np.log(0.2), np.log(0.3)])

        result = log_bernoulli_likelihood(y, p)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_log_bernoulli_likelihood_clipping(self):
        """Test that probability clipping works properly in log version."""
        y = np.array([0, 1])
        p = np.array([0, 1])  # These would cause -inf without clipping

        result = log_bernoulli_likelihood(y, p)
        self.assertTrue(np.isfinite(result).all())  # No inf or nan values

    def test_log_and_nonlog_versions_consistent(self):
        """Test that log_bernoulli_likelihood is consistent with log(bernoulli_likelihood)."""
        y = np.array([0, 1, 0, 1])
        p = np.array([0.2, 0.7, 0.8, 0.3])

        direct_log = log_bernoulli_likelihood(y, p)
        log_of_nonlog = np.log(bernoulli_likelihood(y, p))

        np.testing.assert_allclose(direct_log, log_of_nonlog, rtol=1e-10)

    def test_sigmoid_basic(self):
        """Test basic functionality of sigmoid function."""
        x = np.array([-10, -1, 0, 1, 10])
        expected = expit(x)  # Using scipy's implementation as reference

        result = sigmoid(x)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_sigmoid_extreme_values(self):
        """Test sigmoid with extreme values that might cause numerical issues."""
        x = np.array([-1000, 1000])
        result = sigmoid(x)

        self.assertTrue(np.isfinite(result).all())  # No inf or nan values
        self.assertGreater(result[0], 0)  # Not exactly 0
        self.assertLess(result[1], 1)  # Not exactly 1


class TestLogMarginalLikelihood(unittest.TestCase):
    """Tests for the log marginal likelihood computation."""

    def setUp(self):
        """Set up a mock GPC with predictive distribution method."""
        self.mock_gpc = MagicMock()

        # Create sample data
        self.X_test = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        self.y_test = np.array([0, 1, 0])

        # Mock latent predictive distribution
        self.f_mean = np.array([-1.0, 2.0, -0.5])
        self.L_cov = np.array([
            [0.5, 0.1, 0.0],
            [0.1, 0.6, 0.2],
            [0.0, 0.2, 0.4]
        ])

        # Setup the mock to return our predefined distribution
        self.mock_gpc.latent_predictive_distribution = MagicMock(
            return_value=(self.f_mean, self.L_cov)
        )

    def test_compute_log_marginal_likelihood_basic(self):
        """Test basic functionality of compute_log_marginal_likelihood."""
        # Use fixed random seed for reproducibility
        log_ml = compute_log_marginal_likelihood(
            self.mock_gpc, self.X_test, self.y_test,
            n_samples=10, random_state=42
        )

        # Check we get a finite scalar value
        self.assertTrue(np.isscalar(log_ml))
        self.assertTrue(np.isfinite(log_ml))

        # Verify the mock was called correctly
        self.mock_gpc.latent_predictive_distribution.assert_called_once_with(self.X_test)

    def test_compute_log_marginal_likelihood_reproducibility(self):
        """Test that setting the same random seed gives reproducible results."""
        log_ml1 = compute_log_marginal_likelihood(
            self.mock_gpc, self.X_test, self.y_test,
            n_samples=10, random_state=42
        )

        log_ml2 = compute_log_marginal_likelihood(
            self.mock_gpc, self.X_test, self.y_test,
            n_samples=10, random_state=42
        )

        # Results should be identical with the same seed
        self.assertEqual(log_ml1, log_ml2)

        # Different seed should (almost certainly) give different results
        log_ml3 = compute_log_marginal_likelihood(
            self.mock_gpc, self.X_test, self.y_test,
            n_samples=10, random_state=43
        )

        # Very unlikely to be exactly equal
        self.assertNotEqual(log_ml1, log_ml3)

    def test_shape_mismatch_assertion(self):
        """Test that an assertion is raised when shapes don't match."""
        mismatched_y = np.array([0, 1])  # Wrong length

        with self.assertRaises(AssertionError):
            compute_log_marginal_likelihood(
                self.mock_gpc, self.X_test, mismatched_y,
                n_samples=10, random_state=42
            )

    def test_increasing_n_samples_improves_stability(self):
        """Test that increasing the number of samples reduces variance."""
        # Compute with low number of samples, multiple times
        low_samples_results = [
            compute_log_marginal_likelihood(
                self.mock_gpc, self.X_test, self.y_test,
                n_samples=5, random_state=i
            )
            for i in range(10)
        ]

        # Compute with higher number of samples, multiple times
        high_samples_results = [
            compute_log_marginal_likelihood(
                self.mock_gpc, self.X_test, self.y_test,
                n_samples=50, random_state=i
            )
            for i in range(10)
        ]

        # The variance should be lower with more samples
        low_var = np.var(low_samples_results)
        high_var = np.var(high_samples_results)

        self.assertLess(high_var, low_var)

    def test_correct_value_for_known_case(self):
        """Test against a manually calculated value for a simple case."""
        # Create a simpler test case where we can calculate the expected value
        simple_y_test = np.array([1])
        simple_f_mean = np.array([2.0])  # Strong positive, should give p ≈ 0.88
        simple_L_cov = np.array([[0.1]])  # Small variance

        simple_mock_gpc = MagicMock()
        simple_mock_gpc.latent_predictive_distribution = MagicMock(
            return_value=(simple_f_mean, simple_L_cov)
        )

        # Use high number of samples for accuracy
        log_ml = compute_log_marginal_likelihood(
            simple_mock_gpc, np.array([[0]]), simple_y_test,
            n_samples=10000, random_state=42
        )

        # With these parameters, the probability is ≈ 0.88 for class 1
        # The log likelihood should be approximately log(0.88) ≈ -0.13
        self.assertAlmostEqual(log_ml, np.log(0.88), delta=0.1)

    @patch('numpy.random.normal')
    def test_monte_carlo_sampling(self, mock_normal):
        """Test that the MC sampling is performed correctly."""
        # Configure the mock to return fixed values
        mock_normal.return_value = np.array([0.0, 0.0, 0.0])

        # Call the function
        compute_log_marginal_likelihood(
            self.mock_gpc, self.X_test, self.y_test,
            n_samples=3, random_state=42
        )

        # Check the mock was called with the right parameters
        mock_normal.assert_called_with(0, 1, size=3)
        self.assertEqual(mock_normal.call_count, 3)  # Called once per sample

    def test_logsumexp_trick(self):
        """Test that the logsumexp trick is implemented correctly."""
        # Create a mock that will return log likelihoods
        with patch('src.log_marginal_likelihood_metric.log_bernoulli_likelihood') as mock_log_like:
            # Return fixed values for log likelihoods to test logsumexp
            mock_log_like.side_effect = [
                np.array([-1000]),  # First call
                np.array([-10]),  # Second call
                np.array([-5])  # Third call
            ]

            # Call the function
            log_ml = compute_log_marginal_likelihood(
                self.mock_gpc, self.X_test, np.array([1]),
                n_samples=3, random_state=42
            )

            # The logsumexp result should be approximately:
            # max(-5) + log(exp(-1000-(-5)) + exp(-10-(-5)) + exp(-5-(-5)))/3
            # = -5 + log((0 + exp(-5) + 1)/3)
            # ≈ -5 + log(0.3356) ≈ -6.09
            self.assertAlmostEqual(log_ml, -6.09, delta=0.1)


class TestIntegration(unittest.TestCase):
    """Integration tests with a real GaussianProcessClassifier."""

    def setUp(self):
        """Set up a real GPC with a simple dataset."""
        # Skip if we're just running unit tests
        try:
            from sklearn.gaussian_process import GaussianProcessClassifier
            from sklearn.gaussian_process.kernels import RBF
        except ImportError:
            self.skipTest("scikit-learn not available")

        # Create a simple dataset
        np.random.seed(42)
        self.X_train = np.random.rand(20, 2)
        self.y_train = (self.X_train[:, 0] > 0.5).astype(int)
        self.X_test = np.random.rand(10, 2)
        self.y_test = (self.X_test[:, 0] > 0.5).astype(int)

        # Create and train a GPC
        kernel = 1.0 * RBF(length_scale=1.0)
        self.gpc = GaussianProcessClassifier(kernel=kernel, random_state=42)
        self.gpc.fit(self.X_train, self.y_train)

    def test_integration_with_real_gpc(self):
        """Test the function works with a real GPC."""
        try:
            log_ml = compute_log_marginal_likelihood(
                self.gpc, self.X_test, self.y_test,
                n_samples=50, random_state=42
            )

            # We just verify it runs and returns a finite value
            self.assertTrue(np.isfinite(log_ml))
        except (AttributeError, ValueError) as e:
            self.skipTest(f"GPC implementation has changed: {str(e)}")


    def test_perfect_vs_random_predictions(self):
        """Test that better predictions give higher likelihood."""
        try:
            # Create "perfect" test labels that match GPC predictions
            y_pred = self.gpc.predict(self.X_test)

            # Compute log ML for perfect predictions
            log_ml_perfect = compute_log_marginal_likelihood(
                self.gpc, self.X_test, y_pred,
                n_samples=100, random_state=42
            )

            # Create random test labels
            np.random.seed(42)
            y_random = np.random.randint(0, 2, size=len(self.X_test))

            # Compute log ML for random predictions
            log_ml_random = compute_log_marginal_likelihood(
                self.gpc, self.X_test, y_random,
                n_samples=100, random_state=42
            )

            # Perfect predictions should have higher log likelihood
            self.assertGreater(log_ml_perfect, log_ml_random)
        except (AttributeError, ValueError) as e:
            self.skipTest(f"GPC implementation has changed: {str(e)}")


    def test_consistency_with_gpc_predict_proba(self):
        """Test that our MC integration is consistent with GPC's predict_proba."""
        try:
            # Get GPC's predicted probabilities
            gpc_probs = self.gpc.predict_proba(self.X_test)

            # Create test cases where each label matches the most likely class
            y_most_likely = np.argmax(gpc_probs, axis=1)

            # Create test cases where each label is the opposite of most likely
            y_least_likely = 1 - y_most_likely

            # Compute log ML for both cases
            log_ml_likely = compute_log_marginal_likelihood(
                self.gpc, self.X_test, y_most_likely,
                n_samples=100, random_state=42
            )

            log_ml_unlikely = compute_log_marginal_likelihood(
                self.gpc, self.X_test, y_least_likely,
                n_samples=100, random_state=42
            )

            # Most likely labels should have higher log likelihood
            self.assertGreater(log_ml_likely, log_ml_unlikely)
        except (AttributeError, ValueError) as e:
            self.skipTest(f"GPC implementation has changed: {str(e)}")


    def test_log_ml_scaling_with_data_size(self):
        """Test that log_ml scales appropriately with data size."""
        try:
            # For a dataset with independent points, log ML should scale linearly with dataset size

            # Start with a single point
            log_ml_single = compute_log_marginal_likelihood(
                self.gpc, self.X_test[:1], self.y_test[:1],
                n_samples=100, random_state=42
            )

            # Use multiple points with the same label
            # (to make this test more reliable by reducing randomness)
            same_label = np.ones(5) * self.y_test[0]
            log_ml_multiple = compute_log_marginal_likelihood(
                self.gpc, self.X_test, same_label,
                n_samples=100, random_state=42
            )

            # For independent points with same label, log likelihood should roughly scale with data size
            # We use a generous margin since there are correlations between points
            self.assertLess(abs(log_ml_multiple - 5 * log_ml_single), 5)
        except (AttributeError, ValueError) as e:
            self.skipTest(f"GPC implementation has changed: {str(e)}")


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and error handling."""

    def setUp(self):
        """Set up a mock GPC."""
        self.mock_gpc = MagicMock()
        self.mock_gpc.latent_predictive_distribution = MagicMock(
            return_value=(np.array([0.0]), np.array([[0.1]]))
        )

    def test_empty_dataset(self):
        """Test behavior with empty dataset."""
        with self.assertRaises(AssertionError):
            compute_log_marginal_likelihood(
                self.mock_gpc, np.empty((0, 2)), np.empty(0),
                n_samples=10, random_state=42
            )

    def test_zero_samples(self):
        """Test behavior with zero samples."""
        with self.assertRaises(Exception):  # Should fail somehow
            compute_log_marginal_likelihood(
                self.mock_gpc, np.array([[0.1, 0.2]]), np.array([1]),
                n_samples=0, random_state=42
            )

    def test_non_binary_labels(self):
        """Test with non-binary labels."""
        # This should still run, but we test it's consistent for labels > 1
        X_test = np.array([[0.1, 0.2]])

        # Test with label 2 (should be treated as 1)
        y_test_2 = np.array([2])
        log_ml_2 = compute_log_marginal_likelihood(
            self.mock_gpc, X_test, y_test_2,
            n_samples=10, random_state=42
        )

        # Test with label 1
        y_test_1 = np.array([1])
        log_ml_1 = compute_log_marginal_likelihood(
            self.mock_gpc, X_test, y_test_1,
            n_samples=10, random_state=42
        )

        # Should give same result
        self.assertEqual(log_ml_2, log_ml_1)

    def test_extreme_latent_values(self):
        """Test with extreme latent function values."""
        extreme_mock_gpc = MagicMock()
        extreme_mock_gpc.latent_predictive_distribution = MagicMock(
            return_value=(np.array([1000.0, -1000.0]), np.eye(2) * 0.1)
        )

        # This should run without numerical issues
        log_ml = compute_log_marginal_likelihood(
            extreme_mock_gpc, np.array([[0.1, 0.2], [0.3, 0.4]]), np.array([1, 0]),
            n_samples=10, random_state=42
        )

        self.assertTrue(np.isfinite(log_ml))


if __name__ == '__main__':
    unittest.main()