import unittest
from scipy.stats import norm

from src.models.laplace_approximation import laplace_approximation_probit


# ==============================================================================
# The Unit Test Class
# ==============================================================================

class TestLaplaceApproximation(unittest.TestCase):

    def test_no_data_returns_prior(self):
        """
        GIVEN no new data (N=0),
        WHEN the approximation is computed,
        THEN the posterior should be identical to the prior.
        """
        mu_prior, sigma_prior = 1.5, 0.8
        mu_post, sigma_post = laplace_approximation_probit(mu_prior, sigma_prior, N=0, k=0)
        self.assertAlmostEqual(mu_post, mu_prior, places=6)
        self.assertAlmostEqual(sigma_post, sigma_prior, places=6)

    def test_strong_prior_dominates_with_little_data(self):
        """
        GIVEN a very confident (low variance) prior,
        WHEN a small amount of contradictory data is observed,
        THEN the posterior mean should remain very close to the prior mean.
        """
        mu_prior, sigma_prior = 2.0, 0.05  # Very confident prior that f is ~2.0
        # Data suggests k/N = 1/10 = 0.1, which implies f is negative.
        N, k = 10, 1

        mu_post, sigma_post = laplace_approximation_probit(mu_prior, sigma_prior, N, k)

        # The posterior should be pulled slightly from the prior, but still dominated by it.
        self.assertAlmostEqual(mu_post, mu_prior, delta=0.1)  # Check it hasn't moved far
        self.assertLess(mu_post, mu_prior)  # Check it moved in the right direction
        self.assertLess(sigma_post, sigma_prior)  # Uncertainty must decrease

    def test_vague_prior_is_driven_by_data(self):
        """
        GIVEN a vague (high variance) prior,
        WHEN data is observed,
        THEN the posterior mean should be close to the value implied by the data.
        """
        mu_prior, sigma_prior = 0.0, 5.0  # Vague prior centered at 0
        N, k = 20, 17  # k/N = 0.85

        # The f* value that would give a probability of 0.85
        f_implied_by_data = norm.ppf(k / N)  # approx 1.036

        mu_post, sigma_post = laplace_approximation_probit(mu_prior, sigma_prior, N, k)

        # The posterior should be very close to the data's implied value
        self.assertAlmostEqual(mu_post, f_implied_by_data, places=1)
        self.assertLess(sigma_post, sigma_prior)  # Uncertainty must decrease drastically

    def test_variance_decreases_with_more_data(self):
        """
        GIVEN two scenarios with the same prior and data proportion but different N,
        WHEN the approximations are computed,
        THEN the scenario with larger N should have a smaller posterior variance.
        """
        mu_prior, sigma_prior = 0.5, 1.0

        # Scenario 1: Small N
        _, sigma_post_small_N = laplace_approximation_probit(mu_prior, sigma_prior, N=10, k=7)

        # Scenario 2: Large N, same proportion
        _, sigma_post_large_N = laplace_approximation_probit(mu_prior, sigma_prior, N=100, k=70)

        self.assertLess(sigma_post_large_N, sigma_post_small_N)
        # The variance for N=100 should be roughly 1/10th the variance for N=10
        # Here we test for a significant reduction
        self.assertLess(sigma_post_large_N, sigma_post_small_N * 0.5)

    def test_all_successes_pushes_mean_up(self):
        """
        GIVEN data is all successes (k=N),
        WHEN the approximation is computed,
        THEN the posterior mean should be significantly higher than the prior mean.
        """
        mu_prior, sigma_prior = 0.0, 1.0
        N, k = 50, 50

        mu_post, sigma_post = laplace_approximation_probit(mu_prior, sigma_prior, N, k)

        self.assertGreater(mu_post, mu_prior)
        self.assertGreater(mu_post, 2.0)  # For N=30, k=30, the mean should be quite high
        self.assertLess(sigma_post, sigma_prior)

    def test_all_failures_pushes_mean_down(self):
        """
        GIVEN data is all failures (k=0),
        WHEN the approximation is computed,
        THEN the posterior mean should be significantly lower than the prior mean.
        """
        mu_prior, sigma_prior = 0.0, 1.0
        N, k = 50, 0

        mu_post, sigma_post = laplace_approximation_probit(mu_prior, sigma_prior, N, k)

        self.assertLess(mu_post, mu_prior)
        self.assertLess(mu_post, -2.0)  # For N=30, k=0, the mean should be quite low
        self.assertLess(sigma_post, sigma_prior)

    def test_symmetric_case(self):
        """
        GIVEN a symmetric prior (mu=0) and symmetric data (k=N/2),
        WHEN the approximation is computed,
        THEN the posterior mean should remain close to 0.
        """
        mu_prior, sigma_prior = 0.0, 1.5
        N, k = 50, 25

        mu_post, sigma_post = laplace_approximation_probit(mu_prior, sigma_prior, N, k)

        # The MAP should be exactly 0 due to symmetry.
        self.assertAlmostEqual(mu_post, 0.0, places=5)
        self.assertLess(sigma_post, sigma_prior)


# ==============================================================================
# Runner
# ==============================================================================

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)