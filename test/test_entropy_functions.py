import unittest
import torch
from torch.distributions import MultivariateNormal

# Assuming your functions are in a file named entropy_estimators.py
from src.batch_al_strategies.entropy_functions import (
    estimate_marginal_entropy_H_Y,
    estimate_conditional_entropy_H_Y_given_Q,
)


class TestEntropyEstimators(unittest.TestCase):
    """
    Unit tests for Monte Carlo entropy estimation functions.
    """

    def setUp(self):
        """
        Set up common parameters and a standard joint distribution for all tests.
        This method is run before each test function.
        """
        self.d = 5  # Dimension of candidate batch (R)
        self.dim_Q = 10  # Dimension of target set (Q)
        self.n_total = self.d + self.dim_Q

        # MC parameters - keep them small for fast tests
        self.M = 100
        self.K = 20
        self.seed = 42

        # Create a standard joint MVN distribution with correlations
        torch.manual_seed(self.seed)
        dummy_mean = torch.randn(self.n_total)
        A = torch.randn(self.n_total, self.n_total)
        # Ensure the covariance matrix is positive semi-definite and has non-zero off-diagonals
        dummy_cov = A @ A.T + 1e-3 * torch.eye(self.n_total)
        self.joint_distribution = MultivariateNormal(dummy_mean, dummy_cov)

    def test_smoke_and_output_properties(self):
        """
        Test 1: Basic "smoke test" to ensure functions run without errors
        and return outputs with the correct shape and basic properties.
        """
        # Test H(Y)
        h_y = estimate_marginal_entropy_H_Y(
            M=10, K=5, joint_mvn=self.joint_distribution, d=self.d, seed=self.seed
        )
        self.assertIsInstance(h_y, torch.Tensor)
        self.assertEqual(h_y.shape, torch.Size([]))  # Should be a scalar tensor
        self.assertFalse(torch.isnan(h_y).item())
        self.assertTrue(h_y.item() >= 0)  # Entropy must be non-negative

        # Test H(Y|Q)
        h_y_given_q = estimate_conditional_entropy_H_Y_given_Q(
            M=10, K=5, joint_mvn=self.joint_distribution, d=self.d, seed=self.seed
        )
        self.assertIsInstance(h_y_given_q, torch.Tensor)
        self.assertEqual(h_y_given_q.shape, torch.Size([]))
        self.assertFalse(torch.isnan(h_y_given_q).item())
        self.assertTrue(h_y_given_q.item() >= 0)

    def test_reproducibility_with_seed(self):
        """
        Test 2: Ensure that running the function twice with the same seed
        produces the exact same result.
        """
        # Test H(Y)
        h_y1 = estimate_marginal_entropy_H_Y(
            self.M, self.K, self.joint_distribution, self.d, seed=self.seed
        )
        h_y2 = estimate_marginal_entropy_H_Y(
            self.M, self.K, self.joint_distribution, self.d, seed=self.seed
        )
        self.assertEqual(h_y1.item(), h_y2.item())

        # Test H(Y|Q)
        h_y_q1 = estimate_conditional_entropy_H_Y_given_Q(
            self.M, self.K, self.joint_distribution, self.d, seed=self.seed
        )
        h_y_q2 = estimate_conditional_entropy_H_Y_given_Q(
            self.M, self.K, self.joint_distribution, self.d, seed=self.seed
        )
        self.assertEqual(h_y_q1.item(), h_y_q2.item())

    def test_information_never_hurts_property(self):
        """
        Test 3: Verify the fundamental property that H(Y) >= H(Y|Q).
        Conditioning on a variable (Q) should, on average, reduce entropy.
        The mutual information I(Y;Q) must be non-negative.
        """
        h_y = estimate_marginal_entropy_H_Y(
            self.M, self.K, self.joint_distribution, self.d, seed=self.seed
        )
        h_y_given_q = estimate_conditional_entropy_H_Y_given_Q(
            self.M, self.K, self.joint_distribution, self.d, seed=self.seed
        )
        # We expect MI to be positive since our default cov matrix has correlations
        self.assertGreaterEqual(
            h_y.item(),
            h_y_given_q.item(),
            msg="H(Y) should be greater than or equal to H(Y|Q)"
        )

    def test_zero_mi_for_independent_R_and_Q(self):
        """
        Test 4: Verify a critical edge case. If R and Q are independent, then
        Q gives no information about Y. Therefore, H(Y) should be equal to H(Y|Q),
        and the mutual information should be zero (within MC error).
        """
        # Create a special joint distribution where R and Q are independent.
        # This means the covariance matrix is block-diagonal.
        torch.manual_seed(self.seed)

        # Independent blocks for R and Q
        cov_RR = torch.randn(self.d, self.d)
        cov_RR = cov_RR @ cov_RR.T + 1e-3 * torch.eye(self.d)

        cov_QQ = torch.randn(self.dim_Q, self.dim_Q)
        cov_QQ = cov_QQ @ cov_QQ.T + 1e-3 * torch.eye(self.dim_Q)

        # Zero blocks for off-diagonals
        cov_RQ = torch.zeros(self.d, self.dim_Q)

        # Assemble the block-diagonal covariance matrix
        top_block = torch.cat((cov_RR, cov_RQ), dim=1)
        bottom_block = torch.cat((cov_RQ.T, cov_QQ), dim=1)
        independent_cov = torch.cat((top_block, bottom_block), dim=0)

        mean = torch.randn(self.n_total)
        independent_joint_dist = MultivariateNormal(mean, independent_cov)

        # Estimate the two entropy terms
        h_y = estimate_marginal_entropy_H_Y(
            self.M, self.K, independent_joint_dist, self.d, seed=self.seed
        )
        h_y_given_q = estimate_conditional_entropy_H_Y_given_Q(
            self.M, self.K, independent_joint_dist, self.d, seed=self.seed
        )

        # The values should be very close. We use assertAlmostEqual to account
        # for the variance inherent in Monte Carlo estimation.
        # The delta value might need tuning depending on M and K.
        self.assertAlmostEqual(
            h_y.item(),
            h_y_given_q.item(),
            delta=0.1,  # A reasonable tolerance for small M/K
            msg="For independent R and Q, H(Y) should equal H(Y|Q)"
        )


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)