import unittest
import torch
import gpytorch

from src.bfgpc import _assemble_T, BFGPC_ELBO


class TestPredictMultiFidelityLatentJoint(unittest.TestCase):

    def setUp(self):
        # Create dummy tensors for BFGPC_ELBO initialization (not directly used by mocks)
        self.dummy_train_x = torch.tensor([[0.0]], dtype=torch.float)  # 1D input
        self.model = BFGPC_ELBO(self.dummy_train_x, self.dummy_train_x, initial_rho=1.5)

    def _get_expected_outputs(self, X_L, X_H, X_prime, rho_val):
        """Helper to manually calculate expected T, mu_G, sigma_G for verification"""
        N_L, N_H, N_prime = X_L.shape[0], X_H.shape[0], X_prime.shape[0]

        # Simulate the unique operations
        # Ensure consistent feature dimension for cat, similar to the main function
        default_dim = 1
        if N_L == 0 and N_H == 0 and N_prime == 0:
            _X_L = torch.empty((0, default_dim), dtype=X_L.dtype, device=X_L.device)
            _X_H = torch.empty((0, default_dim), dtype=X_H.dtype, device=X_H.device)
            _X_prime = torch.empty((0, default_dim), dtype=X_prime.dtype, device=X_prime.device)
        else:
            dims = []
            if N_L > 0: dims.append(X_L.shape[1])
            if N_H > 0: dims.append(X_H.shape[1])
            if N_prime > 0: dims.append(X_prime.shape[1])
            target_dim = dims[0] if dims else default_dim
            _X_L = X_L if N_L > 0 else torch.empty((0, target_dim), dtype=X_L.dtype, device=X_L.device)
            _X_H = X_H if N_H > 0 else torch.empty((0, target_dim), dtype=X_H.dtype, device=X_H.device)
            _X_prime = X_prime if N_prime > 0 else torch.empty((0, target_dim), dtype=X_prime.dtype,
                                                               device=X_prime.device)

        all_L_points = torch.cat([_X_L, _X_H, _X_prime], dim=0)
        if all_L_points.numel() == 0:
            X_L_unique_eval = torch.empty((0, all_L_points.shape[1]), dtype=all_L_points.dtype)
            inverse_indices_L = torch.empty(0, dtype=torch.long)
        else:
            X_L_unique_eval, inverse_indices_L = torch.unique(all_L_points, dim=0, return_inverse=True)
        N_f_L_unique = X_L_unique_eval.shape[0]

        all_delta_points = torch.cat([_X_H, _X_prime], dim=0)
        if all_delta_points.numel() == 0:
            X_delta_unique_eval = torch.empty((0, all_delta_points.shape[1]), dtype=all_delta_points.dtype)
            inverse_indices_delta = torch.empty(0, dtype=torch.long)
        else:
            X_delta_unique_eval, inverse_indices_delta = torch.unique(all_delta_points, dim=0, return_inverse=True)
        N_f_delta_unique = X_delta_unique_eval.shape[0]

        T_expected = _assemble_T(N_H, N_L, N_f_L_unique, N_f_delta_unique, N_prime,
                                 inverse_indices_L, inverse_indices_delta, rho_val)

        # Get outputs from mock models
        mock_lf_output = self.model.lf_model(X_L_unique_eval)
        mock_delta_output = self.model.delta_model(X_delta_unique_eval)

        mu_G_expected = torch.cat([mock_lf_output.mean, mock_delta_output.mean], dim=0)

        cov_list = []
        if mock_lf_output.covariance_matrix.numel() > 0 or mock_lf_output.covariance_matrix.shape == (0, 0):
            cov_list.append(mock_lf_output.covariance_matrix)
        if mock_delta_output.covariance_matrix.numel() > 0 or mock_delta_output.covariance_matrix.shape == (0, 0):
            cov_list.append(mock_delta_output.covariance_matrix)

        if not cov_list:
            sigma_G_expected = torch.empty((0, 0), dtype=mu_G_expected.dtype, device=mu_G_expected.device)
        else:
            valid_covs = [c for c in cov_list if c.shape[0] > 0 or c.shape == (0, 0)]
            if not valid_covs:
                sigma_G_expected = torch.empty((0, 0), dtype=mu_G_expected.dtype, device=mu_G_expected.device)
            else:
                sigma_G_expected = torch.block_diag(*valid_covs)

        expected_mu = T_expected @ mu_G_expected
        expected_sigma = T_expected @ sigma_G_expected @ T_expected.T

        if expected_sigma.numel() > 0:
            expected_sigma += 1e-6 * torch.eye(expected_sigma.shape[0], device=expected_sigma.device,
                                               dtype=expected_sigma.dtype)
        elif expected_sigma.shape == (0, 0) and (N_L + N_H + N_prime) > 0:
            expected_sigma = 1e-6 * torch.eye(N_L + N_H + N_prime, device=expected_sigma.device,
                                              dtype=expected_sigma.dtype)

        return expected_mu, expected_sigma

    def run_test_scenario(self, X_L, X_H, X_prime, test_description):
        # print(f"\nRunning: {test_description}")
        # print(f"X_L: {X_L.shape}, X_H: {X_H.shape}, X_prime: {X_prime.shape}")

        rho_val = self.model.rho.item()
        expected_mu, expected_sigma = self._get_expected_outputs(X_L, X_H, X_prime, rho_val)

        output_dist = self.model.predict_multi_fidelity_latent_joint(X_L, X_H, X_prime)

        self.assertIsInstance(output_dist, gpytorch.distributions.MultivariateNormal,
                              f"{test_description}: Output type mismatch")

        torch.testing.assert_close(output_dist.mean, expected_mu,
                                   msg=f"{test_description}: Mean mismatch")
        torch.testing.assert_close(output_dist.covariance_matrix, expected_sigma,
                                   msg=f"{test_description}: Covariance mismatch")

        expected_total_N = X_L.shape[0] + X_H.shape[0] + X_prime.shape[0]
        self.assertEqual(output_dist.mean.shape[0], expected_total_N,
                         f"{test_description}: Mean shape mismatch")
        self.assertEqual(output_dist.covariance_matrix.shape, (expected_total_N, expected_total_N),
                         f"{test_description}: Covariance shape mismatch")

    def test_basic_case_all_non_empty(self):
        X_L = torch.tensor([[1.0], [2.0]], dtype=torch.float)
        X_H = torch.tensor([[1.0], [3.0]], dtype=torch.float)  # X_H[0] is same as X_L[0]
        X_prime = torch.tensor([[4.0]], dtype=torch.float)
        self.run_test_scenario(X_L, X_H, X_prime, "Basic case: all non-empty")

    def test_X_L_empty(self):
        X_L = torch.empty((0, 1), dtype=torch.float)  # Ensure 2D for cat
        X_H = torch.tensor([[1.0], [3.0]], dtype=torch.float)
        X_prime = torch.tensor([[4.0]], dtype=torch.float)
        self.run_test_scenario(X_L, X_H, X_prime, "X_L empty")

    def test_X_H_empty(self):
        X_L = torch.tensor([[1.0], [2.0]], dtype=torch.float)
        X_H = torch.empty((0, 1), dtype=torch.float)
        X_prime = torch.tensor([[4.0]], dtype=torch.float)
        self.run_test_scenario(X_L, X_H, X_prime, "X_H empty")

    def test_X_L_and_X_H_empty(self):
        X_L = torch.empty((0, 1), dtype=torch.float)
        X_H = torch.empty((0, 1), dtype=torch.float)
        X_prime = torch.tensor([[4.0], [5.0]], dtype=torch.float)
        self.run_test_scenario(X_L, X_H, X_prime, "X_L and X_H empty")

    def test_unique(self):
        X_L = torch.tensor([[1.0], [2.0]], dtype=torch.float)
        X_H = torch.tensor([[4.0], [3.0]], dtype=torch.float)
        X_prime = torch.tensor([[5.0]], dtype=torch.float)
        self.run_test_scenario(X_L, X_H, X_prime, "X_L non unique")

    def test_non_unique(self):
        X_L = torch.tensor([[1.0], [2.0]], dtype=torch.float)
        X_H = torch.tensor([[1.0], [2.0]], dtype=torch.float)
        X_prime = torch.tensor([[1.0]], dtype=torch.float)
        self.run_test_scenario(X_L, X_H, X_prime, "X_L non unique")


if __name__ == '__main__':
    unittest.main(exit=False)