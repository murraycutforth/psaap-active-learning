import unittest
from unittest.mock import MagicMock, patch, call, ANY
import numpy as np
import torch
import dataclasses

from src.batch_al_strategies.mutual_information_strategy_bmfal import MutualInformationBMFALStrategy

BiFidelityModel = MagicMock
BiFidelityDataset = MagicMock
ALExperimentConfig = MagicMock


class TestMutualInformationBMFALStrategy(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures, including mock objects for dependencies."""
        self.mock_model = BiFidelityModel()
        self.mock_dataset = BiFidelityDataset()
        self.mock_config = ALExperimentConfig()

        self.mock_dataset.c_LF = 1.0
        self.mock_dataset.c_HF = 10.0
        self.mock_config.N_cand_LF = 20
        self.mock_config.N_cand_HF = 20
        self.mock_config.domain_bounds = [(0., 1.), (0., 1.)]

        self.strategy = MutualInformationBMFALStrategy(
            model=self.mock_model,
            dataset=self.mock_dataset,
            seed=42,
            N_MC=5,
            max_pool_subset=50
        )

    def test_estimate_MI_with_proposals(self):
        """
        Test the mutual information calculation, mocking the joint MVN from the model.
        """
        # --- Arrange ---
        proposal_set = [
            (0, np.array([0.1, 0.1])),
            (1, np.array([0.2, 0.2]))
        ]
        n_proposals = len(proposal_set)
        n_mc = self.strategy.N_test_points
        X_prime = torch.rand(n_mc, 2)
        mock_mvn = MagicMock()

        A = torch.eye(n_proposals) * 2.0
        C = torch.eye(n_mc) * 3.0
        B = torch.ones(n_proposals, n_mc) * 0.5
        sigma_joint = torch.cat(
            (torch.cat((A, B), dim=1),
             torch.cat((B.T, C), dim=1)),
            dim=0
        )
        mock_mvn.covariance_matrix = sigma_joint
        self.mock_model.predict_multi_fidelity_latent_joint.return_value = mock_mvn

        logdet_A = torch.logdet(A)
        logdet_C = torch.logdet(C)
        logdet_sigma_joint = torch.logdet(sigma_joint)
        expected_mi = 0.5 * (logdet_A + logdet_C - logdet_sigma_joint)

        # --- Act ---
        calculated_mi = self.strategy._estimate_MI(proposal_set, self.mock_model, X_prime)

        # --- Assert ---
        self.mock_model.predict_multi_fidelity_latent_joint.assert_called_once()
        call_args, _ = self.mock_model.predict_multi_fidelity_latent_joint.call_args
        self.assertEqual(call_args[0].shape, (1, 2))
        self.assertEqual(call_args[1].shape, (1, 2))
        self.assertEqual(call_args[2].shape, (n_mc, 2))

        self.assertAlmostEqual(calculated_mi, expected_mi.item(), places=5)

    def test_estimate_MI_no_proposals(self):
        """
        Test that MI is 0 when the proposal set is empty, without calling the model.
        """
        calculated_mi = self.strategy._estimate_MI([], self.mock_model, torch.rand(self.strategy.N_test_points, 2))
        self.assertEqual(calculated_mi, 0)
        self.mock_model.predict_multi_fidelity_latent_joint.assert_not_called()

    def test_check_fidelity_feasibility(self):
        """Test the budget feasibility check for different scenarios."""
        budget = 50.0
        self.strategy.dataset.c_LF = 10.0
        self.strategy.dataset.c_HF = 45.0
        np.testing.assert_array_equal(self.strategy._check_fidelity_feasibility(0.0, budget), [True, True])
        np.testing.assert_array_equal(self.strategy._check_fidelity_feasibility(10.0, budget), [True, False])
        np.testing.assert_array_equal(self.strategy._check_fidelity_feasibility(41.0, budget), [False, False])

    @patch.object(MutualInformationBMFALStrategy, '_estimate_MI', autospec=True)
    def test_max_greedy_acquisition_selects_optimal_fidelity(self, mock_estimate_mi):
        """Test that the greedy acquisition selects the point with the highest cost-weighted MI."""
        mock_estimate_mi.side_effect = [0.0, 2.0, 15.0]
        X_LF = np.array([[0.1, 0.2]])
        X_HF = np.array([[0.8, 0.9]])
        flags = np.array([True, True])
        fidelity, ind = self.strategy._max_greedy_acquisition(X_LF, X_HF, [], [], self.mock_model, flags)
        self.assertEqual(fidelity, 0)
        self.assertEqual(ind, 0)
        self.assertEqual(mock_estimate_mi.call_count, 3)

    @patch.object(MutualInformationBMFALStrategy, '_estimate_MI', autospec=True)
    def test_max_greedy_acquisition_respects_feasibility_flags(self, mock_estimate_mi):
        """Test that the greedy acquisition ignores non-feasible fidelities."""
        mock_estimate_mi.side_effect = [0.0, 1.0]
        X_LF = np.array([[0.1, 0.2]])
        X_HF = np.array([[0.8, 0.9]])
        flags = np.array([True, False])
        fidelity, ind = self.strategy._max_greedy_acquisition(X_LF, X_HF, [], [], self.mock_model, flags)
        self.assertEqual(fidelity, 0)
        self.assertEqual(ind, 0)
        self.assertEqual(mock_estimate_mi.call_count, 2)

    @patch.object(MutualInformationBMFALStrategy, '_max_greedy_acquisition', autospec=True)
    @patch.object(MutualInformationBMFALStrategy, '_generate_lhs_samples', autospec=True)
    def test_select_batch_orchestration(self, mock_generate_lhs, mock_max_greedy):
        """Test the main select_batch method's loop and logic."""
        # --- Arrange ---
        strategy = MutualInformationBMFALStrategy(
            model=self.mock_model,
            dataset=self.mock_dataset,
            seed=42,
            N_MC=5,
            max_pool_subset=50
        )
        strategy.dataset.c_LF = 1.0
        strategy.dataset.c_HF = 10.0

        X_LF_cand_pool = np.arange(40, dtype=float).reshape(20, 2) / 40.0
        X_HF_cand_pool = np.arange(40, 80, dtype=float).reshape(20, 2) / 80.0
        mock_generate_lhs.side_effect = [X_LF_cand_pool, X_HF_cand_pool]

        budget_this_step = 12.0

        # THIS IS THE KEY: A side_effect function that acts as a "spy"
        def greedy_spy_side_effect(slf, x_lf, x_hf, inds_lf, inds_hf, model, flags):
            """
            This function is called INSTEAD of the mock's default behavior.
            It allows us to assert the state of arguments AT THE TIME OF THE CALL.
            """
            # Use the mock's own call_count to know which iteration we are in.
            call_number = mock_max_greedy.call_count
            if call_number == 1:
                # On the first call, the lists must be empty.
                self.assertEqual(inds_lf, [])
                self.assertEqual(inds_hf, [])
                # Return the value the code expects for the first iteration.
                return (1, 5)
            elif call_number == 2:
                # On the second call, HF list should be populated.
                self.assertEqual(inds_lf, [])
                self.assertEqual(inds_hf, [5])
                # Return the value for the second iteration.
                return (0, 8)
            else:
                # Fail fast if called more times than expected.
                self.fail(f"Mock called unexpectedly {call_number} times.")

        # Assign our spy function to the mock's side_effect
        mock_max_greedy.side_effect = greedy_spy_side_effect

        # --- Act ---
        X_LF_new, X_HF_new = strategy.select_batch(
            self.mock_config, self.mock_model, budget_this_step
        )

        # --- Assert ---
        # The primary assertions on the final output are still crucial.
        np.testing.assert_array_equal(X_LF_new, X_LF_cand_pool[[8]])
        np.testing.assert_array_equal(X_HF_new, X_HF_cand_pool[[5]])

        # We can now simply assert the mock was called the correct number of times,
        # as the detailed argument checks happened inside our spy.
        self.assertEqual(mock_max_greedy.call_count, 2)
        self.assertEqual(mock_generate_lhs.call_count, 2)


