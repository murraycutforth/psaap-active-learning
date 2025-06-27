# tests/test_mud_strategy.py

import unittest
from unittest.mock import MagicMock, patch
import numpy as np

# Make sure the src path is available
from src.active_learning.strategies.new_strategy import MUDStrategy
from src.active_learning.util_classes import BiFidelityModel, BiFidelityDataset, ALExperimentConfig


# --- Mock Objects for Testing (Corrected) ---

class MockModel(BiFidelityModel):
    """A mock BiFidelityModel that implements all abstract methods."""

    # Methods used by MUDStrategy
    def predict_prob_mean(self, X):
        pass  # Behavior will be mocked in each test

    def predict_prob_var(self, X):
        pass  # Behavior will be mocked in each test

    # --- Dummy implementations to satisfy the abstract base class contract ---
    def train(self, dataset: BiFidelityDataset):
        pass

    def __str__(self):
        return "MockModel"

    def forward(self, x):
        pass

    def predict_f_H(self, X: np.ndarray, full_cov: bool = False):
        pass

    def predict_multi_fidelity_latent_joint(self, X_L: np.ndarray, X_H: np.ndarray):
        pass

    def train_model(self, X_L: np.ndarray, y_L: np.ndarray, X_H: np.ndarray, y_H: np.ndarray):
        pass

    def evaluate_elpp(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        return 0.0


class MockDataset(BiFidelityDataset):
    """A mock BiFidelityDataset that implements all abstract methods."""

    def __init__(self):
        # Pool of available indices to be sampled from
        self.lf_candidate_pool_indices = list(range(200, 220))  # e.g., [200, 201, ...]
        self.hf_candidate_pool_indices = list(range(100, 105))  # e.g., [100, 101, ...]

    def get_lf_candidate_pool_features(self):
        return np.random.rand(len(self.lf_candidate_pool_indices), 2)

    def get_hf_candidate_pool_features(self):
        return np.random.rand(len(self.hf_candidate_pool_indices), 2)

    # --- Dummy implementations to satisfy the abstract base class contract ---
    def add_labels(self, lf_indices: list[int], hf_indices: list[int], lf_labels: np.ndarray, hf_labels: np.ndarray):
        pass

    def get_training_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (np.array([]), np.array([]), np.array([]), np.array([]))


# --- The Test Class ---

class TestMUDStrategy(unittest.TestCase):

    def setUp(self):
        """Set up common objects for all tests."""
        self.mock_model = MockModel()  # This should now instantiate without error
        self.mock_dataset = MockDataset()
        self.config = ALExperimentConfig(cost_HF=10.0, cost_LF=1.0)

    def test_initialization(self):
        """Test the constructor and basic attributes."""
        strategy = MUDStrategy(self.mock_model, self.mock_dataset, beta=0.5, gamma=0.3)
        self.assertEqual(strategy.beta, 0.5)
        self.assertEqual(strategy.gamma, 0.3)
        self.assertIs(strategy.model, self.mock_model)
        self.assertIs(strategy.dataset, self.mock_dataset)

    def test_invalid_gamma_raises_error(self):
        """Test that gamma outside [0, 1] raises a ValueError."""
        with self.assertRaises(ValueError):
            MUDStrategy(self.mock_model, self.mock_dataset, gamma=-0.1)
        with self.assertRaises(ValueError):
            MUDStrategy(self.mock_model, self.mock_dataset, gamma=1.1)

    def test_str_representation(self):
        """Test the __str__ method for clear representation."""
        strategy = MUDStrategy(self.mock_model, self.mock_dataset, beta=1.5, gamma=0.2)
        self.assertEqual(str(strategy), "MUDStrategy(beta=1.5, gamma=0.2)")

    def test_calculate_acquisition_scores(self):
        """Test the core score calculation logic."""
        strategy = MUDStrategy(self.mock_model, self.mock_dataset, beta=1.0)

        X_cand = np.array([[1], [2], [3]])
        mock_means = np.array([0.1, 0.5, 0.99])
        mock_vars = np.array([0.2, 0.05, 0.01])

        # Mock the model's predictions
        self.mock_model.predict_prob_mean = MagicMock(return_value=mock_means)
        self.mock_model.predict_prob_var = MagicMock(return_value=mock_vars)

        entropy_p1 = -0.1 * np.log2(0.1) - 0.9 * np.log2(0.9)
        entropy_p2 = -0.5 * np.log2(0.5) - 0.5 * np.log2(0.5)
        entropy_p3 = -0.99 * np.log2(0.99) - 0.01 * np.log2(0.01)

        expected_scores = np.array([
            0.2 + entropy_p1,
            0.05 + entropy_p2,
            0.01 + entropy_p3
        ])

        scores = strategy._calculate_acquisition_scores(X_cand, self.mock_model)

        self.mock_model.predict_prob_mean.assert_called_once_with(X_cand)
        self.mock_model.predict_prob_var.assert_called_once_with(X_cand)
        np.testing.assert_allclose(scores, expected_scores)

    @patch('src.active_learning.strategies.new_strategy.KMeans')
    def test_select_diverse_batch_from_pool(self, mock_kmeans):
        """Test diversity selection using a mocked KMeans."""
        strategy = MUDStrategy(self.mock_model, self.mock_dataset)

        X_pool = np.array([[0, 0], [0, 1], [10, 10], [10, 11]])
        scores = np.array([0.8, 0.9, 0.5, 0.7])
        n_to_select = 2

        mock_kmeans_instance = MagicMock()
        mock_kmeans_instance.labels_ = np.array([0, 0, 1, 1])
        mock_kmeans.return_value = mock_kmeans_instance

        selected_indices = strategy._select_diverse_batch_from_pool(X_pool, scores, n_to_select)

        expected = [1, 3]
        self.assertCountEqual(selected_indices, expected)

    def test_select_diverse_batch_less_points_than_clusters(self):
        """Test that if n_to_select > len(pool), it selects all available points."""
        strategy = MUDStrategy(self.mock_model, self.mock_dataset)
        X_pool = np.array([[0, 0], [1, 1]])
        scores = np.array([0.8, 0.9])
        n_to_select = 5

        selected_indices = strategy._select_diverse_batch_from_pool(X_pool, scores, n_to_select)

        self.assertCountEqual(selected_indices, [0, 1])

    def test_select_batch_full_flow(self):
        """Test the main select_batch method, including budget split and index mapping."""
        strategy = MUDStrategy(self.mock_model, self.mock_dataset, gamma=0.25)
        budget_this_step = 42.0

        hf_scores = np.array([0.1, 0.9, 0.3, 0.4, 0.2])
        lf_scores = np.linspace(0.1, 0.8, 20)

        strategy._calculate_acquisition_scores = MagicMock(side_effect=[hf_scores, lf_scores])

        def mock_diverse_selection(X_pool, scores, n_to_select):
            if n_to_select == 1:
                return [np.argmax(scores)]
            else:
                return list(np.argsort(scores)[-n_to_select:])

        strategy._select_diverse_batch_from_pool = MagicMock(side_effect=mock_diverse_selection)

        final_lf_indices, final_hf_indices = strategy.select_batch(
            self.config, self.mock_model, budget_this_step
        )

        # Expected budget split:
        # n_hf = floor(0.25 * 42 / 10.0) = floor(1.05) = 1
        # n_lf = floor(0.75 * 42 / 1.0) = floor(31.5) = 31
        # LF pool only has 20 points, so we expect 20.

        self.assertEqual(len(final_hf_indices), 1)
        self.assertEqual(final_hf_indices[0], self.mock_dataset.hf_candidate_pool_indices[1])

        self.assertEqual(len(final_lf_indices), 20)
        self.assertCountEqual(final_lf_indices, self.mock_dataset.lf_candidate_pool_indices)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)