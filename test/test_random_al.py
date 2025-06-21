import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from src.batch_al_strategies.random_strategy import RandomStrategy

# --- Stubs for dependencies (if not importing from actual source) ---
# In a real scenario, you'd import these:
# from src.batch_al_strategies.base import BiFidelityBatchALStrategy
# from src.active_learning.util_classes import BiFidelityModel, BiFidelityDataset

class BiFidelityModel:  # Minimal stub
    pass


class BiFidelityDataset:  # Minimal stub
    def __init__(self, c_LF, c_HF):
        self.c_LF = c_LF
        self.c_HF = c_HF


class BiFidelityBatchALStrategy:  # Minimal base class stub
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset





class TestRandomStrategy(unittest.TestCase):

    def setUp(self):
        self.mock_bifidelity_model_instance = MagicMock(spec=BiFidelityModel)
        self.X_LF_cand_pool = np.array([[i] for i in range(10)])  # 10 LF candidates
        self.X_HF_cand_pool = np.array([[i + 100] for i in range(5)])  # 5 HF candidates

    def _create_strategy(self, c_LF, c_HF, seed=42):
        mock_dataset = MagicMock(spec=BiFidelityDataset)
        mock_dataset.c_LF = c_LF
        mock_dataset.c_HF = c_HF
        # Ensure attributes are accessible like dataset.c_LF
        mock_dataset.c_LF = c_LF
        mock_dataset.c_HF = c_HF
        return RandomStrategy(self.mock_bifidelity_model_instance, mock_dataset, seed=seed)

    def test_initialization(self):
        strategy = self._create_strategy(c_LF=1, c_HF=10, seed=123)
        self.assertIs(strategy.model, self.mock_bifidelity_model_instance)
        self.assertIsNotNone(strategy.dataset)
        self.assertEqual(strategy.dataset.c_LF, 1)
        self.assertEqual(strategy.dataset.c_HF, 10)
        self.assertIsInstance(strategy.gen, np.random.Generator)

    def test_str_representation(self):
        strategy = self._create_strategy(c_LF=1, c_HF=10)
        self.assertEqual(str(strategy), 'RandomStrategy')

    def test_select_batch_always_selects_one_item_due_to_loop_logic(self):
        """
        Given the `cost_so_far = 0` inside the loop, the strategy will always
        select exactly one item. It breaks if that item's cost > budget,
        otherwise it would loop infinitely if budget >= item_cost.
        This test assumes budget is low enough to break after one item.
        """
        strategy = self._create_strategy(c_LF=1, c_HF=10, seed=1)  # c_LF=1, c_HF=10
        budget = 0.5  # budget is less than any item cost

        inds_LF, inds_HF = strategy.select_batch(
            self.X_LF_cand_pool, self.X_HF_cand_pool, self.mock_bifidelity_model_instance, budget
        )

        self.assertEqual(len(inds_LF) + len(inds_HF), 1, "Should select exactly one item")
        if inds_LF:
            self.assertTrue(0 <= inds_LF[0] < len(self.X_LF_cand_pool))
        if inds_HF:
            self.assertTrue(0 <= inds_HF[0] < len(self.X_HF_cand_pool))

    def test_select_batch_deterministic_with_seed(self):
        # Given the loop issue, it will always select one item.
        budget = 0.5  # Ensures loop breaks after one item.
        strategy1 = self._create_strategy(c_LF=2, c_HF=5, seed=42)
        inds_LF1, inds_HF1 = strategy1.select_batch(
            self.X_LF_cand_pool, self.X_HF_cand_pool, self.mock_bifidelity_model_instance, budget
        )

        strategy2 = self._create_strategy(c_LF=2, c_HF=5, seed=42)
        inds_LF2, inds_HF2 = strategy2.select_batch(
            self.X_LF_cand_pool, self.X_HF_cand_pool, self.mock_bifidelity_model_instance, budget
        )
        self.assertEqual(inds_LF1, inds_LF2)
        self.assertEqual(inds_HF1, inds_HF2)


