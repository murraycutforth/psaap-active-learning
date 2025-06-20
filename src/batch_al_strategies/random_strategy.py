import numpy as np

from src.batch_al_strategies.base import BiFidelityBatchALStrategy
from src.active_learning.util_classes import BiFidelityModel, BiFidelityDataset


class RandomStrategy(BiFidelityBatchALStrategy):
    """Baseline active learning strategy, randomly pick a fidelity and input coord until budget exhausted
    """
    def __init__(self, model: BiFidelityModel, dataset: BiFidelityDataset, seed=42):
        super().__init__(model, dataset)
        self.gen = np.random.default_rng(seed=seed)
        assert dataset.c_HF > 0
        assert dataset.c_LF > 0

    def select_batch(self,
                     X_LF_cand_pool: np.ndarray,
                     X_HF_cand_pool: np.ndarray,
                     current_model_trained: BiFidelityModel,  # Pass the currently trained model
                     budget_this_step: float
                     ) -> tuple[list[int], list[int]]:  # LF indices from X_LF_cand_pool, HF indices from X_HF_cand_pool
        inds_LF = []
        inds_HF = []
        cost_so_far = 0

        while True:

            # Randomly choose a fidelity, such that on average equal resources are devoted to each fidelity
            x = self.gen.random()
            use_LF = x < self.dataset.c_HF / (self.dataset.c_HF + self.dataset.c_LF)

            if use_LF:
                # Add LF to pool
                ind = self.gen.choice(len(X_LF_cand_pool), 1, replace=False)[0]
                inds_LF.append(ind)
                cost_so_far += self.dataset.c_LF
            else:
                # Add HF to pool
                ind = self.gen.choice(len(X_HF_cand_pool), 1, replace=False)[0]
                inds_HF.append(ind)
                cost_so_far += self.dataset.c_HF

            if cost_so_far > budget_this_step:
                break

        return inds_LF, inds_HF

    def __str__(self):
        return 'RandomStrategy'