import numpy as np

from src.batch_al_strategies.base import BiFidelityBatchALStrategy
from src.active_learning.util_classes import BiFidelityModel, BiFidelityDataset, ALExperimentConfig


class RandomStrategy(BiFidelityBatchALStrategy):
    """Baseline active learning strategy, randomly pick a fidelity and input coord until budget exhausted
    """
    def __init__(self, dataset: BiFidelityDataset, seed=42, gamma=0.5):
        super().__init__(dataset)
        self.gen = np.random.default_rng(seed=seed)
        self.gamma = gamma
        assert 0.0 <= gamma <= 1.0
        assert dataset.c_HF > 0
        assert dataset.c_LF > 0

    def select_batch(self,
                     config: ALExperimentConfig,
                     current_model_trained: BiFidelityModel,  # Pass the currently trained model
                     budget_this_step: float
                     ) -> tuple[np.ndarray, np.ndarray]:

        X_LF_cand_pool = self._generate_lhs_samples(config, config.N_cand_LF)
        X_HF_cand_pool = self._generate_lhs_samples(config, config.N_cand_HF)

        inds_LF = []
        inds_HF = []
        cost_so_far = 0

        while True:

            # Randomly choose a fidelity, such that on average of gamma resources are devoted to HF
            x = self.gen.random()
            use_LF = x < ((1.0 - self.gamma) * self.dataset.c_HF) / (self.dataset.c_HF + self.gamma * (self.dataset.c_LF - self.dataset.c_HF))

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

        X_LF_new = X_LF_cand_pool[inds_LF]
        X_HF_new = X_HF_cand_pool[inds_HF]

        return X_LF_new, X_HF_new

    def __str__(self):
        return f'RandomStrategy(gamma={self.gamma:.4f})'