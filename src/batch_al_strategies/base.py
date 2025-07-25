from abc import ABC, abstractmethod

import numpy as np
import pyDOE

from src.active_learning.util_classes import BiFidelityModel, BiFidelityDataset, ALExperimentConfig


class BiFidelityBatchALStrategy(ABC):
    """Core logic used to select a batch of points
    """
    def __init__(self, dataset: BiFidelityDataset):
        self.dataset = dataset

    @abstractmethod
    def select_batch(self,
                     config: ALExperimentConfig,
                     current_model_trained: BiFidelityModel,  # Pass the currently trained model
                     budget_this_step: float
                     ) -> tuple[np.ndarray, np.ndarray]:  # LF indices from X_LF_cand_pool, HF indices from X_HF_cand_pool
        pass

    @abstractmethod
    def __str__(self):
        pass

    def _generate_lhs_samples(self, config, n_samples: int) -> np.ndarray:
        assert n_samples > 0
        # Use the global seed for LHS if specific random_state is not used by all criteria.
        # For 'maximin', random_state is used.
        samples_unit_hypercube = pyDOE.lhs(2, samples=n_samples)
        scaled_samples = np.zeros_like(samples_unit_hypercube)
        for i in range(2):
            min_val, max_val = config.domain_bounds[i]
            scaled_samples[:, i] = samples_unit_hypercube[:, i] * (max_val - min_val) + min_val
        return scaled_samples
