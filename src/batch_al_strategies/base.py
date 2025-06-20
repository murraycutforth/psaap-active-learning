from abc import ABC, abstractmethod

import numpy as np

from src.active_learning.util_classes import BiFidelityModel, BiFidelityDataset, ALExperimentConfig


class BiFidelityBatchALStrategy(ABC):
    """Core logic used to select a batch of points
    """
    def __init__(self, model: BiFidelityModel, dataset: BiFidelityDataset):
        self.model = model
        self.dataset = dataset

    @abstractmethod
    def select_batch(self,
                     config: ALExperimentConfig,
                     current_model_trained: BiFidelityModel,  # Pass the currently trained model
                     budget_this_step: float
                     ) -> tuple[list[int], list[int]]:  # LF indices from X_LF_cand_pool, HF indices from X_HF_cand_pool
        pass

    @abstractmethod
    def __str__(self):
        pass
