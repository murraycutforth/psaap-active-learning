import logging

import numpy as np
import torch

from src.active_learning.util_classes import BiFidelityModel, BiFidelityDataset
from src.batch_al_strategies.mutual_information_strategy_bmfal import MutualInformationBMFALStrategy
from src.batch_al_strategies.entropy_functions import estimate_marginal_entropy_H_Y, estimate_conditional_entropy_H_Y_given_Q

logger = logging.getLogger(__name__)



class MutualInformationBMFALObservables(MutualInformationBMFALStrategy):
    """
    Modification of the BMFAL strategy.
    We now estimate the mutual information between a proposal set of observed variables Y(X), with a set of latent
    function outputs F(X*). Note that from the data processing inequality, MI(Y(X);F(X*)) <= MI(F(X);F(X*)), but the
    degree of information loss depends on the value of F. Estimating this MI is challenging, because the
    """
    def __init__(self, model: BiFidelityModel, dataset: BiFidelityDataset, seed=42, N_test_points=100, max_pool_subset=50, plot_all_scores: bool = False, M: int = 100, K: int=100):
        self.M = M
        self.K = K
        super().__init__(model, dataset, seed, N_test_points, max_pool_subset, plot_all_scores)

    def __str__(self):
        return f'MutualInformationBMFALObservables(M={self.M},K={self.K})'

    def _estimate_MI(self, proposal_set: list, model: BiFidelityModel, X_prime: torch.Tensor) -> float:
        """MC estimate of MI between observables in proposal set, and f_H at sampled x'
        """
        # First assemble tensors of X_L, X_H, and X_prime
        X_L = []
        X_H = []

        for x in proposal_set:
            if x[0] == 0:
                X_L.append(x[1])
            elif x[0] == 1:
                X_H.append(x[1])
            else:
                raise ValueError(f"{x[0]} is not a valid fidelity")

        X_L = torch.from_numpy(np.array(X_L, dtype=np.float32))  # Construct torch.tensor from list of np.arrays is slow
        X_H = torch.from_numpy(np.array(X_H, dtype=np.float32))
        N_L = len(X_L)
        N_H = len(X_H)
        N = N_L + N_H

        if N_L + N_H == 0:
            return 0  # MI of empty set with X_prime

        X_H_and_prime = torch.cat((X_H, X_prime), dim=0)
        uniqueness_test = torch.unique(X_H_and_prime, dim=0)
        assert uniqueness_test.shape[0] == X_H_and_prime.shape[0], f"{X_H_and_prime.shape[0]} != {uniqueness_test.shape[0]}, X_H shape = {X_H.shape}, X_prime shape = {X_prime.shape}, proposal set: {proposal_set}, X_prime = {X_prime}"

        joint_mvn = model.predict_multi_fidelity_latent_joint(X_L, X_H, X_prime)

        H_Y = estimate_marginal_entropy_H_Y(self.M, self.K, joint_mvn, N)
        H_Y_given_Q = estimate_conditional_entropy_H_Y_given_Q(self.M, self.K, joint_mvn, N)
        mi = H_Y - H_Y_given_Q

        return max(mi, 0.0)


