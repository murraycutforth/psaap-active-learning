import dataclasses
import logging
import math
from collections import Counter

import numpy as np
import torch

from src.active_learning.util_classes import BiFidelityModel, BiFidelityDataset, ALExperimentConfig
from src.batch_al_strategies.mutual_information_strategy_grid_latents import MutualInformationGridStrategy

logger = logging.getLogger(__name__)


class MutualInformationGridStrategyObservables(MutualInformationGridStrategy):
    """
    This class modifies the mutual information approach as follows.

    - We now estimate the MI between the subsets of the Bernoulli outcomes (not the latents)
    """
    def __init__(self, model: BiFidelityModel, dataset: BiFidelityDataset, seed=42, N_MC=100, N_y_samples=42):
        super().__init__(model, dataset, seed, N_MC)
        self.N_y_samples = N_y_samples

    def __str__(self):
        return "MutualInformationGridStrategyObservables"

    def _calculate_entropy_from_samples(self, samples: torch.Tensor) -> float:
        """
        Calculates the empirical entropy H(X) from a set of samples.
        H(X) = - sum_{x} p(x) log(p(x))

        Args:
            samples (torch.Tensor): A tensor of shape (n_samples, n_dimensions)
                                    where each row is a sample.

        Returns:
            float: The estimated entropy in nats.
        """
        if samples.shape[1] == 0:
            return 0.0

        counts = Counter(map(tuple, samples.tolist()))

        n_total_samples = samples.shape[0]
        entropy = 0.0

        for count in counts.values():
            p = count / n_total_samples
            if p > 0:
                entropy -= p * math.log(p)  # log is base e (nats)

        return entropy

    def _calculate_entropy_from_samples_miller_madow(self, samples: torch.Tensor) -> float:
        """
        Calculates the empirical entropy H(X) from a set of samples with Miller-Madow correction for small sample size
        """
        n_samples, n_dim = samples.shape
        if n_dim == 0 or n_samples == 0:
            return 0.0

        counts = Counter(map(tuple, samples.tolist()))
        k_observed = len(counts)

        entropy_ml = 0.0
        for count in counts.values():
            p = count / n_samples
            entropy_ml -= p * math.log(p)

        # Additive bias correction: H_MM = H + (k_observed - 1) / (2 * N)
        bias_correction = (k_observed - 1) / (2 * n_samples)
        return entropy_ml + bias_correction

    def _estimate_MI(self, proposal_set: list, model: BiFidelityModel, X_prime: torch.Tensor) -> float:
        """MC estimate of MI between outputs at x, and outputs at sampled x'

        Note that we compute the joint distribution with all the MC points in one go.
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

        N_proposal = N_L + N_H
        N_prime = len(X_prime)

        if N_proposal == 0 and N_prime == 0:
            return 0.0
        if N_proposal == 0:
            return 0.0
        if N_prime == 0:
            return 0.0

        joint_mvn = model.predict_multi_fidelity_latent_joint(X_L, X_H, X_prime)

        # Sample N_y_samples of joint_mvn
        # For each vector F, apply probit component-wise, and then get a single Bernoulli sample of outcomes Y
        # Shape: (N_y_samples, N_proposal + N_prime)
        latent_samples = joint_mvn.sample(torch.Size((self.N_y_samples,)))

        probit_link = torch.distributions.Normal(0, 1).cdf
        outcome_probabilities = probit_link(latent_samples)
        y_joint_samples = torch.bernoulli(outcome_probabilities)

        y_proposal_samples = y_joint_samples[:, :N_proposal]
        y_prime_samples = y_joint_samples[:, N_proposal:]

        entropy_fn = self._calculate_entropy_from_samples_miller_madow

        H_y = entropy_fn(y_proposal_samples)
        H_y_prime = entropy_fn(y_prime_samples)
        H_joint = entropy_fn(y_joint_samples)
        mi = H_y + H_y_prime - H_joint

        # MI should be non-negative, but numerical errors in MC estimation
        # can sometimes lead to small negative values.
        return max(0.0, mi)

