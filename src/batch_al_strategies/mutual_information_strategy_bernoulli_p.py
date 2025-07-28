import dataclasses
import logging
from pathlib import Path

import numpy as np
import pyDOE
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.batch_al_strategies.base import BiFidelityBatchALStrategy
from src.active_learning.util_classes import BiFidelityModel, BiFidelityDataset, ALExperimentConfig
from src.batch_al_strategies.mutual_information_strategy_bmfal import MutualInformationBMFALStrategy

logger = logging.getLogger(__name__)


class MutualInformationBernoulliPStrategy(MutualInformationBMFALStrategy):
    """
    Batch, multi-fidelity active learning with MI-based acquisition function.

    - Extension so that instead of looking MI between latent functions, we look at MI between
    the Bernoulli parameters: I( {p_i} ; {p_j})
    - Here, {p_i} is the set of bernoulli parameters at proposal points, and {p_j} is the set of
    bernoulli parameters at test points
    - We approximate {p_i} and {p_j} using a linear approximation of the link function.
    - This approach means that the mutual information gain will correctly account for saturation effects.

    This class makes heavy use of inherited methods from MutualInformationBMFALStrategy and just
    re-defines the compute_MI method to reflect the alternative mutual information strategy.
    """
    def __init__(self, dataset: BiFidelityDataset, N_test_points=100, max_pool_subset=50, plot_all_scores: bool = False):
        super().__init__(dataset, N_test_points=N_test_points, max_pool_subset=max_pool_subset, plot_all_scores=plot_all_scores)

    def __str__(self):
        return 'MutualInformationBernoulliPStrategy'

    def _estimate_MI(self, proposal_set: list, model: BiFidelityModel, X_prime: torch.Tensor) -> float:
        """MC estimate of MI between outputs at x, and outputs at sampled x'

        Note that we compute the joint distribution with all the MC points in one go
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

        if N_L + N_H == 0:
            return 0  # MI of empty set with X_prime

        X_H_and_prime = torch.cat((X_H, X_prime), dim=0)
        uniqueness_test = torch.unique(X_H_and_prime, dim=0)
        assert uniqueness_test.shape[0] == X_H_and_prime.shape[0], f"{X_H_and_prime.shape[0]} != {uniqueness_test.shape[0]}, X_H shape = {X_H.shape}, X_prime shape = {X_prime.shape}, proposal set: {proposal_set}, X_prime = {X_prime}"

        joint_mvn = model.predict_multi_fidelity_latent_joint(X_L, X_H, X_prime)

        """Now, the first N_L + N_H components of this joint distribution are the proposal points,
        and the remaining components are for the test points. We use a linear approximation to the
        link function to transform this joint distribution between latent functions to a joint
        distribution between approximate bernoulli parameters. As a result, the saturation of these
        values reduces the information gain where f is already very positive of very negative.
        """

        # Define the derivative of the probit link function (Gaussian PDF)
        normal_dist = torch.distributions.Normal(0.0, 1.0)
        phi_prime = lambda x: normal_dist.log_prob(x).exp()

        # Extract the full mean and covariance of the joint latent distribution
        mu_joint = joint_mvn.mean
        sigma_joint = joint_mvn.covariance_matrix

        # --- This is the key change: linearize around the entire mean vector ---
        # The derivative matrix D will now apply to both proposal and prime variables
        d_full = phi_prime(mu_joint)
        D_full = torch.diag(d_full)

        # Transform the entire joint covariance matrix in one go
        # Cov([Z_prop, Z_prime]) = D_full * Cov([F_prop, F_prime]) * D_full
        sigma_joint_new = D_full @ sigma_joint @ D_full

        # --- Step 3: Compute MI using the new joint distribution ---

        # Add jitter for numerical stability, especially if saturation is high
        jitter = 1e-6
        sigma_joint_new = sigma_joint_new + torch.eye(sigma_joint_new.shape[0], device=mu_joint.device) * jitter

        # Re-extract marginals after jittering the joint matrix
        sigma_z_proposal = sigma_joint_new[:N_proposal, :N_proposal]
        sigma_z_prime = sigma_joint_new[N_proposal:, N_proposal:]

        # Calculate log-determinants for the MI formula
        logdet_sigma_z_proposal = torch.logdet(sigma_z_proposal)
        logdet_sigma_z_prime = torch.logdet(sigma_z_prime)
        logdet_sigma_joint_new = torch.logdet(sigma_joint_new)

        # The analytical MI formula for the approximated Gaussian distribution
        mi = 0.5 * (logdet_sigma_z_proposal + logdet_sigma_z_prime - logdet_sigma_joint_new)

        # Ensure the result is non-negative
        return max(0.0, mi.item())
