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
    - N_MC is not used (N_cand_LF used to set number of grid samples)
    - N_y_samples is the number of Bernoulli samples used in discrete MI estimate
    - Unlike sampling the latents, we want to be able to decide to sample more than once at a given candidate loc
    """
    def __init__(self, model: BiFidelityModel, dataset: BiFidelityDataset, seed=42, max_pool_subset=50, N_y_samples=50, plot_all_scores=True):
        super().__init__(model, dataset, seed, max_pool_subset, plot_all_scores)
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

    def _max_greedy_acquisition(self, X_G, inds_LF, inds_HF, model, flags, plot=False):
        """Compute acqusition function for each fidelity and each candidate position and return the max
        """
        assert flags.any()

        plot_data = {
            'X_LF': [],
            'X_HF': [],
            'MI_LF': [],
            'MI_HF': []
        }

        # Evaluate on HF at these points - grid points which have not been selected yet
        eval_inds = set(range(len(X_G))).difference(set(inds_LF).union(set(inds_HF)))

        @dataclasses.dataclass
        class Optimum:
            fidelity: int
            mi: float
            cand_ind: int

        optimum = Optimum(-1, -np.inf, -1)  # (Fidelity, MI)

        X_HF_candidates, X_HF_cand_ind_map, X_LF_candidates, X_LF_cand_ind_map, current_proposals = self.assemble_current_proposals(
            X_G, inds_HF, inds_LF)

        X_prime = torch.from_numpy(X_G[list(eval_inds)]).float()
        base_mi = self._estimate_MI(current_proposals, model, X_prime)

        logger.info(f"Number of current proposals: {len(current_proposals)}")
        logger.info(f"Number of X_LF_candidates: {len(X_LF_candidates)}")
        logger.info(f"Number of X_HF_candidates: {len(X_HF_candidates)}")
        logger.info(f"Base MI: {base_mi}")

        if flags[0]:
            # Check all LF candidate points
            # For efficiency, just check a random subset of 50 points if there are more proposals than that
            if len(X_LF_candidates) > self.max_pool_subset:
                inds = np.random.choice(range(len(X_LF_candidates)), self.max_pool_subset, replace=False)
            else:
                inds = range(len(X_LF_candidates))

            for i in inds:
                x = X_LF_candidates[i]
                new_x_prime_inds = eval_inds.difference({X_LF_cand_ind_map[i]})
                X_prime = torch.from_numpy(X_G[list(new_x_prime_inds)]).float()
                mi = self._estimate_MI(current_proposals + [(0, x)], model, X_prime)
                cost_weighted_delta_mi = (mi - base_mi) / self.dataset.c_LF

                plot_data['X_LF'].append(x)
                plot_data['MI_LF'].append(cost_weighted_delta_mi)

                if cost_weighted_delta_mi > optimum.mi:
                    optimum.mi = cost_weighted_delta_mi
                    optimum.fidelity = 0
                    optimum.cand_ind = X_LF_cand_ind_map[i]

                logger.debug(f"LF loop. X_prime.shape={X_prime.shape}, len(current_proposals)={len(current_proposals)}")
                logger.debug(f"LF: {x}, MI={mi:.4f}, WDMI={cost_weighted_delta_mi:.4f}")

        if flags[1]:
            if len(X_HF_candidates) > self.max_pool_subset:
                inds = np.random.choice(range(len(X_HF_candidates)), self.max_pool_subset, replace=False)
            else:
                inds = range(len(X_HF_candidates))

            for i in inds:
                x = X_HF_candidates[i]
                new_x_prime_inds = eval_inds.difference({X_HF_cand_ind_map[i]})
                X_prime = torch.from_numpy(X_G[list(new_x_prime_inds)]).float()
                mi = self._estimate_MI(current_proposals + [(1, x)], model, X_prime)
                cost_weighted_delta_mi = (mi - base_mi) / self.dataset.c_HF

                plot_data['X_HF'].append(x)
                plot_data['MI_HF'].append(cost_weighted_delta_mi)

                if cost_weighted_delta_mi > optimum.mi:
                    optimum.mi = cost_weighted_delta_mi
                    optimum.fidelity = 1
                    optimum.cand_ind = X_HF_cand_ind_map[i]

                logger.debug(f"HF loop. X_prime.shape={X_prime.shape}, len(current_proposals)={len(current_proposals)}")
                logger.debug(f"HF: {x}, MI={mi:.4f}, WDMI={cost_weighted_delta_mi:.4f}")

        logger.info(f"Greedy solve completed, optimum: ({optimum.fidelity},{optimum.cand_ind}) MI={optimum.mi:.4f}")

        if plot:
            #X_prime = X_G[list(eval_inds)]
            #plt.scatter(X_prime[:, 0], X_prime[:, 1], c='blue')
            #X_LF = X_G[inds_LF]
            #plt.scatter(X_LF[:, 0], X_LF[:, 1], c='red', marker='x')
            #X_HF = X_G[inds_HF]
            #plt.scatter(X_HF[:, 0], X_HF[:, 1], c='green', marker='*')
            #plt.title(f"{len(X_HF)}, {len(X_LF)}")
            #plt.show()
            self._plot_all_scores(plot_data)

        assert optimum.cand_ind >= 0

        return optimum.fidelity, optimum.cand_ind

