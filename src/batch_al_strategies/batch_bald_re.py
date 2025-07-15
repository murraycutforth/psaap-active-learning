import dataclasses
import logging

import numpy as np
import pyDOE
import torch

from src.batch_al_strategies.base import BiFidelityBatchALStrategy
from src.active_learning.util_classes import BiFidelityModel, BiFidelityDataset, ALExperimentConfig

from toma import toma
from tqdm import tqdm
import math

from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


def compute_expected_conditional_entropy(log_probs_N_M_C: torch.Tensor) -> torch.Tensor:
    # log_probs_N_M_C = log p(Y|x,w) should contain log probabilities for Bernoulli outcomes
    # Shape: (N, M, 2) where N is number of samples, M is number of MC samples

    # Second term in equation (8) in https://proceedings.neurips.cc/paper_files/paper/2019/file/95323660ed2124450caaac2c46b5ed90-Paper.pdf
    # E_w[H(Y|x,w)] = E_w[-E_Y[log p(Y|x,w)]] = -E_w[E_Y[log p(Y|x,w)]]
    # \approx -1/M \sum_{i=1}^N \sum_{j=1}^M log p(Y_i|x,\theta_j)
    N, M, _ = log_probs_N_M_C.shape
    entropies_N = torch.empty(N, dtype=torch.double)
    pbar = tqdm(total=N, desc="Conditional Entropy", leave=False)
    @toma.execute.chunked(log_probs_N_M_C, 1024)
    def compute(log_probs_n_M_C, start: int, end: int):
        nats_n_M_C = log_probs_n_M_C * torch.exp(log_probs_n_M_C)
        entropies_N[start:end].copy_(-torch.sum(nats_n_M_C, dim=(1, 2)) / M)
        pbar.update(end - start)
    pbar.close()
    return entropies_N


def compute_predicitive_entropy(log_probs_N_M_C: torch.Tensor) -> torch.Tensor:
    # log_probs_N_M should contain log probabilities for Bernoulli outcomes
    # Shape: (N, M, 2) where N is number of samples, M is number of MC samples

    # First term in equation (8) in https://proceedings.neurips.cc/paper_files/paper/2019/file/95323660ed2124450caaac2c46b5ed90-Paper.pdf
    # H(Y|x) = -E_Y[log p(Y|x)] = -\sum_{=1}^C p(y=c|x) log p(y=c|x) 
    # where C is the number of classes, i.e. 2 in our case 
    # and p(y=c|x) \approx 1/M \sum_{i=1}^M p(y=c|x, \theta_i) MC integration

    N, M, _ = log_probs_N_M_C.shape
    
    entropies_N = torch.empty(N, dtype=torch.double)

    pbar = tqdm(total=N, desc="Predicitive Entropy", leave=False)

    @toma.execute.chunked(log_probs_N_M_C, 1024)
    def compute(log_probs_n_M_C, start: int, end: int):
        mean_log_probs_n_C = torch.logsumexp(log_probs_n_M_C, dim=1) - math.log(M)
        nats_n_C = mean_log_probs_n_C * torch.exp(mean_log_probs_n_C)

        entropies_N[start:end].copy_(-torch.sum(nats_n_C, dim=1))
        pbar.update(end - start)

    pbar.close()
    return entropies_N


def StochasticGreedySubmodularMaximization(max_loop: int):
    # TODO: if we treat the problem as a submodular maximization problem, we can use the following algorithm
    pass



class BatchBALDBMFALStrategy(BiFidelityBatchALStrategy):
    """
    Batch, multi-fidelity active learning with Batch-BALD acquisition function.

    This strategy interprets Batch-BALD in a Gaussian Process context by maximizing the joint entropy
    of the posterior distribution of the Bernoulli observations for the candidate batch. 

    The acquisition function is defined as:
    I(Y_{1:n}) = H(Y_{1:n}|X_{1:n}) - E_w[H(Y_{1:n}|X_{1:n},w)]
    where H(Y_{1:n}|X_{1:n}) is the predictive entropy of the Bernoulli observations, and E_w[H(Y_{1:n}|X_{1:n},w)] is the expected conditional entropy of the Bernoulli observations given the latent function values.

    The expected conditional entropy is computed by Monte Carlo integration over the latent function values.

    The acquisition function is then used to select the next batch of points to evaluate.
    """

    def __init__(self, model: BiFidelityModel, 
                 dataset: BiFidelityDataset, 
                 num_mc_samples:int=20,
                 seed:int=42, 
                 max_pool_subset:int=50):
        super().__init__(model, dataset)
        self.gen = np.random.default_rng(seed=seed)
        self.max_pool_subset = max_pool_subset
        self.num_mc_samples = num_mc_samples

    def __str__(self):
        return 'BatchBALDBMFALStrategy'

    def _check_fidelity_feasibility(self, cost_so_far, budget):
        """Checks if adding a point of a given fidelity would exceed the budget."""
        flags = np.zeros((2,), dtype=bool)
        flags[0] = cost_so_far + self.dataset.c_LF <= budget
        flags[1] = cost_so_far + self.dataset.c_HF <= budget
        return flags


    def select_batch(self, 
                     config: ALExperimentConfig, 
                     current_model_trained: BiFidelityModel, 
                     budget_this_step: float) -> tuple[np.ndarray, np.ndarray]:
        # Generate new candidate pools for each selection step

        X_lf_cand_pool = self._generate_lhs_samples(config, config.N_cand_LF)
        X_hf_cand_pool = self._generate_lhs_samples(config, config.N_cand_HF)

        selected_indices_LF = []
        selected_indices_HF = []
        selected_points_LF = []
        selected_points_HF = []

        
        #########################################################
        #########################################################
        # See equation (8) in https://proceedings.neurips.cc/paper_files/paper/2019/file/95323660ed2124450caaac2c46b5ed90-Paper.pdf
        # I(Y_{1:n}) = H(Y_{1:n}|X_{1:n}) - E_w[H(Y_{1:n}|X_{1:n},w)]
        # E_w[H(Y_{1:n}|X_{1:n},w)] in equation (9), w is the bernoulli parameters over the field in our problem
        # H(Y_{1:n}|X_{1:n}) in equation (10-12)
        
        # 1. we need p(y_i|x_i,w) for each x_i in the current batch: y_i|x_i \sim Bernoulli(w), w\sim \Phi(f(x_i))
        # TODO: can add a parameter to specifiy how many MC samples to draw
        theta_lf = current_model_trained.predict_lf(torch.from_numpy(X_lf_cand_pool).float(), 
                                                    num_samples=20)
        theta_hf = current_model_trained.forward(torch.from_numpy(X_hf_cand_pool).float(), 
                                                     num_samples=20, 
                                                     return_lf=False)["hf_samples"]


        theta_lf = theta_lf.probs.T
        theta_hf = theta_hf.probs.T

        one_minus_theta_lf = 1 - theta_lf
        one_minus_theta_hf = 1 - theta_hf

        logp_n_m_c_lf = torch.stack([theta_lf, one_minus_theta_lf], dim=-1) # (N, M, 2)
        logp_n_m_c_hf = torch.stack([theta_hf, one_minus_theta_hf], dim=-1) # (N, M, 2)

        # 2 Compute H(Y|x) and E_w[H(Y|x,w)]
        H_Y_x_N_lf = compute_predicitive_entropy(logp_n_m_c_lf) # H(Y|x) for all pool points
        H_Y_x_N_hf = compute_predicitive_entropy(logp_n_m_c_hf) # H(Y|x) for all pool points

        E_theta_H_N_lf = compute_expected_conditional_entropy(logp_n_m_c_lf) # E_|theta|H(Y|x, theta) for all pool points
        E_theta_H_N_hf = compute_expected_conditional_entropy(logp_n_m_c_hf) # E_|theta|H(Y|x, theta) for all pool points

        # 3 Compute MI(Y,w|x)
        MI_lf = H_Y_x_N_lf - E_theta_H_N_lf
        MI_hf = H_Y_x_N_hf - E_theta_H_N_hf 

        # weighted by cost
        # consider also weighetd by the distance 
        weighted_MI_lf = MI_lf / self.dataset.c_LF
        weighted_MI_hf = MI_hf / self.dataset.c_HF

        # greedy selection over weighted MI
        cost_so_far = 0
        while True:
            # check budget feasibility
            flags = self._check_fidelity_feasibility(cost_so_far, budget_this_step)
            if not flags.any():
                logger.info("Budget exhausted or no feasible fidelities left. Finalizing batch.")
                break

            # TODO: or Random decide the order of selection?
            # evaluate HF candidates
            if flags[1] and len(X_hf_cand_pool) > 0:
                if len(selected_indices_HF) > 0:
                    current_pool_hf = np.delete(X_hf_cand_pool, selected_indices_HF, axis=0)
                    current_pool_hf_map = np.delete(np.arange(len(X_hf_cand_pool)), selected_indices_HF)
                    current_weighted_MI_hf = weighted_MI_hf[current_pool_hf_map]


                    relative_distance = cdist(current_pool_hf, np.vstack(selected_points_HF), metric='euclidean') # (N_cand_HF - len(selected_indices_HF), len(selected_points_HF))
                    
                    bandwidth = np.median(relative_distance)
                    distance_weight = np.exp(-np.square(relative_distance).sum(axis=1)/(2*bandwidth**2))
                    current_weighted_MI_hf = current_weighted_MI_hf * distance_weight
                    
                    # select the best point
                    best_ind = np.argmax(current_weighted_MI_hf)
                    selected_indices_HF.append(current_pool_hf_map[best_ind])
                    selected_points_HF.append(X_hf_cand_pool[current_pool_hf_map[best_ind]][np.newaxis, :])
                    cost_so_far += self.dataset.c_HF
                else:
                    # select the best point
                    best_ind = np.argmax(weighted_MI_hf)
                    selected_indices_HF.append(best_ind)
                    selected_points_HF.append(X_hf_cand_pool[best_ind][np.newaxis, :])
                    cost_so_far += self.dataset.c_HF
            
            # evaluate LF candidates
            if flags[0] and len(X_lf_cand_pool) > 0:
                if len(selected_indices_LF) > 0:
                    current_pool_lf = np.delete(X_lf_cand_pool, selected_indices_LF, axis=0)
                    current_pool_lf_map = np.delete(np.arange(len(X_lf_cand_pool)), selected_indices_LF)
                    current_weighted_MI_lf = weighted_MI_lf[current_pool_lf_map]

                    relative_distance = cdist(current_pool_lf, np.vstack(selected_points_LF), metric='euclidean') # (N_cand_LF - len(selected_indices_LF), len(selected_points_LF))
                    bandwidth = np.median(relative_distance)
                    distance_weight = np.exp(-np.square(relative_distance).sum(axis=1)/(2*bandwidth**2))
                    current_weighted_MI_lf = current_weighted_MI_lf * distance_weight

                    # select the best point
                    best_ind = np.argmax(current_weighted_MI_lf)
                    selected_indices_LF.append(current_pool_lf_map[best_ind])
                    selected_points_LF.append(X_lf_cand_pool[current_pool_lf_map[best_ind]][np.newaxis, :])
                    cost_so_far += self.dataset.c_LF
                else:
                    # select the best point
                    best_ind = np.argmax(weighted_MI_lf)
                    selected_indices_LF.append(best_ind)
                    selected_points_LF.append(X_lf_cand_pool[best_ind][np.newaxis, :])
                    cost_so_far += self.dataset.c_LF
            
            # check if the budget is exhausted
            if cost_so_far >= budget_this_step:
                break
        
        return np.vstack(selected_points_LF), np.vstack(selected_points_HF)


    def _generate_lhs_samples(self, config: ALExperimentConfig, n_samples: int) -> np.ndarray:
        """Generates samples using Latin Hypercube Sampling within the specified domain."""
        if n_samples <= 0:
            return np.array([[]])

        input_dim = len(config.domain_bounds)
        samples_unit_hypercube = pyDOE.lhs(input_dim, samples=n_samples, criterion='maximin')

        scaled_samples = np.zeros_like(samples_unit_hypercube)
        for i, (min_val, max_val) in enumerate(config.domain_bounds):
            scaled_samples[:, i] = samples_unit_hypercube[:, i] * (max_val - min_val) + min_val

        return scaled_samples
