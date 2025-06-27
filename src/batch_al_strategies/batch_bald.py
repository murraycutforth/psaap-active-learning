import dataclasses
import logging

import numpy as np
import pyDOE
import torch

from src.batch_al_strategies.base import BiFidelityBatchALStrategy
from src.active_learning.util_classes import BiFidelityModel, BiFidelityDataset, ALExperimentConfig

logger = logging.getLogger(__name__)


class BatchBALDBMFALStrategy(BiFidelityBatchALStrategy):
    """
    Batch, multi-fidelity active learning with Batch-BALD acquisition function.

    This strategy interprets Batch-BALD in a Gaussian Process context by maximizing the joint entropy
    of the posterior distribution of the latent function values for the candidate batch. This is
    equivalent to maximizing the log-determinant of the posterior covariance matrix of the batch,
    a criterion that encourages selecting diverse and uncertain points.

    - Acquisition function for a batch Q is a(Q) = log |K_post(Q, Q)|, where K_post is the posterior
      covariance matrix of the latent function values at the points in Q.
    - A greedy approach is used to construct the batch due to the submodular nature of the objective.
    - Given a set of k inputs and fidelities Q_k, the next (x, m) pair is chosen by maximising the
      cost-normalized gain in the log-determinant:
        a_{k+1} = (log|K_post(Q_k U {(x,m)})| - log|K_post(Q_k)|) / c_m
        s.t. total cost remains below the budget.
    """

    def __init__(self, model: BiFidelityModel, dataset: BiFidelityDataset, seed=42, max_pool_subset=50):
        super().__init__(model, dataset)
        self.gen = np.random.default_rng(seed=seed)
        self.max_pool_subset = max_pool_subset

    def select_batch(self,
                     config: ALExperimentConfig,
                     current_model_trained: BiFidelityModel,
                     budget_this_step: float
                     ) -> tuple[np.ndarray, np.ndarray]:
        """Greedy algorithm to select a batch of points that maximizes the joint entropy (log-determinant).
        """
        # Generate new candidate pools for each selection step
        X_LF_cand_pool = self._generate_lhs_samples(config, config.N_cand_LF)
        X_HF_cand_pool = self._generate_lhs_samples(config, config.N_cand_HF)

        inds_LF = []
        inds_HF = []
        cost_so_far = 0
        i = 1

        while True:
            # Check if adding either an LF or HF point is feasible within the budget
            flags = self._check_fidelity_feasibility(cost_so_far, budget_this_step)

            if not flags.any():
                logger.info("Budget exhausted or no feasible fidelities left. Finalizing batch.")
                break

            # Greedily select the best point (and its fidelity) to add to the batch
            fidelity, ind = self._max_greedy_acquisition(
                X_LF_cand_pool, X_HF_cand_pool, inds_LF, inds_HF, current_model_trained, flags
            )

            if fidelity == -1:
                logger.warning("No optimal point found. Finalizing batch.")
                break

            assert fidelity in {0, 1}

            # Add the selected point to the batch and update the cost
            if fidelity:  # 1 for HF
                cost_so_far += self.dataset.c_HF
                inds_HF.append(ind)
            else:  # 0 for LF
                cost_so_far += self.dataset.c_LF
                inds_LF.append(ind)

            # Sanity checks for no duplicate indices
            assert len(set(inds_LF)) == len(inds_LF)
            assert len(set(inds_HF)) == len(inds_HF)

            logger.info(f"Step {i} complete. Added fidelity {fidelity}. Cost so far: {cost_so_far:.4f}. "
                        f"Len(inds_LF): {len(inds_LF)}, Len(inds_HF): {len(inds_HF)}")
            i += 1

        X_LF_new = X_LF_cand_pool[inds_LF]
        X_HF_new = X_HF_cand_pool[inds_HF]

        return X_LF_new, X_HF_new

    def __str__(self):
        return 'BatchBALDBMFALStrategy'

    def _check_fidelity_feasibility(self, cost_so_far, budget):
        """Checks if adding a point of a given fidelity would exceed the budget."""
        flags = np.zeros((2,), dtype=bool)
        flags[0] = cost_so_far + self.dataset.c_LF <= budget
        flags[1] = cost_so_far + self.dataset.c_HF <= budget
        return flags

    def _max_greedy_acquisition(self, X_LF_cand, X_HF_cand, inds_LF, inds_HF, model, flags):
        """Finds the single candidate point that provides the maximum cost-weighted gain in log-determinant."""
        assert flags.any()

        @dataclasses.dataclass
        class Optimum:
            fidelity: int = -1
            score: float = -np.inf
            cand_ind: int = -1

        optimum = Optimum()

        # Assemble the set of points already selected for the batch in this step
        current_proposals = []
        X_LF_already_selected = X_LF_cand[inds_LF]
        for x in X_LF_already_selected:
            current_proposals.append((0, x))

        X_HF_already_selected = X_HF_cand[inds_HF]
        for x in X_HF_already_selected:
            current_proposals.append((1, x))

        # Calculate the log-determinant of the posterior covariance of the current batch
        base_log_det = self._calculate_log_det(current_proposals, model)
        logger.info(f"Base log-determinant for {len(current_proposals)} points: {base_log_det:.4f}")

        # Identify candidate points not yet selected
        X_LF_pool = np.delete(X_LF_cand, inds_LF, axis=0)
        X_LF_pool_map = np.delete(np.arange(len(X_LF_cand)), inds_LF)

        X_HF_pool = np.delete(X_HF_cand, inds_HF, axis=0)
        X_HF_pool_map = np.delete(np.arange(len(X_HF_cand)), inds_HF)

        # Evaluate LF candidates
        if flags[0] and len(X_LF_pool) > 0:
            if len(X_LF_pool) > self.max_pool_subset:
                subset_inds = self.gen.choice(len(X_LF_pool), self.max_pool_subset, replace=False)
            else:
                subset_inds = range(len(X_LF_pool))

            for i in subset_inds:
                x_cand = X_LF_pool[i]
                new_log_det = self._calculate_log_det(current_proposals + [(0, x_cand)], model)
                gain = new_log_det - base_log_det
                cost_weighted_gain = gain / self.dataset.c_LF

                if cost_weighted_gain > optimum.score:
                    optimum.score = cost_weighted_gain
                    optimum.fidelity = 0
                    optimum.cand_ind = X_LF_pool_map[i]

        # Evaluate HF candidates
        if flags[1] and len(X_HF_pool) > 0:
            if len(X_HF_pool) > self.max_pool_subset:
                subset_inds = self.gen.choice(len(X_HF_pool), self.max_pool_subset, replace=False)
            else:
                subset_inds = range(len(X_HF_pool))

            for i in subset_inds:
                x_cand = X_HF_pool[i]
                new_log_det = self._calculate_log_det(current_proposals + [(1, x_cand)], model)
                gain = new_log_det - base_log_det
                cost_weighted_gain = gain / self.dataset.c_HF

                if cost_weighted_gain > optimum.score:
                    optimum.score = cost_weighted_gain
                    optimum.fidelity = 1
                    optimum.cand_ind = X_HF_pool_map[i]

        logger.info(f"Greedy solve completed, optimum: fidelity={optimum.fidelity}, score={optimum.score:.4f}")
        assert optimum.cand_ind >= 0 or (
                    len(X_LF_pool) == 0 and len(X_HF_pool) == 0), "No optimum found but pool was not empty"

        return optimum.fidelity, optimum.cand_ind

    def _calculate_log_det(self, proposal_set: list[tuple[int, np.ndarray]], model: BiFidelityModel) -> float:
        """Calculates the log-determinant of the posterior covariance matrix for a given set of proposals."""
        if not proposal_set:
            return 0.0  # Log-det of an empty matrix is defined as 0 for this iterative algorithm

        X_L_list, X_H_list = [], []
        input_dim = proposal_set[0][1].shape[0]

        for fidelity, x in proposal_set:
            if fidelity == 0:
                X_L_list.append(x)
            elif fidelity == 1:
                X_H_list.append(x)

        X_L = torch.from_numpy(np.array(X_L_list, dtype=np.float32))
        X_H = torch.from_numpy(np.array(X_H_list, dtype=np.float32))

        # We only need the posterior of the proposed batch, so X_prime is empty.
        # The model's joint prediction needs a tensor, even if it's empty.
        X_prime_empty = torch.empty(0, input_dim, dtype=torch.float32)

        joint_mvn = model.predict_multi_fidelity_latent_joint(X_L, X_H, X_prime_empty)

        cov_matrix = joint_mvn.covariance_matrix
        # Add a small value to the diagonal for numerical stability before logdet
        cov_matrix += torch.eye(cov_matrix.shape[0]) * 1e-6
        log_det = torch.logdet(cov_matrix)

        return log_det.item()

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