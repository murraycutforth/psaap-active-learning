import dataclasses
import logging

import numpy as np
import pyDOE
import torch

from src.batch_al_strategies.base import BiFidelityBatchALStrategy
from src.active_learning.util_classes import BiFidelityModel, BiFidelityDataset, ALExperimentConfig

logger = logging.getLogger(__name__)


class MutualInformationBMFALStrategy(BiFidelityBatchALStrategy):
    """
    Batch, multi-fidelity active learning with MI-based acquisition function.

    - f_H is the predicted latent HF, which is normally distributed in MFGP-ELBO model
    - Use normally-distributed latents so MI computation is analytic
    - Acquisition function for single sample at fidelity M is a(m, x) = E_p(x')[I(f_m(x), f_H(x')]
    - Monte Carlo approximation of expectation is: a(m, x) = 1/N \\sum_i I(f_m(x), f_H(x_i))
    - Greedy approach used to solve the batched acquisition function
    - Given a set of k inputs and fidelities Q_k = {(x_1, m_1), ..., (x_k, m_k)} the next (x, m) pair is chosen by maximising:
        a_{k+1} = 1/N \\sum_i (I(Q_k U {f_m(x)}, f_H(x')) - I(I(Q_k, f_H(x'))) / c_m
        s.t. total cost remains below budget

    See: "Batch Multi-Fidelity Active Learning with Budget Constraints", Li et al., 2022
    """
    def __init__(self, model: BiFidelityModel, dataset: BiFidelityDataset, seed=42, N_MC=100, max_pool_subset=50):
        super().__init__(model, dataset)
        self.gen = np.random.default_rng(seed=seed)
        self.N_MC = N_MC
        self.max_pool_subset = max_pool_subset

    def select_batch(self,
                     config: ALExperimentConfig,
                     current_model_trained: BiFidelityModel,  # Pass the currently trained model
                     budget_this_step: float
                     ) -> tuple[np.ndarray, np.ndarray]:  # LF indices from X_LF_cand_pool, HF indices from X_HF_cand_pool
        """Greedy algorithm to select batch of runs under MI acquisition function
        """
        # New candidate pool of LHS each round
        X_LF_cand_pool = self._generate_lhs_samples(config, config.N_cand_LF)
        X_HF_cand_pool = self._generate_lhs_samples(config, config.N_cand_HF)

        inds_LF = []
        inds_HF = []
        cost_so_far = 0
        i = 1

        while True:
            flags = self._check_fidelity_feasibility(cost_so_far, budget_this_step)

            if not flags.any():
                break

            fidelity, ind = self._max_greedy_acquisition(X_LF_cand_pool, X_HF_cand_pool, inds_LF, inds_HF, current_model_trained, flags)

            assert fidelity in {0, 1}

            if fidelity:  # 1 for HF, 0 for LF
                assert ind < len(X_HF_cand_pool)
                cost_so_far += self.dataset.c_HF
                inds_HF.append(ind)
            else:
                assert ind < len(X_LF_cand_pool)
                cost_so_far += self.dataset.c_LF
                inds_LF.append(ind)

            assert len(set(inds_LF)) == len(inds_LF)
            assert len(set(inds_HF)) == len(inds_HF)

            logger.info(f"Step {i} complete. Cost so far: {cost_so_far:.4f}. Len(inds_LF): {len(inds_LF)}, Len(inds_HF): {len(inds_HF)}")
            i += 1

        X_LF_new = X_LF_cand_pool[inds_LF]
        X_HF_new = X_HF_cand_pool[inds_HF]

        return X_LF_new, X_HF_new

    def __str__(self):
        return 'MutualInformationBMFALStrategy'

    def _check_fidelity_feasibility(self, cost_so_far, budget):
        c_LF = self.dataset.c_LF
        c_HF = self.dataset.c_HF

        flags = np.zeros((2,), dtype=bool)
        flags[0] = cost_so_far + c_LF < budget
        flags[1] = cost_so_far + c_HF < budget
        return flags

    def _max_greedy_acquisition(self, X_LF, X_HF, inds_LF, inds_HF, model, flags):
        """Compute acqusition function for each fidelity and each candidate position and return the max
        """
        assert flags.any()

        # Evaluate the expected MI with HF over the interior [0.1, 0.9] x [0.1, 0.9]
        # MI is prone to over-emphasizing the boundary region
        X_prime = torch.from_numpy(self.gen.random(size=(self.N_MC, 2)) * 0.8 + 0.1).float()

        @dataclasses.dataclass
        class Optimum:
            fidelity: int
            mi: float
            cand_ind: int

        optimum = Optimum(-1, -np.inf, -1)  # (Fidelity, MI)

        # Assemble current included set, and current candidates
        current_proposals = []  # (Fidelity, x)
        X_LF_current = []
        X_LF_current_ind_map = []
        for i in range(len(X_LF)):
            if i in inds_LF:
                current_proposals.append((0, X_LF[i]))
            else:
                X_LF_current.append(X_LF[i])
                X_LF_current_ind_map.append(i)

        X_HF_current = []
        X_HF_current_ind_map = []
        for i in range(len(X_HF)):
            if i in inds_HF:
                current_proposals.append((1, X_HF[i]))
            else:
                X_HF_current.append(X_HF[i])
                X_HF_current_ind_map.append(i)

        base_mi = self._estimate_MI(current_proposals, model, X_prime)

        logger.info(f"Number of current proposals: {len(current_proposals)}")
        logger.info(f"Number of X_LF_current: {len(X_LF_current)}")
        logger.info(f"Number of X_HF_current: {len(X_HF_current)}")
        logger.info(f"Base MI: {base_mi}")

        if flags[0]:
            # Check all LF candidate points
            # For efficiency, just check a random subset of 50 points if there are more proposals than that
            if len(X_LF_current) > self.max_pool_subset:
                inds = np.random.choice(range(len(X_LF_current)), self.max_pool_subset, replace=False)
            else:
                inds = range(len(X_LF_current))
            for i in inds:
                x = X_LF_current[i]
                mi = self._estimate_MI(current_proposals + [(0, x)], model, X_prime)
                cost_weighted_delta_mi = (mi - base_mi) / self.dataset.c_LF

                if cost_weighted_delta_mi > optimum.mi:
                    optimum.mi = cost_weighted_delta_mi
                    optimum.fidelity = 0
                    optimum.cand_ind = X_LF_current_ind_map[i]

                logger.debug(f"LF: {x}, MI={mi:.4f}, WDMI={cost_weighted_delta_mi:.4f}")

        if flags[1]:
            if len(X_HF_current) > self.max_pool_subset:
                inds = np.random.choice(range(len(X_HF_current)), self.max_pool_subset, replace=False)
            else:
                inds = range(len(X_HF_current))
            for i in inds:
                x = X_HF_current[i]
                mi = self._estimate_MI(current_proposals + [(1, x)], model, X_prime)
                cost_weighted_delta_mi = (mi - base_mi) / self.dataset.c_HF

                if cost_weighted_delta_mi > optimum.mi:
                    optimum.mi = cost_weighted_delta_mi
                    optimum.fidelity = 1
                    optimum.cand_ind = X_HF_current_ind_map[i]

                logger.debug(f"HF: {x}, MI={mi:.4f}, WDMI={cost_weighted_delta_mi:.4f}")

        logger.info(f"Greedy solve completed, optimum: {optimum.fidelity}, MI={optimum.mi:.4f}")

        assert optimum.cand_ind >= 0

        return optimum.fidelity, optimum.cand_ind

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

        if N_L + N_H == 0:
            return 0  # MI of empty set with X_prime

        joint_mvn = model.predict_multi_fidelity_latent_joint(X_L, X_H, X_prime)

        sigma_joint = joint_mvn.covariance_matrix
        logdet_sigma_joint = torch.logdet(sigma_joint)

        # Extract marginal covariance matrices for test points and non test points
        N = N_L + N_H
        sigma_x = sigma_joint[:N, :][:, :N]
        sigma_x_prime = sigma_joint[N:, :][:, N:]

        logdet_sigma_x = torch.logdet(sigma_x)
        logdet_sigma_x_prime = torch.logdet(sigma_x_prime)

        mi = 0.5 * (logdet_sigma_x + logdet_sigma_x_prime - logdet_sigma_joint)

        return mi.item()

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





