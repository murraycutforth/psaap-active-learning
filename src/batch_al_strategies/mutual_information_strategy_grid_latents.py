import dataclasses
import logging

import numpy as np
import torch

from src.active_learning.util_classes import BiFidelityModel, BiFidelityDataset, ALExperimentConfig
from src.batch_al_strategies.mutual_information_strategy_bmfal import MutualInformationBMFALStrategy

logger = logging.getLogger(__name__)


class MutualInformationGridStrategy(MutualInformationBMFALStrategy):
    """
    This class modifies the mutual information approach as follows.

    - A regular grid of points is placed over parameter space, call this point set X_G
    - We aim to find the subset of points, X_C such that I[Q_C={f_j(x_j)}_j; f_H(X_G \\ X_C)] is maximised
    - As before, a greedy cost-weighted algorithm is used to solve this in the multifidelity case

    See: "Sequential Design with Mutual Information for Computer Experiments (MICE): Emulation of a Tsunami Model",
    Beck and Guillas, 2015
    """
    def __init__(self, model: BiFidelityModel, dataset: BiFidelityDataset, seed=42, N_MC=100):
        super().__init__(model, dataset)
        self.gen = np.random.default_rng(seed=seed)
        self.N_MC = N_MC

    def select_batch(self,
                     config: ALExperimentConfig,
                     current_model_trained: BiFidelityModel,  # Pass the currently trained model
                     budget_this_step: float
                     ) -> tuple[np.ndarray, np.ndarray]:  # LF indices from X_LF_cand_pool, HF indices from X_HF_cand_pool
        """Greedy algorithm to select batch of runs under MI acquisition function
        """
        X_G = self._generate_grid_samples(config)

        inds_LF = []
        inds_HF = []
        cost_so_far = 0
        i = 1

        while True:
            flags = self._check_fidelity_feasibility(cost_so_far, budget_this_step)

            if not flags.any():
                break

            fidelity, ind = self._max_greedy_acquisition(X_G, inds_LF, inds_HF, current_model_trained, flags)

            assert fidelity in {0, 1}

            if fidelity:  # 1 for HF, 0 for LF
                assert ind < len(X_G)
                cost_so_far += self.dataset.c_HF
                inds_HF.append(ind)
            else:
                assert ind < len(X_G)
                cost_so_far += self.dataset.c_LF
                inds_LF.append(ind)

            assert len(set(inds_LF)) == len(inds_LF)
            assert len(set(inds_HF)) == len(inds_HF)

            logger.info(f"Step {i} complete. Cost so far: {cost_so_far:.4f}. Len(inds_LF): {len(inds_LF)}, Len(inds_HF): {len(inds_HF)}")
            i += 1

        X_LF_new = X_G[inds_LF].copy()
        X_HF_new = X_G[inds_HF].copy()

        return X_LF_new, X_HF_new

    def _generate_grid_samples(self, config: ALExperimentConfig):
        xs = np.linspace(0, 1, int(np.sqrt(config.N_cand_LF)))
        ys = np.linspace(0, 1, int(np.sqrt(config.N_cand_LF)))
        xx, yy = np.meshgrid(xs, ys)
        grid_points_np = np.vstack([xx.ravel(), yy.ravel()]).T
        assert grid_points_np.shape[1] == 2
        return grid_points_np


    def __str__(self):
        return 'MutualInformationStrategy'

    def _check_fidelity_feasibility(self, cost_so_far, budget):
        c_LF = self.dataset.c_LF
        c_HF = self.dataset.c_HF

        flags = np.zeros((2,), dtype=bool)
        flags[0] = cost_so_far + c_LF < budget
        flags[1] = cost_so_far + c_HF < budget
        return flags

    def _max_greedy_acquisition(self, X_G, inds_LF, inds_HF, model, flags):
        """Compute acqusition function for each fidelity and each candidate position and return the max
        """
        assert flags.any()

        non_cand_inds = set(range(len(X_G))).difference(set(inds_LF).union(set(inds_HF)))
        cand_inds = set(range(len(X_G))).difference(non_cand_inds)

        @dataclasses.dataclass
        class Optimum:
            fidelity: int
            mi: float
            cand_ind: int

        optimum = Optimum(-1, -np.inf, -1)  # (Fidelity, MI)

        current_proposals = []  # (Fidelity, x) the points which have been picked
        X_LF_current = []  # The points in X_G which have not been picked yet
        X_LF_current_ind_map = []
        for i in range(len(X_G)):
            if i in inds_LF:
                current_proposals.append((0, X_G[i]))
            else:
                X_LF_current.append(X_G[i])
                X_LF_current_ind_map.append(i)

        X_HF_current = []
        X_HF_current_ind_map = []
        for i in range(len(X_G)):
            if i in inds_HF:
                current_proposals.append((1, X_G[i] + np.array([1e-6, 1e-6])))  # Stops overlap with X_prime which can cause lin alg errors down the line
            else:
                X_HF_current.append((X_G[i] + np.array([1e-6, 1e-6])))
                X_HF_current_ind_map.append(i)

        X_prime = torch.from_numpy(X_G[list(non_cand_inds)]).float()
        base_mi = self._estimate_MI(current_proposals, model, X_prime)

        logger.info(f"Number of current proposals: {len(current_proposals)}")
        logger.info(f"Number of X_LF_current: {len(X_LF_current)}")
        logger.info(f"Number of X_HF_current: {len(X_HF_current)}")
        logger.info(f"Base MI: {base_mi}")

        if flags[0]:
            # Check all LF candidate points
            # For efficiency, just check a random subset of 50 points if there are more proposals than that
            if len(X_LF_current) > 50:
                inds = np.random.choice(range(len(X_LF_current)), 50, replace=False)
            else:
                inds = range(len(X_LF_current))

            for i in inds:
                x = X_LF_current[i]
                X_prime = torch.from_numpy(X_G[list(non_cand_inds.difference({i}))]).float()
                mi = self._estimate_MI(current_proposals + [(0, x)], model, X_prime)
                cost_weighted_delta_mi = (mi - base_mi) / self.dataset.c_LF

                if cost_weighted_delta_mi > optimum.mi:
                    optimum.mi = cost_weighted_delta_mi
                    optimum.fidelity = 0
                    optimum.cand_ind = X_LF_current_ind_map[i]

                logger.debug(f"LF loop. X_prime.shape={X_prime.shape}, len(current_proposals)={len(current_proposals)}")
                logger.debug(f"LF: {x}, MI={mi:.4f}, WDMI={cost_weighted_delta_mi:.4f}")

        if flags[1]:
            if len(X_HF_current) > 50:
                inds = np.random.choice(range(len(X_HF_current)), 50, replace=False)
            else:
                inds = range(len(X_HF_current))
            for i in inds:
                x = X_HF_current[i]
                X_prime = torch.from_numpy(X_G[list(non_cand_inds.difference({i}))]).float()
                mi = self._estimate_MI(current_proposals + [(1, x)], model, X_prime)
                cost_weighted_delta_mi = (mi - base_mi) / self.dataset.c_HF

                if cost_weighted_delta_mi > optimum.mi:
                    optimum.mi = cost_weighted_delta_mi
                    optimum.fidelity = 1
                    optimum.cand_ind = X_HF_current_ind_map[i]

                logger.debug(f"HF loop. X_prime.shape={X_prime.shape}, len(current_proposals)={len(current_proposals)}")
                logger.debug(f"HF: {x}, MI={mi:.4f}, WDMI={cost_weighted_delta_mi:.4f}")

        logger.info(f"Greedy solve completed, optimum: ({optimum.fidelity},{optimum.cand_ind}) MI={optimum.mi:.4f}")

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




