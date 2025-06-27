import dataclasses
import logging

import numpy as np
import torch
from matplotlib import pyplot as plt

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
    def __init__(self, model: BiFidelityModel, dataset: BiFidelityDataset, seed=42, max_pool_subset=50, plot_all_scores=False):
        super().__init__(model, dataset, seed=seed, plot_all_scores=plot_all_scores, max_pool_subset=max_pool_subset)

    def __str__(self):
        return "MutualInformationGridStrategy"

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
        plot_scores = self.plot_all_scores

        while True:
            flags = self._check_fidelity_feasibility(cost_so_far, budget_this_step)

            if not flags.any():
                break

            fidelity, ind = self._max_greedy_acquisition(X_G, inds_LF, inds_HF, current_model_trained, flags, plot=plot_scores)
            plot_scores = False

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

    def _check_fidelity_feasibility(self, cost_so_far, budget):
        c_LF = self.dataset.c_LF
        c_HF = self.dataset.c_HF

        flags = np.zeros((2,), dtype=bool)
        flags[0] = cost_so_far + c_LF < budget
        flags[1] = cost_so_far + c_HF < budget
        return flags

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

    def assemble_current_proposals(self, X_G, inds_HF, inds_LF):
        current_proposals = []  # (Fidelity, x) the points which have been picked from X_G
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
                current_proposals.append((1, X_G[i]))
            else:
                X_HF_current.append((X_G[i]))
                X_HF_current_ind_map.append(i)
        return X_HF_current, X_HF_current_ind_map, X_LF_current, X_LF_current_ind_map, current_proposals





