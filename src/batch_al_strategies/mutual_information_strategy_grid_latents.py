import dataclasses
import logging

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

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
                     current_model_trained: BiFidelityModel,
                     budget_this_step: float
                     ) -> tuple[np.ndarray, np.ndarray]:
        """Greedy algorithm to select batch of runs under MI acquisition function"""
        X_G = self._generate_grid_samples(config)

        # --- IMPROVEMENT 1: Efficient candidate management ---
        # Use a set for O(1) lookups and removals.
        available_indices = set(range(len(X_G)))
        current_proposals = []  # List of (fidelity, x) tuples

        inds_LF_global = []
        inds_HF_global = []
        cost_so_far = 0
        i = 1
        plot_scores = self.plot_all_scores

        while True:
            flags = self._check_fidelity_feasibility(cost_so_far, budget_this_step)

            if not flags.any():
                break

            fidelity, selected_ind = self._max_greedy_acquisition(
                X_G,
                available_indices,
                current_proposals,
                current_model_trained,
                flags,
                plot=plot_scores
            )
            #plot_scores = False

            assert fidelity in {0, 1}
            assert selected_ind in available_indices

            # Update state based on selection
            available_indices.remove(selected_ind)
            x_selected = X_G[selected_ind]
            current_proposals.append((fidelity, x_selected))

            if fidelity == 1:  # HF
                cost_so_far += self.dataset.c_HF
                inds_HF_global.append(selected_ind)
            else:  # LF
                cost_so_far += self.dataset.c_LF
                inds_LF_global.append(selected_ind)

            logger.info(
                f"Step {i} complete. Cost so far: {cost_so_far:.4f}. Len(inds_LF): {len(inds_LF_global)}, Len(inds_HF): {len(inds_HF_global)}")
            i += 1

        X_LF_new = X_G[inds_LF_global].copy() if inds_LF_global else np.array([]).reshape(0, X_G.shape[1])
        X_HF_new = X_G[inds_HF_global].copy() if inds_HF_global else np.array([]).reshape(0, X_G.shape[1])

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

    def _max_greedy_acquisition(self, X_G_np, available_indices, current_proposals, model, flags, plot=False):
        """Compute acquisition function for each fidelity and each candidate position and return the max"""
        assert flags.any()

        plot_data = {
            'X_LF': [],
            'X_HF': [],
            'MI_LF': [],
            'MI_HF': []
        }

        # --- IMPROVEMENT 2 & 4: Optimize inner loop and use Tensors early ---
        # Convert to tensor once
        X_G = torch.from_numpy(X_G_np).float()

        # The set of points for which we evaluate the posterior entropy is X_G \ X_C,
        # which is precisely the set of available (unevaluated) points.
        eval_inds_list = list(available_indices)
        X_pred_base = X_G[eval_inds_list]

        # Create a map from the global index in X_G to the local index in X_pred_base
        # This allows us to quickly find which row to remove.
        global_to_local_pred_idx_map = {global_idx: local_idx for local_idx, global_idx in enumerate(eval_inds_list)}

        # Calculate base MI with all available points as the prediction set
        base_mi = self._estimate_MI(current_proposals, model, X_pred_base)

        logger.debug(f"Number of current proposals: {len(current_proposals)}")
        logger.debug(f"Number of candidate points: {len(available_indices)}")
        logger.debug(f"Base MI: {base_mi}")

        @dataclasses.dataclass
        class Optimum:
            fidelity: int = -1
            mi: float = -np.inf
            cand_ind: int = -1

        optimum = Optimum()

        # --- Determine the pool of candidates to check ---
        # `available_indices` is now our single source of truth for candidates.
        if len(available_indices) > self.max_pool_subset:
            candidate_indices_to_check = np.random.choice(list(available_indices), self.max_pool_subset, replace=False)
        else:
            candidate_indices_to_check = list(available_indices)

        for cand_ind in tqdm(candidate_indices_to_check, desc='Checking candidate indices'):
            x_cand = X_G[cand_ind]

            # Find the local index of the candidate to remove it from X_pred_base
            local_idx_to_remove = global_to_local_pred_idx_map[cand_ind]

            # Create new prediction set by removing one row. This is much faster.
            # Note: This still creates a new tensor. For ultimate speed, _estimate_MI
            # could be modified to accept indices to ignore.
            rows_to_keep = torch.arange(X_pred_base.shape[0]) != local_idx_to_remove
            X_prime_new = X_pred_base[rows_to_keep]

            # Check LF
            if flags[0]:
                mi_lf = self._estimate_MI(current_proposals + [(0, x_cand.numpy())], model, X_prime_new)
                cost_weighted_delta_mi = (mi_lf - base_mi) / self.dataset.c_LF
                if cost_weighted_delta_mi > optimum.mi:
                    optimum.mi = cost_weighted_delta_mi
                    optimum.fidelity = 0
                    optimum.cand_ind = cand_ind

                plot_data['X_LF'].append(x_cand)
                plot_data['MI_LF'].append(cost_weighted_delta_mi)

            # Check HF
            if flags[1]:
                mi_hf = self._estimate_MI(current_proposals + [(1, x_cand.numpy())], model, X_prime_new)
                cost_weighted_delta_mi = (mi_hf - base_mi) / self.dataset.c_HF
                if cost_weighted_delta_mi > optimum.mi:
                    optimum.mi = cost_weighted_delta_mi
                    optimum.fidelity = 1
                    optimum.cand_ind = cand_ind

                plot_data['X_HF'].append(x_cand)
                plot_data['MI_HF'].append(cost_weighted_delta_mi)

        logger.debug(f"Greedy solve completed, optimum: ({optimum.fidelity},{optimum.cand_ind}) MI={optimum.mi:.4f}")

        # Plotting logic would need slight adaptation if used, but is omitted for brevity.
        if plot:
            self._plot_all_scores(plot_data)
        assert optimum.cand_ind >= 0

        return optimum.fidelity, optimum.cand_ind
