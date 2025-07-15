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
    def __init__(self, dataset: BiFidelityDataset, N_test_points=100, max_pool_subset=50, plot_all_scores: bool = False):
        super().__init__(dataset)
        self.gen = np.random.default_rng()
        self.N_test_points = N_test_points
        self.max_pool_subset = max_pool_subset
        self.plot_all_scores = plot_all_scores
        self.plot_ind = 0
        self.fig_outdir = Path(__file__).parent / 'figures' / str(self)
        self.fig_outdir.mkdir(parents=True, exist_ok=True)

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
        plot_scores = self.plot_all_scores

        pbar = tqdm(total=budget_this_step, desc="Selecting batch")

        while True:
            flags = self._check_fidelity_feasibility(cost_so_far, budget_this_step)
            pbar.n = cost_so_far
            pbar.refresh()

            if not flags.any():
                break

            fidelity, ind = self._max_greedy_acquisition(X_LF_cand_pool, X_HF_cand_pool, inds_LF, inds_HF, current_model_trained, flags, plot=plot_scores)
            #plot_scores = False  # Just plot the first set of scores on each AL round

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

            logger.debug(f"Step {i} complete. Cost so far: {cost_so_far:.4f}. Len(inds_LF): {len(inds_LF)}, Len(inds_HF): {len(inds_HF)}")
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

    def _max_greedy_acquisition(self, X_LF, X_HF, inds_LF, inds_HF, model, flags, plot: bool = False):
        """Compute acqusition function for each fidelity and each candidate position and return the max
        """
        assert flags.any()

        X_prime = torch.from_numpy(self.gen.random(size=(self.N_test_points, 2))).float()

        X_HF_candidates, X_HF_cand_ind_map, X_LF_candidates, X_LF_cand_ind_map, current_proposals = self.assemble_current_proposals(
            X_HF, X_LF, inds_HF, inds_LF)

        base_mi = self._estimate_MI(current_proposals, model, X_prime)

        logger.debug(f"Number of current proposals: {len(current_proposals)}")
        logger.debug(f"Number of X_LF_candidates: {len(X_LF_candidates)}")
        logger.debug(f"Number of X_HF_candidates: {len(X_HF_candidates)}")
        logger.debug(f"Base MI: {base_mi}")

        @dataclasses.dataclass
        class CandidateResult:
            mi: float
            fidelity: int
            cand_ind: int

        cand_results = []

        if flags[0]:
            # Check all LF candidate points
            # For efficiency, just check a random subset of 50 points if there are more proposals than that
            if len(X_LF_candidates) > self.max_pool_subset:
                inds = np.random.choice(range(len(X_LF_candidates)), self.max_pool_subset, replace=False)
            else:
                inds = range(len(X_LF_candidates))

            for i in inds:
                x = X_LF_candidates[i]
                mi = self._estimate_MI(current_proposals + [(0, x)], model, X_prime)

                cand_results.append(CandidateResult(
                    mi=mi,
                    fidelity=0,
                    cand_ind=X_LF_cand_ind_map[i],
                ))

        if flags[1]:
            if len(X_HF_candidates) > self.max_pool_subset:
                inds = np.random.choice(range(len(X_HF_candidates)), self.max_pool_subset, replace=False)
            else:
                inds = range(len(X_HF_candidates))
            for i in inds:
                x = X_HF_candidates[i]
                mi = self._estimate_MI(current_proposals + [(1, x)], model, X_prime)

                cand_results.append(CandidateResult(
                    mi=mi,
                    fidelity=1,
                    cand_ind=X_HF_cand_ind_map[i],
                ))

        all_cost_weighted_delta_mi = []
        for cand_result in cand_results:
            if cand_result.fidelity == 0:
                all_cost_weighted_delta_mi.append((cand_result.mi - base_mi) / self.dataset.c_LF)
            elif cand_result.fidelity == 1:
                all_cost_weighted_delta_mi.append((cand_result.mi - base_mi) / self.dataset.c_HF)

        # Now select index with highest cost weighted delta mi
        i = np.argmax(all_cost_weighted_delta_mi)
        optimum_cand_ind = cand_results[i].cand_ind
        optimum_fidelity = cand_results[i].fidelity
        optimum_mi = all_cost_weighted_delta_mi[i]

        logger.debug(f"Greedy solve completed, optimum: {optimum_fidelity}, MI={optimum_mi:.4f}")

        if plot:
            self._plot_all_scores(
                cand_results,
                np.array(X_LF),
                np.array(X_HF),
                base_mi
            )

        assert optimum_cand_ind >= 0

        return optimum_fidelity, optimum_cand_ind

    def assemble_current_proposals(self, X_HF, X_LF, inds_HF, inds_LF):
        """Assemble current proposals into X_HF and X_LF
        """
        current_proposals = []  # (Fidelity, x) for all data points selected for current batch
        X_LF_candidates = []  # Elements from X_LF which have not been selected so far
        X_LF_cand_ind_map = []  # The j-th element in this list tells us the index of the j-th candidate in X_HF
        for i in range(len(X_LF)):
            if i in list(set(inds_LF)):
                current_proposals.append((0, X_LF[i]))
            else:
                X_LF_candidates.append(X_LF[i])
                X_LF_cand_ind_map.append(i)

        X_HF_candidates = []
        X_HF_cand_ind_map = []
        for i in range(len(X_HF)):
            if i in list(set(inds_HF)):
                current_proposals.append((1, X_HF[i]))
            else:
                X_HF_candidates.append(X_HF[i])
                X_HF_cand_ind_map.append(i)

        return X_HF_candidates, X_HF_cand_ind_map, X_LF_candidates, X_LF_cand_ind_map, current_proposals

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

        X_H_and_prime = torch.cat((X_H, X_prime), dim=0)
        uniqueness_test = torch.unique(X_H_and_prime, dim=0)
        assert uniqueness_test.shape[0] == X_H_and_prime.shape[0], f"{X_H_and_prime.shape[0]} != {uniqueness_test.shape[0]}, X_H shape = {X_H.shape}, X_prime shape = {X_prime.shape}, proposal set: {proposal_set}, X_prime = {X_prime}"

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

    def _plot_all_scores(self, cand_results, X_LF_cand_pool, X_HF_cand_pool, base_mi):
        # We'll color by cost-weighted delta MI
        lf_coords = []
        lf_scores = []
        hf_coords = []
        hf_scores = []

        for cr in cand_results:
            if cr.fidelity == 0:
                coord = X_LF_cand_pool[cr.cand_ind]
                score = (cr.mi - base_mi) / self.dataset.c_LF
                lf_coords.append(coord)
                lf_scores.append(score)
            elif cr.fidelity == 1:
                coord = X_HF_cand_pool[cr.cand_ind]
                score = (cr.mi - base_mi) / self.dataset.c_HF
                hf_coords.append(coord)
                hf_scores.append(score)

        lf_coords = np.array(lf_coords) if lf_coords else np.zeros((0, 2))
        hf_coords = np.array(hf_coords) if hf_coords else np.zeros((0, 2))

        fig, axs = plt.subplots(1, 2, figsize=(8, 3), dpi=200)
        ax_lf, ax_hf = axs

        if len(lf_coords):
            im_lf = ax_lf.scatter(lf_coords[:, 0], lf_coords[:, 1], c=lf_scores, s=20)
            fig.colorbar(im_lf, ax=ax_lf, label="Acquisition fn")
        if len(hf_coords):
            im_hf = ax_hf.scatter(hf_coords[:, 0], hf_coords[:, 1], c=hf_scores, s=20)
            fig.colorbar(im_hf, ax=ax_hf, label="Acquisition fn")

        ax_lf.set_title("LF")
        ax_hf.set_title("HF")
        ax_lf.set_aspect('equal')
        ax_hf.set_aspect('equal')
        fig.tight_layout()
        fig.savefig(self.fig_outdir / f"{self.plot_ind}.png")
        plt.close(fig)
        self.plot_ind += 1



