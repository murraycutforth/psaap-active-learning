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
from src.batch_al_strategies.mutual_information_strategy_bernoulli_p import MutualInformationBernoulliPStrategy
from src.batch_al_strategies.mutual_information_strategy_bmfal import MutualInformationBMFALStrategy

logger = logging.getLogger(__name__)


class MutualInformationBernoulliPRepeatsStrategy(MutualInformationBernoulliPStrategy):
    """
    Batch, multi-fidelity active learning with MI-based acquisition function.

    - In addition, when a point is chosen then it is incorrect to just sample a single Bernoulli outcome
    - In this extension, after choosing a location, the prior estimate of p at this location is used to
    set the number of samples. As a rough, first order, approximation, we can use:
    SEM = p(1-p)/sqrt(N), or N = p^2(1-p)^2 / SEM^2. We then say that N should be proportional to p^2(1-p)^2,
    without setting a specific SEM. And then, we rescale so that N(p) lies in [0, Nmax].
    - I.e. N(p) = A * p^2(1-p)^2, and solve N(0.5) = Nmax for A.
    - This neglects the epistemic uncertainty, which could also be accounted for in a future extension.
    """
    def __init__(self, dataset: BiFidelityDataset, N_test_points=100, max_pool_subset=50, plot_all_scores: bool = False,
                 repeat_jitter=0.002, Nmax=5):
        self.repeat_jitter = repeat_jitter
        self.Nmax = Nmax
        self.A = float(Nmax) / (0.25**2)
        super().__init__(dataset, N_test_points=N_test_points, max_pool_subset=max_pool_subset, plot_all_scores=plot_all_scores)

    def __str__(self):
        return f'MutualInformationBernoulliPRepeatsStrategy(N={self.Nmax},jitter={self.repeat_jitter:.3g})'

    def _compute_num_repeats(self, p_cand):
        assert 0.0 <= p_cand <= 1.0
        N_float = self.A * p_cand**2 * (1 - p_cand)**2
        N_float = np.clip(N_float, 1.0, self.Nmax)
        return int(np.round(N_float))

    def select_batch(self,
                     config: ALExperimentConfig,
                     current_model_trained: BiFidelityModel,
                     budget_this_step: float
                     ) -> tuple[np.ndarray, np.ndarray]:  # LF points, HF points
        """Greedy algorithm to select batch of runs under MI acquisition function
        """
        # New candidate pool of LHS each round
        X_LF_cand_pool = self._generate_lhs_samples(config, config.N_cand_LF)
        X_HF_cand_pool = self._generate_lhs_samples(config, config.N_cand_HF)

        N_reps_LF = []
        N_reps_HF = []

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

            assert fidelity in {0, 1}

            if fidelity:  # 1 for HF, 0 for LF
                x = X_HF_cand_pool[ind]
                p_cand = current_model_trained.predict_hf_prob(x[None, :])[0]
                N_reps = self._compute_num_repeats(p_cand)

                assert ind < len(X_HF_cand_pool)
                cost_so_far += self.dataset.c_HF * N_reps
                inds_HF.extend([ind] * N_reps)
                N_reps_HF.append(N_reps)
            else:
                assert ind < len(X_LF_cand_pool)
                x = X_LF_cand_pool[ind]
                p_cand = current_model_trained.predict_lf_prob(x[None, :])[0]
                N_reps = self._compute_num_repeats(p_cand)

                cost_so_far += self.dataset.c_LF * N_reps
                inds_LF.extend([ind] * N_reps)
                N_reps_LF.append(N_reps)

            logger.debug(f"Step {i} complete. Cost so far: {cost_so_far:.4f}. Len(inds_LF): {len(inds_LF)}, Len(inds_HF): {len(inds_HF)}")
            i += 1

        X_LF_new = X_LF_cand_pool[inds_LF]
        X_HF_new = X_HF_cand_pool[inds_HF]

        jitter_LF = np.random.randn(*X_LF_new.shape) * self.repeat_jitter
        jitter_HF = np.random.randn(*X_HF_new.shape) * self.repeat_jitter

        X_LF_new += jitter_LF
        X_HF_new += jitter_HF

        return X_LF_new, X_HF_new
