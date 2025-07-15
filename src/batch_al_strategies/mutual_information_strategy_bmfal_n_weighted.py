import dataclasses
import logging
from math import ceil

import numpy as np
import torch
from scipy.optimize import root_scalar  # or use bisect if you prefer

from src.active_learning.util_classes import BiFidelityModel, BiFidelityDataset, ALExperimentConfig
from src.batch_al_strategies.mutual_information_strategy_bmfal import MutualInformationBMFALStrategy
from src.models.laplace_approximation import laplace_approximation_probit

logger = logging.getLogger(__name__)



class MutualInformationBMFALNweightedStrategy(MutualInformationBMFALStrategy):
    """
    Modification of the BMFAL strategy.
    As before, we use MI to select the best position. However, using the distribution of the latent function
    (this is always Gaussian) we compute the number of trials needed to bring the standard deviation of the posterior down to
    some pre-set threshold, and then weight the MI improvement by the number of trials. In essence, this approach modifies
    the cost of each proposal to also account for the number of trials needed, as well as the fidelity.
    """
    def __init__(self, dataset: BiFidelityDataset, N_test_points=100, max_pool_subset=50, plot_all_scores: bool = False,
                 max_N: int = 10, jitter_scale: float = 0.002, sigma_reduction_prop: float = 0.9):
        self.max_N = max_N
        self.jitter_scale = jitter_scale
        self.sigma_reduction_prop = sigma_reduction_prop
        super().__init__(dataset, N_test_points, max_pool_subset, plot_all_scores)

    def __str__(self):
        return f'MutualInformationBMFALSNWeightedtrategy(maxN={self.max_N},jitter={self.jitter_scale},sigma_reduction_prop={self.sigma_reduction_prop})'

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

        N_reps_LF = []
        N_reps_HF = []

        inds_LF = []
        inds_HF = []
        cost_so_far = 0
        i = 1
        plot_scores = self.plot_all_scores

        while True:
            flags = self._check_fidelity_feasibility(cost_so_far, budget_this_step)

            if not flags.any():
                break

            fidelity, ind, N_reps = self._max_greedy_acquisition(X_LF_cand_pool, X_HF_cand_pool, inds_LF, inds_HF, current_model_trained, flags, plot=plot_scores)
            #plot_scores = False  # Just plot the first set of scores on each AL round

            assert fidelity in {0, 1}

            if fidelity:  # 1 for HF, 0 for LF
                assert ind < len(X_HF_cand_pool)
                cost_so_far += self.dataset.c_HF * N_reps
                inds_HF.extend([ind] * N_reps)
                N_reps_HF.append(N_reps)
            else:
                assert ind < len(X_LF_cand_pool)
                cost_so_far += self.dataset.c_LF * N_reps
                inds_LF.extend([ind] * N_reps)
                N_reps_LF.append(N_reps)

            logger.info(f"Step {i} complete. Cost so far: {cost_so_far:.4f}. Len(inds_LF): {len(inds_LF)}, Len(set(inds(LF)): {len(set(inds_LF))}, Len(inds_HF): {len(inds_HF)}, len(set(inds(HF)): {len(set(inds_HF))})")
            i += 1

        X_LF_new = X_LF_cand_pool[inds_LF]
        X_HF_new = X_HF_cand_pool[inds_HF]

        jitter_LF = np.random.randn(*X_LF_new.shape) * self.jitter_scale
        jitter_HF = np.random.randn(*X_HF_new.shape) * self.jitter_scale

        X_LF_new += jitter_LF
        X_HF_new += jitter_HF

        return X_LF_new, X_HF_new


    def _max_greedy_acquisition(self, X_LF, X_HF, inds_LF, inds_HF, model, flags, plot: bool = False):
        """Compute acqusition function for each fidelity and each candidate position and return the max
        """
        assert flags.any()

        X_prime = torch.from_numpy(self.gen.random(size=(self.N_test_points, 2))).float()

        #f_L_X_prime = model.predict_f_L(X_prime)
        #f_H_X_prime = model.predict_f_H(X_prime)
        #f_L_vars = f_L_X_prime.variance
        #f_H_vars = f_H_X_prime.variance
        #f_L_threshold = 0.5 * torch.sqrt(f_L_vars).mean().item()
        #f_H_threshold = 0.5 * torch.sqrt(f_H_vars).mean().item()
        #logger.info(f'f_L_threshold: {f_L_threshold}')
        #logger.info(f'f_H_threshold: {f_H_threshold}')
        ## TODO: OR- just aim to reduce variance by a pre-set proportion (like 20%)
        #sigma_reduction_prop = 0.95

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
            N: int

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

                f_L_prior = model.predict_f_L(torch.tensor(x[np.newaxis, :]).float())
                mu_prior = f_L_prior.mean.item()
                sigma_prior = torch.sqrt(f_L_prior.variance).item()
                N_est = estimate_N(mu_prior, sigma_prior, self.sigma_reduction_prop * sigma_prior)
                #print(f"For LF proposal with mu_prior={mu_prior}, sigma_prior={sigma_prior}, then N_est={N_est}")

                mi = self._estimate_MI(current_proposals + [(0, x)], model, X_prime)

                cand_results.append(CandidateResult(
                    mi=mi,
                    fidelity=0,
                    cand_ind=X_LF_cand_ind_map[i],
                    N=N_est,
                ))

        if flags[1]:
            if len(X_HF_candidates) > self.max_pool_subset:
                inds = np.random.choice(range(len(X_HF_candidates)), self.max_pool_subset, replace=False)
            else:
                inds = range(len(X_HF_candidates))
            for i in inds:
                x = X_HF_candidates[i]

                f_H_prior = model.predict_f_H(torch.tensor(x[np.newaxis, :]).float())
                mu_prior = f_H_prior.mean.item()
                sigma_prior = torch.sqrt(f_H_prior.variance).item()
                N_est = estimate_N(mu_prior, sigma_prior, self.sigma_reduction_prop * sigma_prior)
                #print(f"For HF proposal with mu_prior={mu_prior}, sigma_prior={sigma_prior}, then N_est={N_est}")

                mi = self._estimate_MI(current_proposals + [(1, x)], model, X_prime)

                cand_results.append(CandidateResult(
                    mi=mi,
                    fidelity=1,
                    cand_ind=X_HF_cand_ind_map[i],
                    N=N_est,
                ))

        all_Ns = np.array([x.N for x in cand_results]).astype(float)
        all_Ns = all_Ns * self.max_N / all_Ns.max()
        all_Ns_normalised = [int(ceil(x)) for x in all_Ns]

        all_costs = []
        all_cost_weighted_delta_mi = []
        for cand_result, N_normalised in zip(cand_results, all_Ns_normalised):
            if cand_result.fidelity == 0:
                all_costs.append(self.dataset.c_LF * N_normalised)
            elif cand_result.fidelity == 1:
                all_costs.append(self.dataset.c_HF * N_normalised)
            all_cost_weighted_delta_mi.append((cand_result.mi - base_mi) / all_costs[-1])

        # Now select index with highest cost weighted delta mi
        i = np.argmax(all_cost_weighted_delta_mi)
        optimum_cand_ind = cand_results[i].cand_ind
        optimum_fidelity = cand_results[i].fidelity
        optimum_N = all_Ns_normalised[i]
        optimum_mi = all_cost_weighted_delta_mi[i]

        logger.info(f"Greedy solve completed, optimum: {optimum_fidelity}, MI={optimum_mi:.4f}")

        if plot:
            #self._plot_all_scores(plot_data)
            self.plot_ind += 1

        assert optimum_cand_ind >= 0

        return optimum_fidelity, optimum_cand_ind, optimum_N


_norm = torch.distributions.Normal(0, 1)


def get_posterior_sigma(mu_prior, sigma_prior, N: int):
    k_expected = N * _norm.cdf(torch.tensor(mu_prior))
    _, sigma_post = laplace_approximation_probit(
        mu_prior, sigma_prior, N, k_expected
    )
    return sigma_post


def estimate_N(mu_prior, sigma_prior, target_posterior_sigma, N_min=1, N_max=100):
    """
    Estimate the minimum N such that the posterior sigma is <= target_posterior_sigma.
    """

    def f(N):
        # N can be non-integer, just for finding root
        return get_posterior_sigma(mu_prior, sigma_prior, int(N)) - target_posterior_sigma

    # First, check if target is already lower than possible
    if f(N_max) > 0:
         return N_max
    if f(N_min) < 0:
        return N_min

    sol = root_scalar(f, bracket=[N_min, N_max], method='bisect', xtol=1.0, maxiter=25)
    if not sol.converged:
        logger.critical("Root minimization did not converge")
    return int(sol.root)
