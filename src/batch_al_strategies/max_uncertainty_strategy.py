import logging
from pathlib import Path

import numpy as np
import pyDOE
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

from src.batch_al_strategies.base import BiFidelityBatchALStrategy
from src.active_learning.util_classes import BiFidelityModel, BiFidelityDataset, ALExperimentConfig

logger = logging.getLogger(__name__)


class MaxUncertaintyStrategy(BiFidelityBatchALStrategy):
    """
    Max-Uncertainty Strategy.

    This strategy selects a batch of points by combining two types of uncertainty
    and enforcing diversity through clustering.

    1.  **Model Uncertainty (Epistemic):** The variance of the model's prediction
        for the Bernoulli probability `p`. Captured by `model.predict_prob_var()`.
        This tells us where the model is uncertain about its own parameters.

    2.  **Data Uncertainty (Aleatoric):** The inherent randomness of the data. For a
        Bernoulli trial, this is maximized when the probability `p` is 0.5.
        We measure this using the entropy of the Bernoulli distribution, calculated
        from the model's mean prediction `model.predict_prob_mean()`.

    The acquisition score for each point is a weighted sum of model and data uncertainty:
        score = model_uncertainty + beta * data_uncertainty

    The bi-fidelity budget is split between HF and LF queries using a `gamma` parameter.
    """

    def __init__(self, dataset: BiFidelityDataset, beta: float = 0.5, gamma: float = 0.5,
                 plot_all_scores: bool = False):
        """
        Initializes the MUD Strategy.

        Args:
            dataset (BiFidelityDataset): The dataset manager.
            beta (float, optional): A weighting factor for the data uncertainty (entropy)
                term in the acquisition score. Defaults to 1.0.
            gamma (float, optional): The fraction of the total budget to be allocated
                to high-fidelity queries. Must be between 0 and 1.
        """
        super().__init__(dataset)
        if not 0 <= gamma <= 1:
            raise ValueError("gamma (HF budget fraction) must be between 0 and 1.")
        self.beta = beta
        self.gamma = gamma
        self.plot_all_scores = plot_all_scores
        self.plot_ind = 0
        self.fig_outdir = Path(__file__).parent / "figures" / str(self)
        self.fig_outdir.mkdir(parents=True, exist_ok=True)

    def __str__(self):
        return f"MaxUncertaintyStrategy(beta={self.beta}, gamma={self.gamma})"

    def _calculate_acquisition_scores(self, X_cand: np.ndarray, model: BiFidelityModel) -> np.ndarray:
        """Calculates the acquisition score for a set of candidate points."""
        if len(X_cand) == 0:
            return np.array([])

        prob_means, prob_vars = model.predict_hf_prob(X_cand), model.predict_hf_prob_var(X_cand)

        # 1. Model Uncertainty (Epistemic)
        model_uncertainty = prob_vars
        model_uncertainty /= model_uncertainty.max()

        # 2. Data Uncertainty (Aleatoric) - measured by Bernoulli entropy
        # -p*log2(p) - (1-p)*log2(1-p)
        # Add a small epsilon to avoid log(0)
        epsilon = 1e-12
        p = np.clip(prob_means, epsilon, 1 - epsilon)
        data_uncertainty = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
        data_uncertainty /= data_uncertainty.max()

        logger.info(f"Computed data_uncertainty: {data_uncertainty}, model_uncertainty: {model_uncertainty}")

        # Combine scores
        acquisition_scores = (1.0 - self.beta) * model_uncertainty + self.beta * data_uncertainty
        return acquisition_scores

    def _select_top_scores_from_pool(self, X_pool: np.ndarray, scores: np.ndarray, n_to_select: int) -> list[int]:
        """Just select the top `n_to_select` candidates from the pool."""
        assert len(X_pool) == len(scores)

        sorted_inds = np.argsort(scores)[::-1]
        return sorted_inds[:n_to_select]

    def _select_diverse_batch_from_pool(self,
                                        X_pool: np.ndarray,
                                        scores: np.ndarray,
                                        n_to_select: int) -> list[int]:
        """
        Selects a diverse batch from a pool of candidates using k-Means.

        Returns:
            A list of original indices from the pool.
        """
        assert len(X_pool) == len(scores)
        X_pool = np.array(X_pool)

        if n_to_select == 0 or len(X_pool) == 0:
            return []

        # Ensure we don't request more clusters than available points
        n_clusters = min(n_to_select, len(X_pool))

        if n_clusters == 1:
            # If only one point is needed, just pick the best one
            best_idx = np.argmax(scores)
            return [best_idx]

        if np.isnan(X_pool).any():
            print("Your data contains NaN values!")

        if np.isinf(X_pool).any():
            print("Your data contains Inf values!")

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=5, max_iter=500, verbose=0)
        kmeans.fit(X_pool)

        selected_indices = []
        for i in range(n_clusters):
            # Get the indices of all points in the current cluster
            cluster_mask = (kmeans.labels_ == i)

            # This can happen if a cluster ends up empty, though rare with standard k-means
            if not np.any(cluster_mask):
                continue

            # Get the original indices and scores for points in this cluster
            original_indices_in_cluster = np.where(cluster_mask)[0]
            scores_in_cluster = scores[cluster_mask]

            # Find the index of the point with the max score *within this cluster*
            best_local_idx = np.argmax(scores_in_cluster)

            # Map it back to the original index in the pool
            best_original_idx = original_indices_in_cluster[best_local_idx]
            selected_indices.append(int(best_original_idx))

        return selected_indices

    def select_batch(self,
                     config: ALExperimentConfig,
                     current_model_trained: BiFidelityModel,
                     budget_this_step: float
                     ) -> tuple[np.ndarray, np.ndarray]:
        """
        Selects a batch of LF and HF points based on uncertainty and diversity.
        """
        # 1. Allocate budget
        hf_budget = self.gamma * budget_this_step
        lf_budget = (1 - self.gamma) * budget_this_step

        n_hf_to_select = int(hf_budget / self.dataset.c_HF)
        n_lf_to_select = int(lf_budget / self.dataset.c_LF)

        logger.info(f"n_hf_to_select: {n_hf_to_select}, n_lf_to_select: {n_lf_to_select}")

        # 2. Process High-Fidelity Pool
        X_hf_cand_pool = self._generate_lhs_samples(config, config.N_cand_HF)
        hf_scores = self._calculate_acquisition_scores(X_hf_cand_pool, current_model_trained)
        #final_hf_indices = self._select_diverse_batch_from_pool(
        #    X_hf_cand_pool, hf_scores, n_hf_to_select
        #)
        final_hf_indices = self._select_top_scores_from_pool(X_hf_cand_pool, hf_scores, n_hf_to_select)

        # 3. Process Low-Fidelity Pool
        X_lf_cand_pool = self._generate_lhs_samples(config, config.N_cand_LF)
        lf_scores = self._calculate_acquisition_scores(X_lf_cand_pool, current_model_trained)
        #final_lf_indices = self._select_diverse_batch_from_pool(X_lf_cand_pool, lf_scores, n_lf_to_select)
        final_lf_indices = self._select_top_scores_from_pool(X_lf_cand_pool, lf_scores, n_lf_to_select)

        if self.plot_all_scores:
            self._plot_all_scores(lf_scores, hf_scores, X_lf_cand_pool, X_hf_cand_pool)

        X_lf_final = X_lf_cand_pool[final_lf_indices]
        X_hf_final = X_hf_cand_pool[final_hf_indices]

        return X_lf_final, X_hf_final

    def _plot_all_scores(self, lf_scores: np.ndarray, hf_scores: np.ndarray, X_lf_cand_pool, X_hf_cand_pool) -> None:
        fig, axs = plt.subplots(1, 2, figsize=(8, 3), dpi=200)

        ax_lf = axs[0]
        ax_hf = axs[1]

        im = ax_lf.scatter(X_lf_cand_pool[:, 0], X_lf_cand_pool[:, 1], c=lf_scores, s=20)
        fig.colorbar(im, ax=ax_lf, label="Acquisition function")

        im = ax_hf.scatter(X_hf_cand_pool[:, 0], X_hf_cand_pool[:, 1], c=hf_scores, s=20)
        fig.colorbar(im, ax=ax_hf, label="Acquisition function")

        ax_lf.set_aspect('equal')
        ax_hf.set_aspect('equal')

        ax_lf.set_title("LF")
        ax_hf.set_title("HF")

        fig.tight_layout()
        fig.savefig(self.fig_outdir / f"{self.plot_ind}.png")
        plt.close(fig)

        self.plot_ind += 1
