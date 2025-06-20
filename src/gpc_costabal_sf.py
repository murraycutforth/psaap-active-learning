#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 16:30:49 2018 (Updated for Python 3 and PyTorch/GPyTorch)

@author: fsc
"""
import numpy as np
import torch
import gpytorch
import sys
from matplotlib import pyplot as plt
from pyDOE import lhs  # For Latin Hypercube Sampling
from sklearn.metrics import precision_recall_fscore_support
from scipy.spatial import cKDTree  # For finding diverse points

from src.gpc_costabal_utils import *

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# --- Base Classifier Class ---
class BaseClassifier(object):
    def __init__(self, lb, ub, sampling_func=None, X_test=None, Y_test=None, dtype=torch.float32):
        self.dtype = dtype
        self.lb = torch.tensor(lb, dtype=self.dtype, device=DEVICE)
        self.ub = torch.tensor(ub, dtype=self.dtype, device=DEVICE)
        self.sampling_func = sampling_func
        self.boundary = None  # For plotting true boundary if known

        self.h = 0.05  # Grid resolution for plotting

        self.X_next = None  # Stores the next point(s) to be queried (normalized tensor)
        self.pred_samples_f_grid = None  # Samples of latent f on a grid (tensor)
        self.pred_samples_f_cand = None  # Samples of latent f on candidate points (tensor)

        self.model = None
        self.likelihood = None

        self.X_test_norm = None  # Normalized test data (tensor)
        self.Y_true_test = None  # True labels for test data (tensor)
        self.error_history = []  # Store error metrics

        if X_test is not None:
            X_test_tensor = torch.tensor(X_test, dtype=self.dtype, device=DEVICE)
            self.X_test_norm = normalize(X_test_tensor, self.lb, self.ub)
            if Y_test is None and self.sampling_func is not None:
                X_test_orig_scale = denormalize(self.X_test_norm, self.lb, self.ub).cpu().numpy()
                y_true_np = self.sampling_func(X_test_orig_scale)
                self.Y_true_test = torch.tensor(y_true_np, dtype=self.dtype, device=DEVICE)
            elif Y_test is not None:
                self.Y_true_test = torch.tensor(Y_test, dtype=self.dtype, device=DEVICE)

    def _create_gpytorch_model(self, train_x, train_y):
        """To be implemented by subclasses to define the GPyTorch model and likelihood."""
        raise NotImplementedError("Subclasses must implement _create_gpytorch_model")

    def train_model(self, train_x, train_y, training_iter=100, lr=0.1):
        """Trains the GPyTorch model."""
        self.model, self.likelihood = self._create_gpytorch_model(train_x, train_y)
        # Ensure model and likelihood are on the correct device (already handled in _create_gpytorch_model usually)
        self.model.to(DEVICE)
        self.likelihood.to(DEVICE)

        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        # Use the VariationalELBO for Approximate GPs
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=train_y.size(0))

        print("Training GPyTorch model...")
        for i in range(training_iter):
            optimizer.zero_grad()
            output = self.model(train_x)  # Output from the GP model (latent function f)
            loss = -mll(output, train_y)  # ELBO is maximized, so minimize -ELBO
            loss.backward()
            optimizer.step()
            if (i + 1) % 20 == 0 or i == 0 or i == training_iter - 1:
                print(f'Iter {i + 1}/{training_iter} - Loss: {loss.item():.3f} ' +
                      f'Lengthscale: {self.model.covar_module.base_kernel.lengthscale.mean().item():.3f} ' +
                      f'Outputscale: {self.model.covar_module.outputscale.item():.3f}')
        print("Training complete.")

    def sample_predictive_f(self, X_eval_norm, n_samples=100):
        """Samples from the posterior latent function f* for given X_eval_norm."""
        if self.model is None or self.likelihood is None:
            raise RuntimeError("Model not trained yet. Call train_model() first.")

        self.model.eval()
        self.likelihood.eval()

        X_eval_norm = X_eval_norm.to(DEVICE)  # Ensure data is on correct device

        predictions_list = []
        # GPyTorch can struggle with very large prediction batches for .sample()
        # Chunking prediction if X_eval_norm is too large
        max_chunk_size = 1024  # Adjust based on GPU memory
        num_chunks = (X_eval_norm.shape[0] + max_chunk_size - 1) // max_chunk_size

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for i in range(num_chunks):
                chunk_start = i * max_chunk_size
                chunk_end = min((i + 1) * max_chunk_size, X_eval_norm.shape[0])
                X_chunk = X_eval_norm[chunk_start:chunk_end]

                f_dist_chunk = self.model(X_chunk)  # Posterior distribution q(f*)
                # Samples from q(f*) have shape (n_samples, chunk_size)
                f_samples_chunk = f_dist_chunk.sample(torch.Size((n_samples,)))
                predictions_list.append(f_samples_chunk)

        # Concatenate samples from all chunks
        if not predictions_list:  # Should not happen if X_eval_norm is not empty
            return torch.tensor([], device=DEVICE)

        f_pred_samples = torch.cat(predictions_list, dim=1)  # Shape (n_samples, X_eval_norm.shape[0])
        return f_pred_samples.cpu()  # Return on CPU

    def sample_grid_f(self):
        """Samples latent function f on a regular grid for plotting."""
        self.xx_plot, self.yy_plot = np.meshgrid(
            np.arange(0, 1 + self.h, self.h),
            np.arange(0, 1 + self.h, self.h)
        )
        X_grid_norm_np = np.c_[self.xx_plot.ravel(), self.yy_plot.ravel()]
        X_grid_norm = torch.tensor(X_grid_norm_np, dtype=self.dtype)  # Device handled in sample_predictive_f

        self.pred_samples_f_grid = self.sample_predictive_f(X_grid_norm, n_samples=100)

    def plot(self, filename, cand=False):
        raise NotImplementedError("Subclasses must implement plot method")

    def compute_next_point_cand(self, N_points_to_select=1):
        """Computes the next candidate point(s) for active learning."""
        # self.X_cand_norm is a tensor of candidate points
        # pred_samples_f_cand has shape (n_samples, n_cand_points)
        self.pred_samples_f_cand = self.sample_predictive_f(self.X_cand_norm, n_samples=500)

        f_cand_mean = self.pred_samples_f_cand.mean(dim=0)  # Shape (n_cand_points)
        f_cand_std = self.pred_samples_f_cand.std(dim=0)  # Shape (n_cand_points)

        # Acquisition score: -|mean(f)| / (std(f) + eps) -> aim for mean(f) near 0 and high std(f)
        # We want to MAXIMIZE this score.
        acquisition_score = -torch.abs(f_cand_mean) / (f_cand_std + 1e-9)

        X_cand_np = self.X_cand_norm.cpu().numpy()
        acquisition_score_np = acquisition_score.cpu().numpy()

        if N_points_to_select == 1:
            best_idx = acquisition_score_np.argmax()
            self.X_next = self.X_cand_norm[best_idx][None, :]  # Keep as tensor, add batch dim
            print(f"Next point (normalized): {self.X_next.cpu().numpy()}, Score: {acquisition_score_np.max():.4f}")
        else:
            # Find N_points_to_select diverse local maxima of the acquisition score
            # Using the cKDTree logic from original code for finding local maxima
            tree = cKDTree(X_cand_np)

            # Start with a small neighborhood, increase until enough local maxima are found
            # or a max neighborhood size is reached.
            selected_indices = []

            # Iteratively find local maxima and filter them
            # This is a simplified greedy approach: sort all candidates by score, pick top N diverse ones.
            # The original logic is more about finding true local maxima.
            # Let's try to implement the original local maxima search more closely.

            sorted_indices_by_score = np.argsort(acquisition_score_np)[::-1]  # Best scores first

            potential_maxima_indices = []
            # Check for local maxima using a growing neighborhood
            # This can be complex. A simpler alternative for batch selection:
            # 1. Pick best point. 2. Penalize area around it. 3. Repeat.
            # Or, use the original logic:

            neigh_to_consider = 2
            current_X_cand = X_cand_np
            current_acq_score = acquisition_score_np

            final_selected_indices_in_original_X_cand = []

            # This loop is a bit tricky; the original finds local maxima from the *full* set X_cand.
            # If we want N points, we want the N "best" local maxima.
            # Let's find all local maxima with a reasonable neighborhood, then pick top N from them.

            # Find all local maxima with a fixed neighborhood size (e.g., 5% of domain width)
            # This is a simplification of the original iterative neigh_to_consider.
            dist_threshold_for_local_max = 0.1  # Heuristic
            num_neighbors_for_local_max_check = 0
            if current_X_cand.shape[0] > 1:
                # Estimate how many neighbors fall within dist_threshold
                dists, _ = tree.query(current_X_cand[0], k=min(20, current_X_cand.shape[0]))
                num_neighbors_for_local_max_check = max(2, np.sum(dists < dist_threshold_for_local_max))

            if current_X_cand.shape[0] <= N_points_to_select or num_neighbors_for_local_max_check <= 1:
                # If not enough candidates, or too few for neighbor check, take top N by score
                top_N_indices = sorted_indices_by_score[:N_points_to_select]
                final_selected_indices_in_original_X_cand = top_N_indices
            else:
                _, neigh_indices = tree.query(current_X_cand, k=num_neighbors_for_local_max_check)

                # A point is a local max if its score is >= all its neighbors' scores
                # (excluding itself, so compare with neigh_indices[:, 1:])
                is_local_max = np.all(current_acq_score[:, None] >= current_acq_score[neigh_indices[:, 1:]], axis=1)
                local_maxima_indices = np.where(is_local_max)[0]

                if len(local_maxima_indices) == 0:  # Fallback if no local maxima found
                    local_maxima_indices = sorted_indices_by_score[:N_points_to_select]

                # Sort these local maxima by their acquisition score
                sorted_local_maxima_indices = local_maxima_indices[
                    np.argsort(current_acq_score[local_maxima_indices])[::-1]
                ]

                # Select top N_points_to_select from these sorted local maxima
                final_selected_indices_in_original_X_cand = sorted_local_maxima_indices[:N_points_to_select]

            self.X_next = self.X_cand_norm[final_selected_indices_in_original_X_cand]

            # Optionally, remove selected points from self.X_cand_norm for future rounds if X_cand is persistent
            # mask = torch.ones(self.X_cand_norm.shape[0], dtype=torch.bool)
            # mask[final_selected_indices_in_original_X_cand] = False
            # self.X_cand_norm = self.X_cand_norm[mask]
            # self.pred_samples_f_cand = self.pred_samples_f_cand[:, mask]

    def append_next_point(self):
        """To be implemented by subclasses to add self.X_next to their training data."""
        raise NotImplementedError("Subclasses must implement append_next_point")

    def test_model(self):
        """Tests the model on the test set and records performance."""
        if self.X_test_norm is None or self.Y_true_test is None:
            print("No test data provided. Skipping model test.")
            return

        # pred_f_test has shape (n_samples, n_test_points)
        pred_f_test = self.sample_predictive_f(self.X_test_norm, n_samples=500)

        prob_test = invlogit(pred_f_test)  # Convert latent f samples to probability samples
        mean_prob_test = prob_test.mean(dim=0)  # Mean probability across samples

        Y_pred_test_np = torch.round(mean_prob_test).cpu().numpy()
        Y_true_test_np = self.Y_true_test.cpu().numpy()

        misclassifications = np.sum(np.abs(Y_true_test_np - Y_pred_test_np))
        total_test_points = self.X_test_norm.shape[0]

        if total_test_points == 0:
            print("Warning: No test points to evaluate.")
            accuracy = 0.0
            p_r_f_s = (0.0, 0.0, 0.0, None)
        else:
            accuracy = 100. * (1 - misclassifications / total_test_points)
            # Use average='binary' if classes are 0 and 1.
            # If Y_true_test_np can be all same class, handle zero_division
            try:
                p_r_f_s = precision_recall_fscore_support(Y_true_test_np, Y_pred_test_np, average='binary',
                                                          zero_division=0)
            except ValueError:  # Happens if Y_true_test_np is all one class for example and average='binary'
                p_r_f_s = precision_recall_fscore_support(Y_true_test_np, Y_pred_test_np, average='weighted',
                                                          zero_division=0)
                print("Warning: Using weighted average for precision/recall/F1 due to data characteristics.")

        self.error_history.append({
            'misclassifications': misclassifications,
            'accuracy': accuracy,
            'precision': p_r_f_s[0],
            'recall': p_r_f_s[1],
            'f1_score': p_r_f_s[2]
        })
        print(
            f"Test Accuracy: {accuracy:.2f}% ({int(misclassifications)} misclassifications out of {total_test_points})")
        print(f"Precision: {p_r_f_s[0]:.3f}, Recall: {p_r_f_s[1]:.3f}, F1-Score: {p_r_f_s[2]:.3f}")

    def _get_current_training_data(self):
        """Helper to be implemented by subclasses to return current training (X,Y)"""
        raise NotImplementedError

    def active_learning(self, N_iterations=15, N_points_per_iteration=1, plot_active=False,
                        filename_template='active_learning_%i.png'):
        current_train_x, current_train_y = self._get_current_training_data()
        self.train_model(current_train_x, current_train_y)  # Trains self.model

        for i in range(N_iterations):
            print(f"\n--- Active Learning Iteration {i + 1}/{N_iterations} ---")
            current_train_x, current_train_y = self._get_current_training_data()

            if plot_active:
                # Ensure directory exists if template includes path
                import os
                os.makedirs(os.path.dirname(filename_template), exist_ok=True)
                self.plot(filename=(
                    filename_template % i if '%' in filename_template else filename_template + f"_{i}.png"),
                    cand=False)

            self.train_model(current_train_x, current_train_y)  # Trains self.model

            if self.X_test_norm is not None:
                self.test_model()

            if i < N_iterations:  # No need to find next point after last iteration's training
                self.compute_next_point_cand(N_points_per_iteration)
                self.append_next_point()  # Adds self.X_next to training data for next iteration

        # Final plot if requested, showing final state without candidate points
        if plot_active:
            self.plot(filename=(
                "output_plots/active_learning_final.png"),
                      cand=False)


# --- GPyTorch Approximate GP Model for (Single Fidelity) Classification ---
class ApproximateGPModelSF(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, dim, initial_lengthscale=0.5):
        # Inducing points should be a tensor
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=False
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=dim if dim > 0 else None)
            # ard_num_dims=None if dim=0 or 1 (not ARD)
        )
        if dim > 0:
            self.covar_module.base_kernel.lengthscale = torch.ones(dim) * initial_lengthscale
        else:  # Should not happen if dim is correctly passed, or for 1D non-ARD
            self.covar_module.base_kernel.lengthscale = initial_lengthscale

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# --- Single-Fidelity Classifier Base ---
class BaseSFclassifier(BaseClassifier):
    def __init__(self, X_train_init, Y_train_init, lb, ub, sampling_func,
                 X_test=None, Y_test=None, N_cand=1000, boundary=None, dtype=torch.float32):
        super().__init__(lb, ub, sampling_func=sampling_func, X_test=X_test, Y_test=Y_test, dtype=dtype)

        self.dim = X_train_init.shape[1]
        # Initial training data (normalized tensors)
        self.train_x_norm = normalize(torch.tensor(X_train_init, dtype=self.dtype, device=DEVICE), self.lb, self.ub)
        self.train_y = torch.tensor(Y_train_init, dtype=self.dtype, device=DEVICE)

        # Candidate points for active learning (normalized tensor)
        X_cand_np = lhs(self.dim, N_cand, criterion='maximin', iterations=50)  # Improved LHS
        self.X_cand_norm = torch.tensor(X_cand_np, dtype=self.dtype,
                                        device=DEVICE)  # To device later if needed by compute_next_point_cand

        self.boundary = boundary  # For plotting true boundary

    def _get_current_training_data(self):
        return self.train_x_norm.to(DEVICE), self.train_y.to(DEVICE)

    def plot(self, filename='GP_SF.png', cand=False):
        assert self.dim == 2, 'can only plot 2D functions for this method'

        if self.model is None:
            print("Model not trained. Skipping plot.")
            return

        plt.figure(figsize=(6, 10))  # Create new figure
        # plt.clf() # Not needed if creating new figure

        train_x_np = self.train_x_norm.cpu().numpy()
        train_y_np = self.train_y.cpu().numpy()

        # Subplot 1: Class Probability
        plt.subplot(211)
        if cand and self.pred_samples_f_cand is not None:
            prob_cand = invlogit(self.pred_samples_f_cand).mean(dim=0).cpu().numpy()
            X_cand_np = self.X_cand_norm.cpu().numpy()
            cnt = plt.tricontourf(X_cand_np[:, 0], X_cand_np[:, 1], prob_cand, levels=np.linspace(0, 1, 100),
                                  cmap='viridis')
            plt.title("Predicted Class Probability (Candidates)")
        else:
            if self.pred_samples_f_grid is None: self.sample_grid_f()
            prob_grid = invlogit(self.pred_samples_f_grid).mean(dim=0).cpu().numpy()
            cnt = plt.contourf(self.xx_plot, self.yy_plot, prob_grid.reshape(self.xx_plot.shape),
                               levels=np.linspace(0, 1, 100), cmap='viridis')
            plt.title("Predicted Class Probability (Grid)")

        if hasattr(cnt, 'collections'):  # For tricontourf
            for c_ in cnt.collections: c_.set_edgecolor("face")

        cb = plt.colorbar(cnt, ticks=[0, 0.5, 1])
        cb.set_label('P(Y=1|X)', labelpad=-10)
        if cb.ax.get_yticklabels():
            cb.ax.get_yticklabels()[0].set_verticalalignment("bottom")
            cb.ax.get_yticklabels()[-1].set_verticalalignment("top")

        plt.scatter(train_x_np[:, 0], train_x_np[:, 1], c=train_y_np, edgecolors='k', cmap='coolwarm', vmin=0, vmax=1,
                    s=50, zorder=2)
        if self.X_next is not None and cand:  # Only plot X_next if cand is true
            X_next_np = self.X_next.cpu().numpy()
            plt.scatter(X_next_np[:, 0], X_next_np[:, 1], color='lime', marker='*', s=250, edgecolors='k',
                        label='Next Point(s)', zorder=3)
        if self.boundary is not None:
            xf_plot = np.linspace(0, 1, 100)[:, None]
            yf_plot = self.boundary(xf_plot)  # Assumes boundary function takes normalized input
            plt.plot(xf_plot.squeeze(), yf_plot.squeeze(), 'k--', label='True Boundary', zorder=1)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        prettyplot("Normalized X_1", "Normalized X_2", ylabelpad=-10)
        plt.legend()

        # Subplot 2: Acquisition Function
        plt.subplot(212)
        if cand and self.pred_samples_f_cand is not None:
            f_cand_mean = self.pred_samples_f_cand.mean(dim=0).cpu().numpy()
            f_cand_std = self.pred_samples_f_cand.std(dim=0).cpu().numpy()
            acq_values = -np.abs(f_cand_mean) / (f_cand_std + 1e-9)
            X_cand_np = self.X_cand_norm.cpu().numpy()
            cnt_acq = plt.tricontourf(X_cand_np[:, 0], X_cand_np[:, 1], acq_values, 100, cmap='magma')
            plt.title("Acquisition Function (Candidates)")
        else:
            if self.pred_samples_f_grid is None: self.sample_grid_f()  # Should have been called by prob plot
            f_grid_mean = self.pred_samples_f_grid.mean(dim=0).cpu().numpy()
            f_grid_std = self.pred_samples_f_grid.std(dim=0).cpu().numpy()
            acq_values_grid = -np.abs(f_grid_mean) / (f_grid_std + 1e-9)
            cnt_acq = plt.contourf(self.xx_plot, self.yy_plot, acq_values_grid.reshape(self.xx_plot.shape), 100,
                                   cmap='magma')
            plt.title("Acquisition Function (Grid)")

        if hasattr(cnt_acq, 'collections'):
            for c_ in cnt_acq.collections: c_.set_edgecolor("face")
        cb_acq = plt.colorbar(cnt_acq, label='Acquisition Score')

        plt.scatter(train_x_np[:, 0], train_x_np[:, 1], c=train_y_np, edgecolors='k', cmap='coolwarm', vmin=0, vmax=1,
                    s=50, zorder=2)
        if self.X_next is not None and cand:
            X_next_np = self.X_next.cpu().numpy()
            plt.scatter(X_next_np[:, 0], X_next_np[:, 1], color='lime', marker='*', s=250, edgecolors='k', zorder=3)
        if self.boundary is not None:
            xf_plot = np.linspace(0, 1, 100)[:, None]
            yf_plot = self.boundary(xf_plot)
            plt.plot(xf_plot.squeeze(), yf_plot.squeeze(), 'k--', zorder=1)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        prettyplot("Normalized X_1", "Normalized X_2", ylabelpad=-10)

        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename, dpi=150)
            print(f"Plot saved to {filename}")
        # plt.show() # Uncomment to display plot interactively, but might interfere with saving in loops

    def append_next_point(self):
        """Adds the selected next point(s) to the training data."""
        if self.X_next is None:
            print("No next point computed yet to append.")
            return

        # self.X_next is already normalized and on DEVICE (or should be)
        self.train_x_norm = torch.cat((self.train_x_norm.to(DEVICE), self.X_next.to(DEVICE)), dim=0)

        # Get new Y value(s) using the sampling function (expects denormalized X)
        X_next_denorm_np = denormalize(self.X_next, self.lb, self.ub).cpu().numpy()

        # Sampling function might return scalar or array depending on X_next_denorm_np shape
        Y_next_np = self.sampling_func(X_next_denorm_np)
        if not isinstance(Y_next_np, np.ndarray):  # Ensure it's an array
            Y_next_np = np.array([Y_next_np])
        Y_next_np = Y_next_np.astype(float).reshape(-1)  # Ensure float and 1D for single output per point

        Y_next_tensor = torch.tensor(Y_next_np, dtype=self.dtype, device=DEVICE)
        self.train_y = torch.cat((self.train_y.to(DEVICE), Y_next_tensor), dim=0)

        print(f"Appended {self.X_next.shape[0]} point(s). New training set size: {self.train_x_norm.shape[0]}")
        self.X_next = None  # Clear X_next after appending


# --- Gaussian Process Classifier (Single Fidelity) ---
class GPClassifier(BaseSFclassifier):
    def _create_gpytorch_model(self, train_x, train_y):
        # Determine number of inducing points (e.g., fraction of data, or fixed)
        num_inducing = min(128, train_x.size(0))

        # Select inducing points (e.g., randomly from training data)
        # Ensure train_x is on CPU for randperm if not already.
        # Or ensure inducing_points is on DEVICE.
        if train_x.size(0) > 0:
            inducing_points_idx = torch.randperm(train_x.size(0), device=train_x.device)[:num_inducing]
            inducing_points = train_x[inducing_points_idx, :]
        else:  # Handle case with no training data initially, though unlikely for GP
            inducing_points = torch.empty(0, train_x.size(1), device=train_x.device)

        likelihood = gpytorch.likelihoods.BernoulliLikelihood().to(DEVICE)
        model = ApproximateGPModelSF(inducing_points=inducing_points, dim=self.dim).to(DEVICE)
        return model, likelihood


# --- Example Usage ---
if __name__ == '__main__':
    # Define a synthetic 2D binary classification problem
    def true_boundary_function_normalized(x_norm):
        # x_norm is an N x 1 array (or N x D, using x_norm[:,0]) of normalized X1 values
        # Returns corresponding Y values for the boundary
        return 0.2 * np.sin(2 * np.pi * x_norm[:, 0]) + 0.5 + 0.1 * x_norm[:, 0]


    def sampling_function(X_denormalized):
        # X_denormalized is N x D (where D=2)
        # Normalize X before applying boundary function
        # This assumes lb_example and ub_example are accessible or passed if this function is external
        lb_ex = np.array([0., 0.])
        ub_ex = np.array([10., 10.])
        X_normalized = normalize(X_denormalized, lb_ex, ub_ex)

        # Decision boundary: Y=1 if X_normalized[:,1] > boundary_func(X_normalized[:,0])
        boundary_vals = true_boundary_function_normalized(X_normalized[:, 0][:, None])
        return (X_normalized[:, 1] > boundary_vals).astype(float)


    # Define bounds for the input space
    lb_example = np.array([0., 0.])
    ub_example = np.array([10., 10.])

    # Initial training data (small set)
    N_initial_train = 50
    X_train_init_np = denormalize(lhs(2, N_initial_train, criterion='maximin', iterations=20), lb_example, ub_example)
    Y_train_init_np = sampling_function(X_train_init_np)

    # Test data
    N_test = 1000
    X_test_np = denormalize(lhs(2, N_test), lb_example, ub_example)
    # Y_test_np = sampling_function(X_test_np) # Y_test can be inferred by BaseClassifier if None

    # Create classifier instance
    gpc = GPClassifier(
        X_train_init=X_train_init_np,
        Y_train_init=Y_train_init_np,
        lb=lb_example,
        ub=ub_example,
        sampling_func=sampling_function,
        X_test=X_test_np,  # Y_test will be generated by sampling_func
        N_cand=1000,  # Number of candidate points for active learning
        boundary=lambda x_norm_X1: true_boundary_function_normalized(x_norm_X1)  # Pass the true boundary for plotting
    )

    # Run active learning
    gpc.active_learning(
        N_iterations=5,  # Number of active learning rounds
        N_points_per_iteration=20,  # Number of points to add per round
        plot_active=True,
        filename_template='output_plots/sf_gpc_active_iter_%i.png'  # Make sure 'output_plots' directory exists
    )

    import os

    os.makedirs("output_plots", exist_ok=True)

    print("\nActive learning finished.")
    if gpc.error_history:
        print("Test accuracy history:")
        for i, err_metric in enumerate(gpc.error_history):
            print(f"Iter {i + 1}: Accuracy = {err_metric['accuracy']:.2f}%")