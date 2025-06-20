#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 16:30:49 2018 (Updated for Python 3 and PyTorch/GPyTorch)

@author: fsc
"""
import linear_operator
import numpy as np
import torch
import gpytorch
import sys
from matplotlib import pyplot as plt
from pyDOE import lhs
from linear_operator.operators import BlockLinearOperator, CatLinearOperator
from sklearn.metrics import precision_recall_fscore_support
from scipy.spatial import cKDTree

from src.gpc_costabal_utils import *
from src.gpc_costabal_sf import BaseClassifier


# Device configuration (can be inherited or redefined if needed, usually set globally)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"MF Classifier using device: {DEVICE}")


# --- Multi-Fidelity Base Classifier ---
class BaseMFclassifier(BaseClassifier):
    def __init__(self, X_L_init, Y_L_init, X_H_init, Y_H_init,
                 lb, ub, sampling_func,
                 X_test=None, Y_test=None, N_cand=1000,
                 boundary_H=None, boundary_L=None, dtype=torch.float32):

        super().__init__(lb, ub, sampling_func=sampling_func, X_test=X_test, Y_test=Y_test, dtype=dtype)

        self.dim = X_H_init.shape[1]

        # Normalized low-fidelity data
        self.train_x_l_norm = normalize(torch.tensor(X_L_init, dtype=self.dtype, device=DEVICE), self.lb, self.ub)
        self.train_y_l = torch.tensor(Y_L_init, dtype=self.dtype, device=DEVICE)

        # Normalized high-fidelity data
        self.train_x_h_norm = normalize(torch.tensor(X_H_init, dtype=self.dtype, device=DEVICE), self.lb, self.ub)
        self.train_y_h = torch.tensor(Y_H_init, dtype=self.dtype, device=DEVICE)

        self.num_low_fidelity = self.train_x_l_norm.shape[0]
        self.num_high_fidelity = self.train_x_h_norm.shape[0]

        # Concatenated training data for the GPyTorch model
        # Order: Low-fidelity points first, then High-fidelity points
        self.train_x_all = torch.cat((self.train_x_l_norm, self.train_x_h_norm), dim=0)
        self.train_y_all = torch.cat((self.train_y_l, self.train_y_h), dim=0)

        # Candidate points for active learning (normalized tensor)
        # Active learning queries are for high-fidelity points
        X_cand_np = lhs(self.dim, N_cand, criterion='maximin', iterations=50)
        self.X_cand_norm = torch.tensor(X_cand_np, dtype=self.dtype, device=DEVICE)

        self.boundary_H = boundary_H  # For plotting true HF boundary
        self.boundary_L = boundary_L  # For plotting true LF boundary

    def _get_current_training_data(self):
        # Update num_low_fidelity for the kernel, in case it changes (though not typical in this setup)
        self.num_low_fidelity = self.train_x_l_norm.shape[0]
        return self.train_x_all.to(DEVICE), self.train_y_all.to(DEVICE)

    def plot(self, filename='MFGP.png', cand=False):
        assert self.dim == 2, 'can only plot 2D functions for this method'

        if self.model is None:
            print("Model not trained. Skipping plot.")
            return

        plt.figure(figsize=(6, 10))

        # Data for scatter plots (CPU numpy arrays)
        train_x_l_np = self.train_x_l_norm.cpu().numpy()
        train_y_l_np = self.train_y_l.cpu().numpy()
        train_x_h_np = self.train_x_h_norm.cpu().numpy()
        train_y_h_np = self.train_y_h.cpu().numpy()

        # Subplot 1: Class Probability (of high-fidelity f_H)
        plt.subplot(211)
        if cand:
            self.sample_cand_f()  # Always resample for candidates
            if self.pred_samples_f_cand is not None:
                prob_cand = invlogit(self.pred_samples_f_cand).mean(dim=0).cpu().numpy()
                X_cand_np = self.X_cand_norm.cpu().numpy()
                cnt = plt.tricontourf(X_cand_np[:, 0], X_cand_np[:, 1], prob_cand, levels=np.linspace(0, 1, 100),
                                      cmap='viridis')
                plt.title("Predicted HF Class Probability P(Y_H=1|X) (Candidates)")
        else:
            self.sample_grid_f()  # Resample for grid
            if self.pred_samples_f_grid is not None:
                prob_grid = invlogit(self.pred_samples_f_grid).mean(dim=0).cpu().numpy()
                cnt = plt.contourf(self.xx_plot, self.yy_plot, prob_grid.reshape(self.xx_plot.shape),
                                   levels=np.linspace(0, 1, 100), cmap='viridis')
                plt.title("Predicted HF Class Probability P(Y_H=1|X) (Grid)")
            else:
                print("Warning: pred_samples_f_grid is None. Cannot plot.")

        if hasattr(cnt, 'collections'):
            for c_ in cnt.collections: c_.set_edgecolor("face")
        cb = plt.colorbar(cnt, ticks=[0, 0.5, 1])
        cb.set_label('P(Y_H=1|X)', labelpad=-10)
        if cb.ax.get_yticklabels():
            cb.ax.get_yticklabels()[0].set_verticalalignment("bottom")
            cb.ax.get_yticklabels()[-1].set_verticalalignment("top")

        # Plot LF points lightly, HF points more prominently
        if train_x_l_np.size > 0:
            plt.scatter(train_x_l_np[:, 0], train_x_l_np[:, 1], c=train_y_l_np, marker='s', alpha=0.5,
                        edgecolors='gray', cmap='coolwarm', vmin=0, vmax=1, s=30, label='LF Data', zorder=1)
        if train_x_h_np.size > 0:
            # Original code plotted self.X_H[:-1,0]... assuming last point is current X_next
            # For simplicity, plot all current HF points
            plt.scatter(train_x_h_np[:, 0], train_x_h_np[:, 1], c=train_y_h_np, marker='o', edgecolors='k',
                        cmap='coolwarm', vmin=0, vmax=1, s=60, label='HF Data', zorder=2)

        if self.X_next is not None and cand:
            X_next_np = self.X_next.cpu().numpy()
            plt.scatter(X_next_np[:, 0], X_next_np[:, 1], color='lime', marker='*', s=250, edgecolors='k',
                        label='Next HF Query', zorder=3)

        xf_plot = np.linspace(0, 1, 100)[:, None]
        if self.boundary_L is not None:
            yf_plot_l = self.boundary_L(xf_plot)
            plt.plot(xf_plot.squeeze(), yf_plot_l.squeeze(), 'b--', label='True LF Boundary', zorder=2)
        if self.boundary_H is not None:
            yf_plot_h = self.boundary_H(xf_plot)
            plt.plot(xf_plot.squeeze(), yf_plot_h.squeeze(), 'r-', label='True HF Boundary', zorder=2)

        plt.xlim([0, 1])
        plt.ylim([0, 1])
        prettyplot("Normalized X_1", "Normalized X_2", ylabelpad=-10)
        plt.legend(fontsize='small')

        # Subplot 2: Acquisition Function (for selecting next HF point)
        plt.subplot(212)
        if cand and self.pred_samples_f_cand is not None:
            f_cand_mean = self.pred_samples_f_cand.mean(dim=0).cpu().numpy()
            f_cand_std = self.pred_samples_f_cand.std(dim=0).cpu().numpy()
            acq_values = -np.abs(f_cand_mean) / (f_cand_std + 1e-9)  # Score for f_H
            X_cand_np = self.X_cand_norm.cpu().numpy()
            cnt_acq = plt.tricontourf(X_cand_np[:, 0], X_cand_np[:, 1], acq_values, 100, cmap='magma')
            plt.title("Acquisition Function for HF (Candidates)")
        else:
            if self.pred_samples_f_grid is None: self.sample_grid_f()
            f_grid_mean = self.pred_samples_f_grid.mean(dim=0).cpu().numpy()
            f_grid_std = self.pred_samples_f_grid.std(dim=0).cpu().numpy()
            acq_values_grid = -np.abs(f_grid_mean) / (f_grid_std + 1e-9)
            cnt_acq = plt.contourf(self.xx_plot, self.yy_plot, acq_values_grid.reshape(self.xx_plot.shape), 100,
                                   cmap='magma')
            plt.title("Acquisition Function for HF (Grid)")

        if hasattr(cnt_acq, 'collections'):
            for c_ in cnt_acq.collections: c_.set_edgecolor("face")
        cb_acq = plt.colorbar(cnt_acq, label='Acquisition Score')

        if train_x_l_np.size > 0:
            plt.scatter(train_x_l_np[:, 0], train_x_l_np[:, 1], c=train_y_l_np, marker='s', alpha=0.5,
                        edgecolors='gray', cmap='coolwarm', vmin=0, vmax=1, s=30, zorder=1)
        if train_x_h_np.size > 0:
            plt.scatter(train_x_h_np[:, 0], train_x_h_np[:, 1], c=train_y_h_np, marker='o', edgecolors='k',
                        cmap='coolwarm', vmin=0, vmax=1, s=60, zorder=2)
        if self.X_next is not None and cand:
            X_next_np = self.X_next.cpu().numpy()
            plt.scatter(X_next_np[:, 0], X_next_np[:, 1], color='lime', marker='*', s=250, edgecolors='k', zorder=3)

        if self.boundary_L is not None: plt.plot(xf_plot.squeeze(), self.boundary_L(xf_plot).squeeze(), 'b--', zorder=0)
        if self.boundary_H is not None: plt.plot(xf_plot.squeeze(), self.boundary_H(xf_plot).squeeze(), 'r-', zorder=0)

        plt.xlim([0, 1])
        plt.ylim([0, 1])
        prettyplot("Normalized X_1", "Normalized X_2", ylabelpad=-10)

        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename, dpi=150)
            print(f"Plot saved to {filename}")

    def append_next_point(self):
        """Adds the selected next point(s) (always high-fidelity) to the training data."""
        if self.X_next is None:
            print("No next point computed yet to append.")
            return

        # Add to high-fidelity data
        self.train_x_h_norm = torch.cat((self.train_x_h_norm.to(DEVICE), self.X_next.to(DEVICE)), dim=0)

        X_next_denorm_np = denormalize(self.X_next, self.lb, self.ub).cpu().numpy()
        Y_next_np = self.sampling_func(X_next_denorm_np)  # This is the high-fidelity sampling_func
        if not isinstance(Y_next_np, np.ndarray): Y_next_np = np.array([Y_next_np])
        Y_next_np = Y_next_np.astype(float).reshape(-1)

        Y_next_tensor = torch.tensor(Y_next_np, dtype=self.dtype, device=DEVICE)
        self.train_y_h = torch.cat((self.train_y_h.to(DEVICE), Y_next_tensor), dim=0)

        # Update concatenated data
        self.train_x_all = torch.cat((self.train_x_l_norm, self.train_x_h_norm), dim=0)
        self.train_y_all = torch.cat((self.train_y_l, self.train_y_h), dim=0)
        self.num_high_fidelity = self.train_x_h_norm.shape[0]  # Update count

        print(
            f"Appended {self.X_next.shape[0]} HF point(s). Total LF: {self.num_low_fidelity}, Total HF: {self.num_high_fidelity}")
        self.X_next = None  # Clear X_next


# --- GPyTorch NARGP Kernel and Model for Multi-Fidelity Classification ---

import torch
import gpytorch
import numpy as np
from linear_operator.operators import CatLinearOperator  # Correct import for newer GPyTorch/LinearOperator


class NARGPKernel(gpytorch.kernels.Kernel):
    is_stationary = False  # Kernel is not stationary due to fidelity-dependent parts

    def __init__(self, dim, num_low_fidelity_total, initial_rho=1.0, initial_lengthscale_l=0.5,
                 initial_lengthscale_delta=0.5, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self._num_low_fidelity_total = num_low_fidelity_total

        self.cov_L = gpytorch.kernels.RBFKernel(ard_num_dims=dim if dim > 0 else None)
        self.cov_delta = gpytorch.kernels.RBFKernel(ard_num_dims=dim if dim > 0 else None)

        if dim > 0:
            self.cov_L.lengthscale = torch.ones(dim) * initial_lengthscale_l
            self.cov_delta.lengthscale = torch.ones(dim) * initial_lengthscale_delta
        else:
            self.cov_L.lengthscale = initial_lengthscale_l
            self.cov_delta.lengthscale = initial_lengthscale_delta

        self.register_parameter(
            name="raw_rho", parameter=torch.nn.Parameter(torch.tensor(np.log(initial_rho), dtype=torch.float32))
        )
        # Add outputscales as parameters, common for RBF kernels
        self.register_parameter(
            name="raw_outputscale_L", parameter=torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        )
        self.register_parameter(
            name="raw_outputscale_delta", parameter=torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        )
        # Apply constraints if desired (e.g., Positive)
        self.register_constraint("raw_outputscale_L", gpytorch.constraints.Positive())
        self.register_constraint("raw_outputscale_delta", gpytorch.constraints.Positive())
        self.register_constraint("raw_rho", gpytorch.constraints.Positive())  # For rho itself

    @property
    def outputscale_L(self):
        return self.raw_outputscale_L_constraint.transform(self.raw_outputscale_L)

    @outputscale_L.setter
    def outputscale_L(self, value):
        self._set_outputscale_L(value)

    def _set_outputscale_L(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_outputscale_L)
        self.initialize(raw_outputscale_L=self.raw_outputscale_L_constraint.inverse_transform(value))

    @property
    def outputscale_delta(self):
        return self.raw_outputscale_delta_constraint.transform(self.raw_outputscale_delta)

    @outputscale_delta.setter
    def outputscale_delta(self, value):
        self._set_outputscale_delta(value)

    def _set_outputscale_delta(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_outputscale_delta)
        self.initialize(raw_outputscale_delta=self.raw_outputscale_delta_constraint.inverse_transform(value))

    @property
    def num_low_fidelity_total(self):
        return self._num_low_fidelity_total

    @num_low_fidelity_total.setter
    def num_low_fidelity_total(self, value):
        # This might need to be handled carefully if it changes after initialization,
        # especially regarding batch shapes or precomputed values.
        # For now, assume it's set once.
        self._num_low_fidelity_total = value

    @property
    def rho(self):
        # If using log_rho for positivity: return torch.exp(self.raw_rho)
        # If using constraint for positivity:
        return self.raw_rho_constraint.transform(self.raw_rho)

    def forward(self, x1_all, x2_all, diag=False, last_dim_is_batch=False, **params):
        n_l = self.num_low_fidelity_total

        if diag:
            # When diag=True, x1_all and x2_all are effectively the same (or x2_all is None).
            # We only need the diagonal elements.
            # x1_all is the full set of points X = [X_L; X_H]

            # Ensure x1_all and x2_all are treated as the same if x2_all is None
            # GPyTorch handles this: if x2 is None, it internally uses x1 for x2.
            # If x2 is provided and different from x1, diag=True for RBF means element-wise op,
            # which would fail if shapes mismatch for the subtraction.
            # The call from _diagonal() ensures x1 and x2 are the same.

            x_L_slice = x1_all[..., :n_l, :]  # Handles batch dims: (..., n_l, dim)
            x_H_slice = x1_all[..., n_l:, :]  # Handles batch dims: (..., n_h, dim)

            # K_LL diagonal part
            diag_LL = torch.empty(x_L_slice.shape[:-1], device=x1_all.device, dtype=x1_all.dtype)  # (..., n_l) or (n_l)
            if x_L_slice.shape[-2] > 0:  # Check if there are any low-fidelity points
                diag_LL = self.outputscale_L * self.cov_L(x_L_slice, diag=True)

            # K_HH diagonal part
            diag_HH_L_part = torch.empty(x_H_slice.shape[:-1], device=x1_all.device, dtype=x1_all.dtype)
            diag_HH_delta_part = torch.empty(x_H_slice.shape[:-1], device=x1_all.device, dtype=x1_all.dtype)

            if x_H_slice.shape[-2] > 0:  # Check if there are any high-fidelity points
                diag_HH_L_part = self.outputscale_L * (self.rho ** 2) * self.cov_L(x_H_slice, diag=True)
                diag_HH_delta_part = self.outputscale_delta * self.cov_delta(x_H_slice, diag=True)

            # Concatenate along the dimension that represents the points (the one before last, which is -1 after diag=True)
            # diag_LL will be (..., n_l), diag_HH will be (..., n_h)
            # Result should be (..., n_l + n_h)
            return torch.cat([diag_LL, diag_HH_L_part + diag_HH_delta_part], dim=-1)

        # Full kernel matrix case (diag=False)
        # Splitting based on last_dim_is_batch convention
        if last_dim_is_batch:
            # x_all shape: (dim, num_total_points, batch_size) -> not standard for GPs
            # GPyTorch typical batch: (batch_size, num_points, dim)
            # Or for kernels: x1: (b1, n1, d), x2: (b2, n2, d)
            # Let's assume standard GPyTorch batching (..., N, D)
            # where ... are batch dimensions
            x1_L = x1_all[..., :n_l, :]
            x1_H = x1_all[..., n_l:, :]
            x2_L = x2_all[..., :n_l, :]
            x2_H = x2_all[..., n_l:, :]
        else:  # No batch dimension, or batch is first
            x1_L = x1_all[..., :n_l, :]
            x1_H = x1_all[..., n_l:, :]
            x2_L = x2_all[..., :n_l, :]
            x2_H = x2_all[..., n_l:, :]

        # K_LL = cov_L(X_L, X_L)
        # Need to handle empty slices to avoid errors with cov_L or CatLinearOperator
        # Base kernels (like RBFKernel) will return a tensor of shape (..., x1_L.size(-2), x2_L.size(-2))

        # Get base kernel evaluations (without outputscales yet)
        k_L_LL = self.cov_L(x1_L, x2_L) if x1_L.shape[-2] > 0 and x2_L.shape[-2] > 0 else \
            _empty_tensor_for_block(x1_L, x2_L, x1_all)

        k_L_LH = self.cov_L(x1_L, x2_H) if x1_L.shape[-2] > 0 and x2_H.shape[-2] > 0 else \
            _empty_tensor_for_block(x1_L, x2_H, x1_all)

        if torch.equal(x1_all, x2_all):  # Exploit symmetry if x1_all and x2_all are the same
            k_L_HL = k_L_LH.transpose(-1, -2)
        else:
            k_L_HL = self.cov_L(x1_H, x2_L) if x1_H.shape[-2] > 0 and x2_L.shape[-2] > 0 else \
                _empty_tensor_for_block(x1_H, x2_L, x1_all)

        k_L_HH = self.cov_L(x1_H, x2_H) if x1_H.shape[-2] > 0 and x2_H.shape[-2] > 0 else \
            _empty_tensor_for_block(x1_H, x2_H, x1_all, zero_fill=True)  # zero_fill for addition later

        k_delta_HH = self.cov_delta(x1_H, x2_H) if x1_H.shape[-2] > 0 and x2_H.shape[-2] > 0 else \
            _empty_tensor_for_block(x1_H, x2_H, x1_all, zero_fill=True)  # zero_fill for addition

        # Apply scaling factors (rho and outputscales)
        K_LL = self.outputscale_L * k_L_LL
        K_LH = self.outputscale_L * self.rho * k_L_LH
        K_HL = self.outputscale_L * self.rho * k_L_HL
        K_HH = self.outputscale_L * (self.rho ** 2) * k_L_HH + self.outputscale_delta * k_delta_HH

        # Handle cases where some blocks might be empty (e.g., no low-fidelity points)
        # CatLinearOperator can handle this if inputs are correctly shaped empty tensors.

        # Check if any components are effectively zero-sized due to empty inputs
        # and construct the full matrix accordingly.

        has_L = x1_L.shape[-2] > 0 or x2_L.shape[-2] > 0  # Simplified: assume if one side has L, block exists
        has_H = x1_H.shape[-2] > 0 or x2_H.shape[-2] > 0  # Simplified

        if has_L and has_H:
            K_row1 = CatLinearOperator(K_LL, K_LH, dim=-1)  # Concatenate columns
            K_row2 = CatLinearOperator(K_HL, K_HH, dim=-1)  # Concatenate columns
            K_full = CatLinearOperator(K_row1, K_row2, dim=-2)  # Concatenate rows
        elif has_L:  # Only low-fidelity points
            K_full = K_LL
        elif has_H:  # Only high-fidelity points
            K_full = K_HH
        else:  # No points, should ideally not happen or return an empty operator
            raise ValueError

        return K_full

    def cov_L_predict(self, x_new):
        """Covariance of f_L at new points: K_L(x_new, x_new)"""
        return self.outputscale_L * self.cov_L(x_new, x_new)

    def cov_H_predict(self, x_new):
        """Covariance of f_H at new points: rho^2 * K_L(x_new, x_new) + K_delta(x_new, x_new)"""
        k_L_new = self.cov_L(x_new, x_new)
        k_delta_new = self.cov_delta(x_new, x_new)
        return self.outputscale_L * (self.rho ** 2) * k_L_new + self.outputscale_delta * k_delta_new


def _empty_tensor_for_block(x1_block, x2_block, x_full, zero_fill=False):
    """
    Helper to create an appropriately shaped empty or zero tensor for a kernel block.
    Assumes x1_block and x2_block are slices like x_full[..., slice_indices, :].
    Output shape: (..., x1_block.size(-2), x2_block.size(-2))
    """
    s1 = x1_block.shape[-2]
    s2 = x2_block.shape[-2]
    batch_shape = x1_block.shape[:-2]  # Or x_full.shape[:-2]

    if zero_fill:
        pt = torch.zeros(*batch_shape, s1, s2, device=x_full.device, dtype=x_full.dtype)
        return linear_operator.to_linear_operator(pt)
    else:
        pt = torch.empty(*batch_shape, s1, s2, device=x_full.device, dtype=x_full.dtype)
        return linear_operator.to_linear_operator(pt)


class ApproximateGPModelMF(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points_all, dim, num_low_fidelity_total, initial_rho=1.0):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points_all.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points_all, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = NARGPKernel(dim=dim, num_low_fidelity_total=num_low_fidelity_total, initial_rho=initial_rho)

    def forward(self, x_all):
        # x_all is the combined [X_L; X_H]
        # The NARGPKernel's forward method needs num_low_fidelity_total to be set correctly
        # This is usually passed during kernel initialization. If it changes (e.g. more LF points added),
        # the kernel's attribute would need updating.
        # self.covar_module.num_low_fidelity_total = current_num_lf_in_x_all
        mean_x = self.mean_module(x_all)
        covar_x = self.covar_module(x_all, x_all)  # K(X_all, X_all)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def predict_f_H(self, x_new_norm):
        """Predicts the high-fidelity latent function f_H at x_new_norm."""
        # This is a simplified way to get E[f_H* | D] and Var[f_H* | D]
        # For full posterior samples of f_H, we'd typically use the __call__ method
        # on x_new_norm after "training" the combined model on x_all, y_all.
        # The NARGPKernel's structure implies that predictions at x_new are predictions of f_H if we use the
        # K_H_new,new and K_all,new_H parts of the kernel matrix.
        # However, with ApproximateGP, model(x_new_norm) gives q(f*) where f* corresponds
        # to treating x_new_norm as if they were *additional high-fidelity points*.

        # The standard way with ApproximateGP:
        # 1. Train on (train_x_all, train_y_all) to get q(u) and hyperparameters.
        # 2. To predict f_H(x_new):
        #    The model self(x_new_norm) will give the predictive posterior for *new points*
        #    assuming they are of the "type" that the overall kernel structure is designed for
        #    when making predictions at the "end" of the NARGP chain (i.e., high-fidelity).

        # Let's clarify: model(x_test) for an NARGP gives the posterior for f_H(x_test).
        # The kernel K(X_all, x_test) will be structured as [K_L,x_test ; K_H,x_test]
        # and K(x_test,x_test) will be K_H(x_test,x_test).

        # So, simply calling self(x_new_norm) should give the posterior of f_H(x_new_norm).
        return self(x_new_norm)


class MFGPClassifier(BaseMFclassifier):
    def _create_gpytorch_model(self, train_x_all, train_y_all):
        # train_x_all is [X_L; X_H], train_y_all is [Y_L; Y_H]
        # Inducing points should ideally span both low and high-fidelity regions if possible,
        # or be chosen strategically. For simplicity, sample from all training points.
        num_inducing = min(128, train_x_all.size(0))
        if train_x_all.size(0) > 0:
            inducing_points_idx = torch.randperm(train_x_all.size(0), device=train_x_all.device)[:num_inducing]
            inducing_points_all = train_x_all[inducing_points_idx, :]
        else:
            inducing_points_all = torch.empty(0, train_x_all.size(1), device=train_x_all.device)

        likelihood = gpytorch.likelihoods.BernoulliLikelihood().to(DEVICE)
        # The model needs to know how many of the train_x_all points are low-fidelity for the kernel
        model = ApproximateGPModelMF(
            inducing_points_all=inducing_points_all,
            dim=self.dim,
            num_low_fidelity_total=self.num_low_fidelity  # From BaseMFclassifier
        ).to(DEVICE)

        # Update the kernel's num_low_fidelity_total before training if it changed.
        # This should be set based on the train_x_all that is passed.
        model.covar_module.num_low_fidelity_total = self.num_low_fidelity
        return model, likelihood

    def train_model(self, train_x_all, train_y_all, training_iter=150, lr=0.05):  # Adjusted defaults
        """Trains the GPyTorch model."""
        self.model, self.likelihood = self._create_gpytorch_model(train_x_all, train_y_all)
        self.model.to(DEVICE)
        self.likelihood.to(DEVICE)

        # Ensure kernel knows current number of LF points (should be set in _create_gpytorch_model)
        self.model.covar_module.num_low_fidelity_total = self.num_low_fidelity

        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=train_y_all.size(0))

        print("Training MF-GPyTorch model...")
        for i in range(training_iter):
            optimizer.zero_grad()
            output = self.model(train_x_all)
            loss = -mll(output, train_y_all)
            loss.backward()
            optimizer.step()
            if (i + 1) % 20 == 0 or i == 0 or i == training_iter - 1:
                print(f'Iter {i + 1}/{training_iter} - Loss: {loss.item():.3f} ' +
                      f'Rho: {self.model.covar_module.rho.item():.3f} ' +
                      f'LS_L: {self.model.covar_module.cov_L.lengthscale.mean().item():.3f} ' +
                      f'LS_delta: {self.model.covar_module.cov_delta.lengthscale.mean().item():.3f}')
        print("Training complete.")

    def sample_predictive_f(self, X_eval_norm, n_samples=100):
        """
        Samples from the posterior latent function f_H* (high-fidelity) for given X_eval_norm.
        """
        if self.model is None or self.likelihood is None:
            raise RuntimeError("Model not trained yet. Call train_model() first.")

        print("Calling sample_predictive_f")

        self.model.eval()
        self.likelihood.eval()

        X_eval_norm = X_eval_norm.to(DEVICE)

        predictions_list = []
        max_chunk_size = 1024
        num_chunks = (X_eval_norm.shape[0] + max_chunk_size - 1) // max_chunk_size

        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.max_cg_iterations(
                100):  # Might need more CG for MF
            for i in range(num_chunks):
                chunk_start = i * max_chunk_size
                chunk_end = min((i + 1) * max_chunk_size, X_eval_norm.shape[0])
                X_chunk = X_eval_norm[chunk_start:chunk_end]

                # model.predict_f_H(X_chunk) returns the distribution q(f_H*)
                f_h_dist_chunk = self.model.predict_f_H(X_chunk)
                f_samples_chunk = f_h_dist_chunk.sample(torch.Size((n_samples,)))  # Shape: (n_samples, chunk_size)
                predictions_list.append(f_samples_chunk)

        if not predictions_list:
            return torch.tensor([], device=DEVICE)

        f_pred_samples = torch.cat(predictions_list, dim=1)
        return f_pred_samples.cpu()


# --- Example Usage for MFGPClassifier ---
if __name__ == '__main__':
    import os

    os.makedirs("output_plots_mf", exist_ok=True)

    # Define synthetic 2D multi-fidelity classification problem
    lb_ex = np.array([0., 0.])
    ub_ex = np.array([1., 1.])  # Work in normalized space [0,1] for simplicity in definitions


    def true_boundary_L_normalized(x_norm_X1):  # x_norm_X1 is N x 1
        return 0.2 * np.sin(3 * np.pi * x_norm_X1) + 0.6


    def true_boundary_H_normalized(x_norm_X1):  # x_norm_X1 is N x 1
        return 0.15 * np.sin(3 * np.pi * x_norm_X1) + 0.1 * x_norm_X1 + 0.45  # Shifted and slightly different


    def sampling_function_L(X_normalized):  # Expects N x D normalized input
        # X_normalized[:, 0] is (N,)
        # X_normalized[:, 0][:, None] is (N, 1)
        boundary_vals_col = true_boundary_L_normalized(X_normalized[:, 0][:, None])  # boundary_vals_col is (N, 1)

        # X_normalized[:, 1] is (N,)
        # Squeeze boundary_vals_col to (N,) for element-wise comparison
        # (N,) > (N,) results in (N,)
        return (X_normalized[:, 1] > boundary_vals_col.squeeze()).astype(float)
        # Alternatively:
        # return (X_normalized[:, 1] > boundary_vals_col[:, 0]).astype(float)


    def sampling_function_H(X_normalized):  # Expects N x D normalized input
        boundary_vals_col = true_boundary_H_normalized(X_normalized[:, 0][:, None])  # boundary_vals_col is (N, 1)
        return (X_normalized[:, 1] > boundary_vals_col.squeeze()).astype(float)
        # Alternatively:
        # return (X_normalized[:, 1] > boundary_vals_col[:, 0]).astype(float)


    # Initial training data (already normalized for simplicity here)
    N_L_init = 300
    N_H_init = 10
    X_L_init_norm = lhs(2, N_L_init, criterion='maximin', iterations=20)
    Y_L_init_norm = sampling_function_L(X_L_init_norm)
    X_H_init_norm = lhs(2, N_H_init, criterion='maximin', iterations=20)
    Y_H_init_norm = sampling_function_H(X_H_init_norm)

    # Plot initial training data
    plt.figure()
    plt.scatter(X_L_init_norm[:, 0], X_L_init_norm[:, 1], c=Y_L_init_norm, cmap='viridis', marker='s', s=30, alpha=0.5)
    plt.scatter(X_H_init_norm[:, 0], X_H_init_norm[:, 1], c=Y_H_init_norm, cmap='viridis', marker='o', s=60)
    plt.show()

    # Test data (normalized)
    N_test = 200
    X_test_norm_np = lhs(2, N_test)
    # For MF, we usually test against the high-fidelity truth
    Y_test_H_norm_np = sampling_function_H(X_test_norm_np)

    # Create MF classifier instance
    # Since data is already normalized, pass identity to denormalize for sampling_func
    # Or, better, make sampling_func expect normalized inputs directly
    mfgpc = MFGPClassifier(
        X_L_init=X_L_init_norm,  # Pass normalized data
        Y_L_init=Y_L_init_norm,
        X_H_init=X_H_init_norm,
        Y_H_init=Y_H_init_norm,
        lb=lb_ex,  # Bounds are [0,1]
        ub=ub_ex,
        sampling_func=sampling_function_H,  # Active learning queries HF
        X_test=X_test_norm_np,
        Y_test=Y_test_H_norm_np,
        N_cand=500,
        boundary_L=true_boundary_L_normalized,
        boundary_H=true_boundary_H_normalized,
        dtype=torch.float32
    )
    # Override sampling_func to ensure it doesn't denormalize if inputs are already fine
    # The sampling_func in BaseClassifier denormalizes self.X_next before calling.
    # If lb=0, ub=1, denormalize(X,0,1) = X. So current setup is fine.

    # Run active learning
    mfgpc.active_learning(
        N_iterations=5,
        N_points_per_iteration=10,
        plot_active=True,
        filename_template='output_plots_mf/mf_gpc_active_iter_%i.png'
    )

    # Plot final training data
    plt.figure()
    X_L_init_norm = mfgpc.train_x_l_norm
    X_H_init_norm = mfgpc.train_x_h_norm
    Y_L_init_norm = mfgpc.train_y_l
    Y_H_init_norm = mfgpc.train_y_h
    plt.scatter(X_L_init_norm[:, 0], X_L_init_norm[:, 1], c=Y_L_init_norm, cmap='viridis', marker='s', s=30, alpha=0.5)
    plt.scatter(X_H_init_norm[:, 0], X_H_init_norm[:, 1], c=Y_H_init_norm, cmap='viridis', marker='o', s=60)
    plt.show()

    print("\nMF Active learning finished.")
    if mfgpc.error_history:
        print("Test accuracy history (vs HF truth):")
        for i, err_metric in enumerate(mfgpc.error_history):
            print(f"Iter {i + 1}: Accuracy = {err_metric['accuracy']:.2f}%")