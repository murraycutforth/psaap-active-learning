"""Bi-fidelity GPC

 - Based on Costabal et al. paper: [https://www.sciencedirect.com/science/article/pii/S0045782519304785?casa_token=V2f8cieaeisAAAAA:dQXNzisdT1rUaGY0Vq3SggoUp1Lz3CW9jzqBomtFS8kgfSKTMLSgrryPlHQJRTfjqv54JKk6lQ]
 - Posterior distribution of kernel parameters is found using MCMC
 - Auto-regressive model for the latent functions:
    f_H(x) = rho * f_L(x) + delta(x)
    where rho is scalar, and f_i and delta are GPs
 - Implemented using pytorch and gpytorch, but initially hardcoded to CPU

 Author: Murray Cutforth
"""
import copy
import logging

import pyDOE
import torch
import numpy as np
import gpytorch
from matplotlib import pyplot as plt
from linear_operator.operators import DiagLinearOperator, CatLinearOperator, AddedDiagLinearOperator
from pyDOE import lhs

from src.utils_plotting import plot_bfgpc_predictions, plot_bf_training_data
from src.active_learning.util_classes import BiFidelityModel


logger = logging.getLogger(__name__)


class GP_Submodel(gpytorch.models.ApproximateGP):
    def __init__(self, train_x: torch.Tensor):
        """
        Note: The VariationalDistribution object holds the trainable params of the approximate posterior
        """
        var_dist = gpytorch.variational.CholeskyVariationalDistribution(train_x.size(0))
        var_strat = gpytorch.variational.VariationalStrategy(self, train_x, var_dist, learn_inducing_locations=True)
        super().__init__(var_strat)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        """Return the PRIOR distribution. However note that the VariationalStrategy overrides this method so forward()
        actually returns the variational posterior.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def _assemble_T(N_H, N_L, N_f_L_unique, N_f_delta, N_prime, inverse_indices_L, rho_val):
    """T maps G, the distribution (f_L(X_L_unique_eval), delta(X_delta_unique_eval))^T to the target distribution
    This is used for BFGPC_ELBO.predict_multi_fidelity_latent_joint()
    """
    T_shape_rows = N_L + N_H + N_prime
    T_shape_cols = N_f_L_unique + N_f_delta

    # Handle cases where T might be empty or have zero columns/rows
    if T_shape_rows == 0 or T_shape_cols == 0:
        return torch.zeros((T_shape_rows, T_shape_cols))

    T = torch.zeros((T_shape_rows, T_shape_cols))

    # Block 1: N_L rows
    # T[i, col_for_f_L] = 1.0
    for i in range(N_L):
        i_nonunique = inverse_indices_L[i]
        T[i, i_nonunique] = 1.0

    # Block 2: N_H rows
    # T[row_idx, col_for_f_L] = rho_val
    # T[row_idx, col_for_delta] = 1.0
    # Row indices in T for this block: N_L to N_L + N_H - 1
    for j in range(N_H):
        row_idx_in_T = N_L + j  # Corrected row index for T

        # inverse_indices_L maps original data points to unique f_L indices.
        # For X_H points, their f_L components start after X_L points in inverse_indices_L.
        i_nonunique_f_L = inverse_indices_L[N_L + j]

        # The first N_H entries are for X_H.
        i_delta = N_f_L_unique + j

        T[row_idx_in_T, i_nonunique_f_L] = rho_val
        T[row_idx_in_T, i_delta] = 1.0

    # Block 3: N_prime rows
    # T[row_idx, col_for_f_L] = rho_val
    # T[row_idx, col_for_delta] = 1.0
    # Row indices in T for this block: N_L + N_H to N_L + N_H + N_prime - 1
    for j in range(N_prime):
        row_idx_in_T = N_L + N_H + j  # Corrected row index for T

        # For X_prime points, their f_L components start after X_L and X_H points in inverse_indices_L.
        i_nonunique_f_L = inverse_indices_L[N_L + N_H + j]

        # The entries for X_prime in inverse_indices_delta start after X_H entries.
        # These indices ALREADY point to the correct columns in T (i.e., offset by N_f_L_unique).
        i_delta = N_f_L_unique + N_H + j

        T[row_idx_in_T, i_nonunique_f_L] = rho_val
        T[row_idx_in_T, i_delta] = 1.0

    # Check T has no identical rows (which would cause degeneracy)
    assert len(torch.unique(T, dim=0)) == len(T)

    return T


class BFGPC_ELBO(torch.nn.Module, BiFidelityModel):
    def __init__(self, train_x_lf=None, train_x_hf=None, n_inducing_pts=128, initial_rho=1.0, l2_reg_lambda=0.01):
        super().__init__()

        if train_x_lf is None:
           train_x_lf = torch.tensor(pyDOE.lhs(2, n_inducing_pts)).float()

        if train_x_hf is None:
            train_x_hf = torch.tensor(pyDOE.lhs(2, n_inducing_pts)).float()

        self.lf_model = GP_Submodel(train_x_lf)
        self.delta_model = GP_Submodel(train_x_hf)

        # Rho: scaling parameter, learned during training
        self.rho = torch.nn.Parameter(torch.tensor([initial_rho]))

        # L2 regularisation applied to
        self.l2_reg_lambda = l2_reg_lambda

        # Likelihoods for classification (maps latent GP to binary probability)
        self.lf_likelihood = gpytorch.likelihoods.BernoulliLikelihood()
        self.hf_likelihood = gpytorch.likelihoods.BernoulliLikelihood()
        
    def __repr__(self):
        return f"BFGPC_ELBO"

    def _calculate_elbo_terms(self, x_l, y_l, x_h, y_h):
        """
        Helper function to calculate terms needed for the ELBO.
        Returns the approximate posterior distributions for latent functions
        and the KL divergence terms.
        """
        # Get approximate posterior q(f_L(X_L))
        # Note: this call is intercepted by the VariationalStrategy to return the approximate posterior NOT the prior
        q_f_l_at_xl = self.lf_model(x_l)
        # Get approximate posterior q(f_L(X_H)) - needed for f_H
        q_f_l_at_xh = self.lf_model(x_h)
        # Get approximate posterior q(δ(X_H))
        q_f_delta_at_xh = self.delta_model(x_h)

        # Construct approximate posterior for f_H(X_H)
        # E[q(f_H)] = rho * E[q(f_L(X_H))] + E[q(δ(X_H))]
        # Var[q(f_H)] = rho^2 * Var[q(f_L(X_H))] + Var[q(δ(X_H))] (assuming independence of q_f_l and q_f_delta)
        mean_fh = self.rho * q_f_l_at_xh.mean + q_f_delta_at_xh.mean

        # For VariationalStrategy, .covariance_matrix might be expensive.
        # .variance gives diagonal. For Bernoulli, only mean and variance of q(f) are needed by expected_log_prob.
        # However, to be general and if q_f_l_at_xh.covariance_matrix is efficient (e.g. DiagVariationalDistribution),
        # one might use it. For CholeskyVariationalDistribution, it's dense.
        # The BernoulliLikelihood's expected_log_prob uses Gauss-Hermite quadrature which needs mean and variance.
        var_fh = (self.rho.pow(2)) * q_f_l_at_xh.variance + q_f_delta_at_xh.variance
        q_f_h_at_xh = gpytorch.distributions.MultivariateNormal(mean_fh, torch.diag_embed(var_fh))

        # Calculate KL divergences for each sub-model's variational parameters
        # These are KL(q(u_L) || p(u_L)) and KL(q(u_δ) || p(u_δ))
        kl_lf = self.lf_model.variational_strategy.kl_divergence().sum()
        kl_delta = self.delta_model.variational_strategy.kl_divergence().sum()

        return q_f_l_at_xl, q_f_h_at_xh, kl_lf, kl_delta

    def _calculate_loss(self, x_l, y_l, x_h, y_h):
        """
        Calculates the negative Evidence Lower Bound (ELBO) to be minimized.
        ELBO = E_q[log p(y_L|f_L)] + E_q[log p(y_H|f_H)] - KL_L - KL_δ
        """
        q_f_l_at_xl, q_f_h_at_xh, kl_lf, kl_delta = self._calculate_elbo_terms(x_l, y_l, x_h, y_h)

        # Get total number of data points and use this to normalize ELBO
        num_lf_data = y_l.size(0)
        num_hf_data = y_h.size(0)
        num_total_data = num_lf_data + num_hf_data

        # Expected log likelihood terms (using Bernoulli likelihoods)
        # E_q[log p(y_L | f_L(X_L))]
        expected_log_prob_lf = self.lf_likelihood.expected_log_prob(y_l, q_f_l_at_xl).sum()
        # E_q[log p(y_H | f_H(X_H))]
        expected_log_prob_hf = self.hf_likelihood.expected_log_prob(y_h, q_f_h_at_xh).sum()

        avg_expected_log_prob = expected_log_prob_lf + expected_log_prob_hf

        # Sum of KL divergences
        total_kl_divergence = kl_lf + kl_delta

        # ELBO
        neg_elbo = - avg_expected_log_prob + total_kl_divergence

        # Regularization of kernel hyperparams (length_scale and output_scale)
        # Note: one issue with this approach is that the ELBO depends on the
        l2_reg = torch.tensor(0.)
        prior_val = 1.0
        if self.l2_reg_lambda > 0:
            lengthscale_lf = self.lf_model.covar_module.base_kernel.lengthscale
            l2_reg += (lengthscale_lf - prior_val * torch.ones_like(lengthscale_lf)).pow(2).sum()

            outputscale_lf = self.lf_model.covar_module.outputscale
            l2_reg += (outputscale_lf - prior_val * torch.ones_like(outputscale_lf)).pow(2).sum()

            lengthscale_delta = self.delta_model.covar_module.base_kernel.lengthscale
            l2_reg += (lengthscale_delta - prior_val * torch.ones_like(lengthscale_delta)).pow(2).sum()

            outputscale_delta = self.delta_model.covar_module.outputscale
            l2_reg += (outputscale_delta - prior_val * torch.ones_like(outputscale_delta)).pow(2).sum()

            l2_reg *= self.l2_reg_lambda

        # print(neg_elbo, l2_reg)

        return neg_elbo + l2_reg

    def forward(self, x_predict, num_samples=None, return_lf=False):
        """
        Makes predictions for the high-fidelity output at new input points x_predict.
        This method is typically called after training.
        It returns the predictive probability P(y_H=1 | x_predict).
        """
        # --- Explanation of forward() methods ---

        # **In `LF_SubModel` and `Delta_SubModel` (subclasses of `gpytorch.models.ApproximateGP`):**
        #   - `forward(self, x)`:
        #     - This method's primary role is to **define the prior Gaussian Process distribution** over the latent function at the given input points `x`. It specifies the prior mean (e.g., `ConstantMean`, `ZeroMean`) and prior covariance (e.g., `RBFKernel`) of the GP.
        #     - It returns a `gpytorch.distributions.MultivariateNormal` object representing this prior distribution: `p(latent_function(x))`.
        #     - Crucially, when you *call* an instance of these sub-models like `lf_model_instance(x)`, GPyTorch's `VariationalStrategy` intercepts this call. It uses the prior defined by `forward(x)` along with its learned variational parameters (which approximate the posterior over inducing point function values `q(u)`) to compute and return the **approximate posterior distribution `q(latent_function(x))`** over the latent function values at `x`.

        # **In `BiFidelityGPClassification` (subclass of `torch.nn.Module`):**
        #   - `forward(self, x_predict)`:
        #     - This method is designed for **making predictions on new, unseen data** `x_predict` *after* the model has been trained.
        #     - It takes input points `x_predict` and aims to predict the high-fidelity class probabilities.
        #     - Inside, it calls the trained `lf_model` and `delta_model` to get their respective approximate posterior latent distributions (`q(f_L(x_predict))` and `q(δ(x_predict))`).
        #     - It then combines these (using the learned `rho`) to construct the approximate posterior latent distribution for the high-fidelity function: `q(f_H(x_predict))`.
        #     - Finally, it passes `q(f_H(x_predict))` through the `hf_likelihood` (a `BernoulliLikelihood`) to obtain the predictive probabilities for the positive class.
        #     - Note: For training, the `calculate_loss` method is used, which internally calls the sub-models on the training data to compute the ELBO.

        self.eval()

        if not torch.is_tensor(x_predict):
            x_predict = torch.tensor(x_predict).float()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            q_f_l_at_xpredict = self.lf_model(x_predict)
            q_f_delta_at_xpredict = self.delta_model(x_predict)

            mean_fh_predict = self.rho * q_f_l_at_xpredict.mean + q_f_delta_at_xpredict.mean
            var_fh_predict = (self.rho.pow(2)) * q_f_l_at_xpredict.variance + q_f_delta_at_xpredict.variance

            q_f_h_predict = gpytorch.distributions.MultivariateNormal(mean_fh_predict,
                                                                      torch.diag_embed(var_fh_predict))

        if return_lf:
            if num_samples is not None:
                results = {"hf_samples": self.hf_likelihood(q_f_h_predict.sample(torch.Size([num_samples]))),
                           "lf_samples": self.lf_likelihood(q_f_l_at_xpredict.sample(torch.Size([num_samples]))),
                           "hf_mean": self.hf_likelihood(q_f_h_predict).mean,
                           "lf_mean": self.lf_likelihood(q_f_l_at_xpredict).mean}
                return results
            else:
                results = {"hf_mean": self.hf_likelihood(q_f_h_predict).mean,
                           "lf_mean": self.lf_likelihood(q_f_l_at_xpredict).mean}
                return results
        else:
            if num_samples is not None:
                results = {"hf_samples": self.hf_likelihood(q_f_h_predict.sample(torch.Size([num_samples]))),
                           "hf_mean": self.hf_likelihood(q_f_h_predict).mean}
                return results
            else:
                return self.hf_likelihood(q_f_h_predict).mean

    def predict_hf_prob(self, x_predict):
        if not torch.is_tensor(x_predict):
            x_predict = torch.tensor(x_predict).float()

        return self.forward(x_predict).detach().numpy()

    def predict_lf_prob(self, x_predict):
        """
        Predicts P(Y_L=1 | x_eval)
        x_eval: points at which to evaluate the LF prediction
        """
        if not torch.is_tensor(x_predict):
            x_predict = torch.tensor(x_predict).float()

        q_f_l_at_xpredict = self.lf_model(x_predict)
        lf_pred_probs = self.lf_likelihood(q_f_l_at_xpredict).mean
        return lf_pred_probs.detach().numpy()

    def predict_hf_prob_var(self, x_predict):
        self.eval()

        if not torch.is_tensor(x_predict):
            x_predict = torch.tensor(x_predict).float()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            q_f_l_at_xpredict = self.lf_model(x_predict)
            q_f_delta_at_xpredict = self.delta_model(x_predict)

            mean_fh_predict = self.rho * q_f_l_at_xpredict.mean + q_f_delta_at_xpredict.mean
            var_fh_predict = (self.rho.pow(2)) * q_f_l_at_xpredict.variance + q_f_delta_at_xpredict.variance

            q_f_h_predict = gpytorch.distributions.MultivariateNormal(mean_fh_predict,
                                                                      torch.diag_embed(var_fh_predict))

            n_samples = 10
            q_samples = q_f_h_predict.sample(torch.Size((n_samples,)))  # Shape (n_samples, len(x_predict))
            output_probs = gpytorch.distributions.base_distributions.Normal(0, 1).cdf(q_samples)

        return output_probs.var(dim=0).detach().numpy()


    def predict_lf(self, x_predict, num_samples=None):
        """
        Predicts P(Y_L=1 | x_eval)
        x_eval: points at which to evaluate the LF prediction
        """
        if num_samples is None:
            q_f_l_at_xpredict = self.lf_model(x_predict)
            return self.lf_likelihood(q_f_l_at_xpredict).mean
        else:
            q_f_l_at_xpredict = self.lf_model(x_predict).sample(torch.Size([num_samples]))
            return self.lf_likelihood(q_f_l_at_xpredict)


    def predict_f_H(self, x_predict):
        """
        Predicts multivariate normal distribution f_H, the latent function for the high-fidelity output.
        x_eval: points at which to evaluate the LF prediction
        """
        if not torch.is_tensor(x_predict):
            x_predict = torch.tensor(x_predict).float()

        self.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            q_f_l_at_xpredict = self.lf_model(x_predict)
            q_f_delta_at_xpredict = self.delta_model(x_predict)

            mean_fh_predict = self.rho * q_f_l_at_xpredict.mean + q_f_delta_at_xpredict.mean
            var_fh_predict = (self.rho.pow(2)) * q_f_l_at_xpredict.variance + q_f_delta_at_xpredict.variance

            q_f_h_predict = gpytorch.distributions.MultivariateNormal(mean_fh_predict,
                                                                      torch.diag_embed(var_fh_predict))

        return q_f_h_predict

    def predict_f_L(self, x_predict):
        if not torch.is_tensor(x_predict):
            x_predict = torch.tensor(x_predict).float()
        self.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            q_f_l_at_xpredict = self.lf_model(x_predict)
        return q_f_l_at_xpredict

    def predict_delta(self, x_predict):
        if not torch.is_tensor(x_predict):
            x_predict = torch.tensor(x_predict).float()
        self.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            q_delta = self.delta_model(x_predict)
        return q_delta

    def predict_multi_fidelity_latent_joint(self, X_L: torch.tensor, X_H: torch.tensor, X_prime: torch.tensor, extra_assertions: bool = False):
        """
        Predicts joint distribution of latent f_L at X_L and f_H at X_H and X_prime.
        Used to compute mutual information between multifidelity proposal set and HF latent.

        Note: this is quite subtle, but if there is overlap between X_H and X_prime then
        the current method which looks at unique subsets of these points will result in a matrix T
        which has identical (linearly dependent) rows, and as a result the final covariance matrix
        is not positive definite.

        :param X_L: low fidelity proposal locations
        :param X_H: high fidelity proposal locations
        :param X_prime: high fidelity MC locations
        :return: gpytorch.distributions.MultivariateNormal
        """
        assert X_L.dtype == torch.float32
        assert X_H.dtype == torch.float32
        assert X_prime.dtype == torch.float32

        if extra_assertions:
            # There cannot be identical points in X_H and X_prime otherwise the resulting distribution will be degenerate
            X_H_and_prime = torch.cat((X_H, X_prime), dim=0)
            uniqueness_test = torch.unique(X_H_and_prime, dim=0)
            assert uniqueness_test.shape[0] == X_H_and_prime.shape[0], f"{X_H_and_prime.shape[0]} != {uniqueness_test.shape[0]}, X_H shape = {X_H.shape}, X_prime shape = {X_prime.shape}"

        self.eval()

        N_L = X_L.shape[0]
        N_H = X_H.shape[0]
        N_prime = X_prime.shape[0]
        rho_val = self.rho.item()

        assert len(X_prime) > 0
        d = X_prime.shape[1]

        if len(X_L) == 0:
            X_L = torch.empty((0, d), dtype=torch.float)

        if len(X_H) == 0:
            X_H = torch.empty((0, d), dtype=torch.float)

        assert len(X_L.shape) == 2
        assert len(X_H.shape) == 2
        assert len(X_prime.shape) == 2
        assert X_L.shape[1] == X_H.shape[1] == X_prime.shape[1] == d

        with torch.no_grad(), gpytorch.settings.fast_pred_var():

            all_L_points = torch.cat([X_L, X_H, X_prime], dim=0)
            X_L_unique_eval, inverse_indices_L = torch.unique(all_L_points, dim=0, return_inverse=True)
            N_f_L_unique = X_L_unique_eval.shape[0]
            assert len(inverse_indices_L) == N_L + N_H + N_prime
            # inverse_indices_L now maps each point in all_L_points back to its row in X_L_unique_eval

            all_delta_points = torch.cat([X_H, X_prime], dim=0)
            N_f_delta = all_delta_points.shape[0]

            # The linear transformation T maps the distribution G to the target distribution
            T = _assemble_T(N_H, N_L, N_f_L_unique, N_f_delta, N_prime, inverse_indices_L, rho_val)

            # These are the approximate predictive posteriors for f_L and delta
            q_f_l_at_xpredict = self.lf_model(X_L_unique_eval)
            q_f_delta_at_xpredict = self.delta_model(all_delta_points)

            # Parameters of Gaussian joint distribution G (f_L and delta are independent GPs)
            mu_G = torch.cat([q_f_l_at_xpredict.mean, q_f_delta_at_xpredict.mean], dim=0)
            sigma_G = torch.block_diag(q_f_l_at_xpredict.covariance_matrix, q_f_delta_at_xpredict.covariance_matrix)

            f_target_mu = T @ mu_G
            K_intermediate = T @ sigma_G @ T.T
            K_intermediate = 0.5 * (K_intermediate + K_intermediate.T)  # Ensure K is exactly symmetric
            f_target_sigma = K_intermediate + 1e-6 * torch.eye(K_intermediate.shape[0])

            return gpytorch.distributions.MultivariateNormal(f_target_mu, f_target_sigma,
                                                             validate_args=extra_assertions)

    def predict_multi_fidelity_latent_joint_lazy(
            self,
            X_L: torch.tensor,
            X_H: torch.tensor,
            X_prime: torch.tensor,
            extra_assertions: bool = False
    ):
        """
        Predicts joint distribution of latent f_L at X_L and f_H at X_H and X_prime
        using GPyTorch's LazyTensor (LinearOperator) framework.

        This avoids forming large, dense covariance matrices, making it highly efficient
        for large X_prime (e.g., a dense grid).

        :param X_L: Low fidelity proposal locations.
        :param X_H: High fidelity proposal locations.
        :param X_prime: High fidelity Monte Carlo locations.
        :return: gpytorch.distributions.MultivariateNormal with a lazy covariance.
        """
        # --- 1. Input Validation and Preparation ---
        assert X_L.dtype == torch.float32 and X_H.dtype == torch.float32 and X_prime.dtype == torch.float32
        if extra_assertions:
            # This check is still valid and important for preventing singular matrices.
            X_H_and_prime = torch.cat((X_H, X_prime), dim=0)
            uniqueness_test = torch.unique(X_H_and_prime, dim=0)
            assert uniqueness_test.shape[0] == X_H_and_prime.shape[0], \
                f"There are duplicate points between X_H and X_prime, which is not allowed."

        self.eval()

        N_L, N_H, N_prime = X_L.shape[0], X_H.shape[0], X_prime.shape[0]
        rho = self.rho.item()
        d = X_prime.shape[1]

        # Handle empty inputs gracefully
        if N_L == 0: X_L = torch.empty((0, d), dtype=X_prime.dtype, device=X_prime.device)
        if N_H == 0: X_H = torch.empty((0, d), dtype=X_prime.dtype, device=X_prime.device)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # --- 2. Get Base Predictive Distributions (as Lazy objects) ---
            # We query the models ONCE with all the points needed for f_L and f_delta.
            # This allows GPyTorch to efficiently compute all required cross-covariances.
            all_L_points = torch.cat([X_L, X_H, X_prime], dim=0)
            all_delta_points = torch.cat([X_H, X_prime], dim=0)

            # These calls return distributions with LAZY covariance matrices
            pred_L_dist = self.lf_model(all_L_points)
            pred_delta_dist = self.delta_model(all_delta_points)

            # --- 3. Construct the Joint Mean Vector ---
            # This is simple as f_H(x) = rho * f_L(x) + delta(x)
            mu_L_all = pred_L_dist.mean
            mu_delta_all = pred_delta_dist.mean

            # Mean of f_L(X_L)
            mean_fL_at_XL = mu_L_all[:N_L]
            # Mean of f_H(X_H) = rho * f_L(X_H) + delta(X_H)
            mean_fH_at_XH = rho * mu_L_all[N_L:N_L + N_H] + mu_delta_all[:N_H]
            # Mean of f_H(X_prime) = rho * f_L(X_prime) + delta(X_prime)
            mean_fH_at_Xprime = rho * mu_L_all[N_L + N_H:] + mu_delta_all[N_H:]

            f_target_mu = torch.cat([mean_fL_at_XL, mean_fH_at_XH, mean_fH_at_Xprime], dim=0)

            # --- 4. Construct the Joint Covariance Matrix (as a Lazy Block Matrix) ---
            # The target covariance is Cov([f_L(X_L), f_H(X_H), f_H(X_prime)]).
            # We build this as a 3x3 block matrix using the lazy covariance operators.
            K_L_lazy = pred_L_dist.lazy_covariance_matrix
            K_delta_lazy = pred_delta_dist.lazy_covariance_matrix

            # Define slices for clarity
            sl_L = slice(None, N_L)
            sl_H = slice(N_L, N_L + N_H)
            sl_p = slice(N_L + N_H, None)

            sl_delta_H = slice(None, N_H)
            sl_delta_p = slice(N_H, None)

            # Block (0,0): Cov(f_L(X_L), f_L(X_L))
            K00 = K_L_lazy[sl_L, sl_L]
            # Block (0,1): Cov(f_L(X_L), f_H(X_H)) = rho * K_L(X_L, X_H)
            K01 = K_L_lazy[sl_L, sl_H].mul(rho)
            # Block (0,2): Cov(f_L(X_L), f_H(X_prime)) = rho * K_L(X_L, X_prime)
            K02 = K_L_lazy[sl_L, sl_p].mul(rho)

            # Block (1,1): Cov(f_H(X_H), f_H(X_H)) = rho^2*K_L(X_H,X_H) + K_delta(X_H,X_H)
            K11 = K_L_lazy[sl_H, sl_H].mul(rho ** 2).add(K_delta_lazy[sl_delta_H, sl_delta_H])
            # Block (1,2): Cov(f_H(X_H), f_H(X_prime)) = rho^2*K_L(X_H,X_prime) + K_delta(X_H,X_prime)
            K12 = K_L_lazy[sl_H, sl_p].mul(rho ** 2).add(K_delta_lazy[sl_delta_H, sl_delta_p])

            # Block (2,2): Cov(f_H(X_prime), f_H(X_prime)) = rho^2*K_L(X_prime,X_prime) + K_delta(X_prime,X_prime)
            K22 = K_L_lazy[sl_p, sl_p].mul(rho ** 2).add(K_delta_lazy[sl_delta_p, sl_delta_p])

            # Assemble the blocks into a single lazy operator.
            f_target_cov_lazy = CatLinearOperator(
                    CatLinearOperator(K00, K01, K02, dim=-1),
                    CatLinearOperator(K01.transpose(-1, -2), K11, K12, dim=-1),
                    CatLinearOperator(K02.transpose(-1, -2), K12.transpose(-1, -2), K22, dim=-1),
                dim=(-2)
            )

            # Add a small diagonal jitter for numerical stability (the lazy way)
            total_size = N_L + N_H + N_prime
            jitter_diag = DiagLinearOperator(torch.full((total_size,), 1e-6, device=f_target_cov_lazy.device))
            f_target_sigma_lazy = AddedDiagLinearOperator(f_target_cov_lazy, jitter_diag)

            return gpytorch.distributions.MultivariateNormal(
                f_target_mu,
                f_target_sigma_lazy,
                validate_args=extra_assertions
            )

    def _reinitialize_parameters(self):
        """
        Helper method to re-initialize model parameters for a new training run.
        This ensures each of the `n_inits` runs starts from a different random state.
        """
        # --- Manually reset the variational parameters to match the N(0, I) prior ---

        # CORRECTED ACCESS PATH: Use the private `_variational_distribution` to get the module.
        lf_dist_module = self.lf_model.variational_strategy._variational_distribution
        # Set mean to 0
        lf_dist_module.variational_mean.data.zero_()
        # Set covariance to Identity by setting its Cholesky factor to Identity
        n_inducing_lf = lf_dist_module.variational_mean.shape[0]
        identity_lf = torch.eye(n_inducing_lf, device=lf_dist_module.chol_variational_covar.device)
        lf_dist_module.chol_variational_covar.data.copy_(identity_lf)

        # Do the same for the Delta model
        delta_dist_module = self.delta_model.variational_strategy._variational_distribution
        # Set mean to 0
        delta_dist_module.variational_mean.data.zero_()
        # Set covariance to Identity
        n_inducing_delta = delta_dist_module.variational_mean.shape[0]
        identity_delta = torch.eye(n_inducing_delta, device=delta_dist_module.chol_variational_covar.device)
        delta_dist_module.chol_variational_covar.data.copy_(identity_delta)

        # Reset kernel hyperparameters and rho by re-initializing their raw, untransformed values.
        # We sample from a standard normal distribution, which is a common practice.
        with torch.no_grad():
            self.lf_model.covar_module.base_kernel.raw_lengthscale.normal_()
            self.lf_model.covar_module.raw_outputscale.normal_()
            self.delta_model.covar_module.base_kernel.raw_lengthscale.normal_()
            self.delta_model.covar_module.raw_outputscale.normal_()
            # Reset rho to a random value. Since it's passed through a sigmoid,
            # a raw value sampled from N(0, 0.5^2) will be centered around 0.5 post-transform.
            self.rho.normal_(mean=0.0, std=0.5)

    def train_model(self, X_LF, Y_LF, X_HF, Y_HF, lr=0.01, n_epochs=1000, n_inits=3, verbose=False):
        """Also confusingly referred to as the inference step in the GP literature.

        Here we maximise the ELBO, optimising both the variational parameters and the kernel hyperparameters.
        To improve robustness, this method performs `n_inits` separate training runs from different
        random initializations and selects the model with the best final ELBO.
        """
        if n_inits < 1:
            raise ValueError("Number of initializations (n_inits) must be at least 1.")

        if not torch.is_tensor(X_LF): X_LF = torch.tensor(X_LF).float()
        if not torch.is_tensor(Y_LF): Y_LF = torch.tensor(Y_LF).float()
        if not torch.is_tensor(X_HF): X_HF = torch.tensor(X_HF).float()
        if not torch.is_tensor(Y_HF): Y_HF = torch.tensor(Y_HF).float()

        best_loss = float('inf')
        best_state_dict = None

        logger.debug(f"Starting training with {n_inits} random initialization(s).")

        for i in range(n_inits):
            logger.debug(f"--- Running initialization {i + 1}/{n_inits} ---")

            # Re-initialize model parameters for a fresh start
            self._reinitialize_parameters()

            # Create a new optimizer for each run to reset its state
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
            self.train()

            # The actual training loop for one initialization
            for epoch in range(n_epochs):
                optimizer.zero_grad()
                loss = self._calculate_loss(X_LF, Y_LF, X_HF, Y_HF)
                loss.backward()
                optimizer.step()
                if verbose and (epoch + 1) % (n_epochs // 10) == 0:
                    logger.debug(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item():.4f}")
                    # Optional: Add back detailed logging if needed under the verbose flag
                    # logger.debug(f"  Rho: {self.rho.item():.4f}") ... etc

            # After training, evaluate the final loss for this initialization
            # We must set model to eval mode to get a deterministic loss value
            self.eval()
            with torch.no_grad():
                final_loss = self._calculate_loss(X_LF, Y_LF, X_HF, Y_HF).item()
            logger.debug(f"Initialization {i + 1} finished with final loss: {final_loss:.4f}")

            # If this run is the best so far, save its parameters
            if final_loss < best_loss:
                best_loss = final_loss
                # Use deepcopy to prevent pointers to the same memory
                best_state_dict = copy.deepcopy(self.state_dict())
                logger.debug(f"  ** New best model found (Loss: {best_loss:.4f}) **")

        # After all initializations, load the best parameters back into the model
        if best_state_dict is not None:
            logger.debug(f"Finished all initializations. Loading best model with loss: {best_loss:.4f}")
            self.load_state_dict(best_state_dict)
        else:
            # This case should not be reached if n_inits >= 1
            logger.warning("Training finished, but no best model state was found.")

        logger.debug("Training finished.")

    def evaluate_elpp(self, X_HF_test: np.ndarray, Y_HF_test: np.ndarray):
        """The expected log predictive probability is a standard metric in the VI literature.
        This metric is suitable for probability estimation, as we have in the PSAAP problem.
        """
        pred_probs_p1 = self.forward(X_HF_test)

        # Ensure Y_HF_test is a torch tensor for calculations
        if not torch.is_tensor(Y_HF_test):
            Y_HF_test = torch.tensor(Y_HF_test).float()

        # 2. Calculate the log probability of the TRUE class for each sample.
        #    This is the log-likelihood of the Bernoulli distribution.
        #    - If Y_HF_test is 1, this becomes: 1 * log(p1) + 0 * log(1-p1) = log(p1)
        #    - If Y_HF_test is 0, this becomes: 0 * log(p1) + 1 * log(1-p1) = log(1-p1)
        #    We add a small epsilon for numerical stability to avoid log(0).
        epsilon = 1e-8
        log_probs_of_true_class = (
                Y_HF_test * torch.log(pred_probs_p1 + epsilon) +
                (1 - Y_HF_test) * torch.log(1 - pred_probs_p1 + epsilon)
        )

        # 3. The ELPP is the average of these log probabilities.
        elpp = torch.mean(log_probs_of_true_class)

        return elpp.item()


    def evaluate_accuracy(self, X_HF_test, Y_HF_test):
        """If we have a classification problem (rather than probability estimation).
        """
        self.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_probs = self(torch.tensor(X_HF_test, dtype=torch.float))
            test_labels = (test_probs > 0.5).float()
        accuracy = (test_labels == torch.tensor(Y_HF_test, dtype=torch.float32)).float().mean().item()

        return accuracy



# --- Example Usage for MFGPClassifier ---
if __name__ == '__main__':
    import os
    import numpy as np

    os.makedirs("../output_plots_bfgpc", exist_ok=True)

    # Define synthetic 2D multi-fidelity classification problem
    lb_ex = np.array([0., 0.])
    ub_ex = np.array([1., 1.])  # Work in normalized space [0,1] for simplicity in definitions


    def true_boundary_L_normalized(x_norm_X1):  # x_norm_X1 is N x 1
        return 0.2 * np.sin(3 * np.pi * x_norm_X1) + 0.6


    def true_boundary_H_normalized(x_norm_X1):  # x_norm_X1 is N x 1
        return 0.15 * np.sin(3 * np.pi * x_norm_X1) + 0.1 * x_norm_X1 + 0.45


    def sampling_function_L(X_normalized):  # Expects N x D normalized input
        boundary_vals_col = true_boundary_L_normalized(X_normalized[:, 0][:, None])
        return (X_normalized[:, 1] > boundary_vals_col.squeeze()).astype(float)


    def sampling_function_H(X_normalized):  # Expects N x D normalized input
        boundary_vals_col = true_boundary_H_normalized(X_normalized[:, 0][:, None])
        return (X_normalized[:, 1] > boundary_vals_col.squeeze()).astype(float)


    # Initial training data (already normalized for simplicity here)
    N_L_init = 200
    N_H_init = 100  # Increased N_H_init slightly for better delta model learning
    X_L_init_norm_np = lhs(2, N_L_init, criterion='maximin', iterations=20)
    Y_L_init_norm_np = sampling_function_L(X_L_init_norm_np)
    X_H_init_norm_np = lhs(2, N_H_init, criterion='maximin', iterations=20)
    Y_H_init_norm_np = sampling_function_H(X_H_init_norm_np)

    plot_bf_training_data(X_L_init_norm_np, Y_L_init_norm_np, X_H_init_norm_np, Y_H_init_norm_np, boundary_LF=true_boundary_L_normalized, boundary_HF=true_boundary_H_normalized)

    X_L_train = torch.tensor(X_L_init_norm_np, dtype=torch.float32)
    Y_L_train = torch.tensor(Y_L_init_norm_np, dtype=torch.float32)
    X_H_train = torch.tensor(X_H_init_norm_np, dtype=torch.float32)
    Y_H_train = torch.tensor(Y_H_init_norm_np, dtype=torch.float32)

    model = BFGPC_ELBO(X_L_train, X_H_train, initial_rho=1.0)
    model.train_model(X_L_train, Y_L_train, X_H_train, Y_H_train, lr=0.01, n_epochs=1000)
    plot_bfgpc_predictions(model, X_LF=X_L_init_norm_np, Y_LF=Y_L_init_norm_np, X_HF=X_H_init_norm_np, Y_HF=Y_H_init_norm_np, boundary_HF=true_boundary_H_normalized)

    X_test_norm_np = lhs(2, 200, criterion='maximin', iterations=20)
    Y_test_H_norm_np = sampling_function_H(X_test_norm_np)
    acc = model.evaluate_accuracy(X_test_norm_np, Y_test_H_norm_np)

