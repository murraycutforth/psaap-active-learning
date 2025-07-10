"""Bi-fidelity GPC

 - Based on Costabal et al. paper: [https://www.sciencedirect.com/science/article/pii/S0045782519304785?casa_token=V2f8cieaeisAAAAA:dQXNzisdT1rUaGY0Vq3SggoUp1Lz3CW9jzqBomtFS8kgfSKTMLSgrryPlHQJRTfjqv54JKk6lQ]
 - Posterior distribution of kernel parameters is found using MCMC
 - Auto-regressive model for the latent functions:
    f_H(x) = rho * f_L(x) + delta(x)
    where rho is scalar, and f_i and delta are GPs
 - Implemented using pytorch and gpytorch, but initially hardcoded to CPU

 Author: Murray Cutforth
"""
import logging

import pyDOE
import torch
import numpy as np
import gpytorch
from matplotlib import pyplot as plt
from pyDOE import lhs

from src.utils_plotting import plot_bfgpc_predictions, plot_bf_training_data
from src.active_learning.util_classes import BiFidelityModel

import pdb

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
    def __init__(self, train_x_lf=None, train_x_hf=None, initial_rho=1.0):
        super().__init__()

        if train_x_lf is None:
           train_x_lf = torch.tensor(pyDOE.lhs(2, 128, criterion='maximin', iterations=10)).float()

        if train_x_hf is None:
            train_x_hf = torch.tensor(pyDOE.lhs(2, 128, criterion='maximin', iterations=10)).float()

        self.lf_model = GP_Submodel(train_x_lf)
        self.delta_model = GP_Submodel(train_x_hf)

        # Rho: scaling parameter, learned during training
        self.rho = torch.nn.Parameter(torch.tensor([initial_rho]))

        # Likelihoods for classification (maps latent GP to binary probability)
        self.lf_likelihood = gpytorch.likelihoods.BernoulliLikelihood()
        self.hf_likelihood = gpytorch.likelihoods.BernoulliLikelihood()

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

        # Expected log likelihood terms (using Bernoulli likelihoods)
        # E_q[log p(y_L | f_L(X_L))]
        expected_log_prob_lf = self.lf_likelihood.expected_log_prob(y_l, q_f_l_at_xl).sum()
        # E_q[log p(y_H | f_H(X_H))]
        expected_log_prob_hf = self.hf_likelihood.expected_log_prob(y_h, q_f_h_at_xh).sum()

        # Sum of KL divergences
        total_kl_divergence = kl_lf + kl_delta

        # ELBO
        elbo = expected_log_prob_lf + expected_log_prob_hf - total_kl_divergence
        return -elbo  # Return negative ELBO for minimization

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

            # Get predictive probabilities from the HF likelihood
            # hf_likelihood(q_f_h_predict) returns a torch.distributions.Bernoulli
            # .mean of Bernoulli is the probability of class 1
            # predictive_probs_hf = self.hf_likelihood(q_f_h_predict).mean

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

    def predict_prob_mean(self, x_predict):
        return self.forward(x_predict).detach().numpy()

    def predict_prob_var(self, x_predict):
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

    def predict_multi_fidelity_latent_joint(self, X_L: torch.tensor, X_H: torch.tensor, X_prime: torch.tensor):
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

            #eigenvals_intermediate = torch.linalg.eigvalsh(K_intermediate)
            #eigenvals_final = torch.linalg.eigvalsh(f_target_sigma)

            #if torch.min(eigenvals_final) <= 0:
            #    logger.error(f"Min eigenval intermediate: {torch.min(eigenvals_intermediate)}")
            #    logger.error(f"Min eigenval final: {torch.min(eigenvals_final)}")

            try:
                return gpytorch.distributions.MultivariateNormal(f_target_mu, f_target_sigma, validate_args=True)
            except ValueError:
                logger.critical("Non positive definite covariance matrix (probably)")
                plt.imshow(f_target_sigma)
                plt.colorbar()
                plt.show()
                f_target_sigma += 1e-3 * torch.eye(K_intermediate.shape[0])
                return gpytorch.distributions.MultivariateNormal(f_target_mu, f_target_sigma, validate_args=True)

    def train_model(self, X_LF, Y_LF, X_HF, Y_HF, lr=0.01, n_epochs=1000):
        """Also confusingly referred to as the inference step in the GP literature.

        Here we maximise the ELBO, optimising both the variational parameters and the kernel hyperparameters.
        The kernel hyperparameters are treated as point estimates (no prior), while the variational parameters u_i
        which are located at the inducing points specified in the submodels for f_L and delta have GP priors and
        approximate posteriors. This approach is known as empirical Bayes.
        """
        if not torch.is_tensor(X_LF):
            X_LF = torch.tensor(X_LF).float()
        if not torch.is_tensor(Y_LF):
            Y_LF = torch.tensor(Y_LF).float()
        if not torch.is_tensor(X_HF):
            X_HF = torch.tensor(X_HF).float()
        if not torch.is_tensor(Y_HF):
            Y_HF = torch.tensor(Y_HF).float()

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.train()

        print("Starting training...")
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            loss = self._calculate_loss(X_LF, Y_LF, X_HF, Y_HF)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % (n_epochs // 10) == 0:
                print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item():.4f}")
                print(f"  Rho: {self.rho.item():.4f}")
                print(f"  LF model lengthscale: {self.lf_model.covar_module.base_kernel.lengthscale.item():.4f}, "
                      f"outputscale: {self.lf_model.covar_module.outputscale.item():.4f}")
                print(f"  Delta model lengthscale: {self.delta_model.covar_module.base_kernel.lengthscale.item():.4f}, "
                      f"outputscale: {self.delta_model.covar_module.outputscale.item():.4f}")

        print("Training finished.")

    def evaluate_elpp(self, X_HF_test: np.ndarray, Y_HF_test: np.ndarray):
        """The expected log predictive probability is a standard metric in the VI literature.
        This metric is suitable for probability estimation, as we have in the PSAAP problem.
        """

        self.eval()
        X_HF_test = torch.tensor(X_HF_test).float()
        Y_HF_test = torch.tensor(Y_HF_test).float()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            q_f_l_at_xpredict = self.lf_model(X_HF_test)
            q_f_delta_at_xpredict = self.delta_model(X_HF_test)

            mean_fh_predict = self.rho * q_f_l_at_xpredict.mean + q_f_delta_at_xpredict.mean
            var_fh_predict = (self.rho.pow(2)) * q_f_l_at_xpredict.variance + q_f_delta_at_xpredict.variance

            q_f_h_predict = gpytorch.distributions.MultivariateNormal(mean_fh_predict,
                                                                      torch.diag_embed(var_fh_predict))

            # This is the E_q[log p(Y_HF_test | f_H(X_HF_test))]
            test_elp = self.hf_likelihood.expected_log_prob(Y_HF_test, q_f_h_predict)
            sum_test_elp = test_elp.sum().numpy().item()

        return sum_test_elp

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

    os.makedirs("output_plots_bfgpc", exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)

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

