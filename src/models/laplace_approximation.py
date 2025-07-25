import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, binom

# The probit function is the CDF of the standard normal distribution
def probit(x):
    return torch.distributions.Normal(0, 1).cdf(x)

def laplace_approximation_probit(mu_prior: float,
                                 sigma_prior: float,
                                 N: int,
                                 k: int) -> tuple[float, float]:
    """
    Computes the Laplace approximation for a Probit-Bernoulli likelihood and a Gaussian prior.

    This function finds the posterior p(f|Y) where the prior is p(f) = N(mu_prior, sigma_prior^2)
    and the likelihood is p(Y|f) = Probit(f)^k * (1-Probit(f))^(N-k).

    Args:
        mu_prior (float): The mean of the Gaussian prior on f.
        sigma_prior (float): The standard deviation of the Gaussian prior on f.
        N (int): The total number of new observations at x*.
        k (int): The number of successes (y=1) among the N observations.

    Returns:
        tuple[float, float]: A tuple containing:
            - mu_posterior (float): The mean of the approximate Gaussian posterior (the MAP estimate).
            - sigma_posterior (float): The standard deviation of the approximate Gaussian posterior.
    """
    if sigma_prior <= 0:
        raise ValueError("Prior standard deviation must be positive.")
    if N < 0:
        raise ValueError("N cannot be negative.")
    if N == 0:
        return mu_prior, sigma_prior

    # Use a small epsilon to prevent log(0) issues if k=0 or k=N
    eps = 1e-9

    # 1. Define the negative log posterior function
    # We want to MINIMIZE this function to find the MAP estimate.
    # log p(f|Y) = log p(Y|f) + log p(f) + const
    #            = k*log(Φ(f)) + (N-k)*log(1-Φ(f)) - 0.5/sigma² * (f - mu)² + const

    var_prior = sigma_prior**2

    def neg_log_posterior(f):
        # Log likelihood term
        log_likelihood = k * torch.log(probit(f) + eps) + \
                         (N - k) * torch.log(1 - probit(f) + eps)

        # Log prior term
        log_prior = -0.5 * (f - mu_prior)**2 / var_prior

        return -(log_likelihood + log_prior)

    # 2. Find the Mode (MAP) using numerical optimization
    # We need an initial guess for f. The prior mean is a safe choice.
    # The true proportion k/N mapped through the inverse probit is also a good start.
    if 0 < k < N:
        initial_guess = float(norm.ppf(k / N))
    else:
        initial_guess = mu_prior

    f_map_tensor = torch.tensor(initial_guess, requires_grad=True, dtype=torch.float64)

    # Use a robust optimizer like LBFGS
    optimizer = torch.optim.LBFGS([f_map_tensor], lr=0.1, max_iter=100)

    def closure():
        optimizer.zero_grad()
        loss = neg_log_posterior(f_map_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)

    mu_posterior = f_map_tensor.item()

    # 3. Calculate the new variance from the curvature at the MAP
    # The new precision is the second derivative of the negative log posterior.
    # We can use torch's functional hessian for this.
    hessian_val = torch.autograd.functional.hessian(
        neg_log_posterior,
        f_map_tensor
    )

    # For a 1D function, the hessian is just a 1x1 tensor
    precision_posterior = hessian_val.item()

    # Handle the case of zero precision (e.g., if N=0)
    if precision_posterior <= 0:
        # This shouldn't happen with N > 0 and a proper prior
        variance_posterior = float('inf')
    else:
        variance_posterior = 1.0 / precision_posterior

    sigma_posterior = np.sqrt(variance_posterior)

    return mu_posterior, sigma_posterior


#_norm_pdf = norm.pdf
#_norm_cdf = norm.cdf
#
#
#def laplace_approximation_probit(mu_prior: float,
#                                 sigma_prior: float,
#                                 N: float,
#                                 k: float,
#                                 max_iter: int = 100,
#                                 tol: float = 1e-6) -> tuple[float, float]:
#    """
#    Computes the Laplace approximation using a numerically stable Newton-Raphson method.
#    """
#    if sigma_prior <= 0:
#        raise ValueError("Prior standard deviation must be positive.")
#    if N < 0:
#        raise ValueError("N cannot be negative.")
#    if N == 0:
#        return mu_prior, sigma_prior
#
#    var_prior = sigma_prior ** 2
#    precision_prior = 1.0 / var_prior
#
#    f_map = mu_prior
#    if 0 < k < N:
#        try:
#            f_map = norm.ppf(k / N)
#        except (ValueError, ZeroDivisionError):
#            f_map = mu_prior
#
#    for _ in range(max_iter):
#        f_old = f_map
#        cdf_f = _norm_cdf(f_map)
#        pdf_f = _norm_pdf(f_map)
#
#        # Guard against numerical instability in the tails
#        if cdf_f < 1e-12 or (1 - cdf_f) < 1e-12:
#            if _ > 0:
#                break
#            else:
#                f_map = mu_prior; continue
#
#        grad_log_lik = (k / cdf_f - (N - k) / (1 - cdf_f)) * pdf_f
#        gradient = -(grad_log_lik) + (f_map - mu_prior) / var_prior
#
#        info_per_sample = pdf_f ** 2 / (cdf_f * (1 - cdf_f))
#        hessian = N * info_per_sample + precision_prior
#
#        if hessian <= 0: break  # Avoid division by zero
#        f_map = f_map - gradient / hessian
#        if abs(f_map - f_old) < tol: break
#
#    cdf_f = _norm_cdf(f_map)
#    pdf_f = _norm_pdf(f_map)
#
#    if cdf_f < 1e-12 or (1 - cdf_f) < 1e-12:
#        info_per_sample = 0
#    else:
#        info_per_sample = pdf_f ** 2 / (cdf_f * (1 - cdf_f))
#
#    precision_posterior = N * info_per_sample + precision_prior
#
#    if precision_posterior <= 0:
#        variance_posterior = float('inf')
#    else:
#        variance_posterior = 1.0 / precision_posterior
#
#    return f_map, np.sqrt(variance_posterior)
