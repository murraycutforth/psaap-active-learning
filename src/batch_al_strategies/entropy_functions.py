import math
from collections import Counter

import torch
from gpytorch.distributions import MultivariateNormal
from torch.distributions import Bernoulli


def calculate_entropy_from_samples(samples: torch.Tensor) -> float:
    """
    Calculates the empirical entropy H(X) from a set of samples.
    H(X) = - sum_{x} p(x) log(p(x))

    Args:
        samples (torch.Tensor): A tensor of shape (n_samples, n_dimensions)
                                where each row is a sample.

    Returns:
        float: The estimated entropy in nats.
    """
    if samples.shape[1] == 0:
        return 0.0

    counts = Counter(map(tuple, samples.tolist()))

    n_total_samples = samples.shape[0]
    entropy = 0.0

    for count in counts.values():
        p = count / n_total_samples
        if p > 0:
            entropy -= p * math.log(p)  # log is base e (nats)

    return entropy


def calculate_entropy_from_samples_miller_madow(samples: torch.Tensor) -> float:
    """
    Calculates the empirical entropy H(X) from a set of samples with Miller-Madow correction for small sample size
    """
    n_samples, n_dim = samples.shape
    if n_dim == 0 or n_samples == 0:
        return 0.0

    counts = Counter(map(tuple, samples.tolist()))
    k_observed = len(counts)

    entropy_ml = 0.0
    for count in counts.values():
        p = count / n_samples
        entropy_ml -= p * math.log(p)

    # Additive bias correction: H_MM = H + (k_observed - 1) / (2 * N)
    bias_correction = (k_observed - 1) / (2 * n_samples)
    return entropy_ml + bias_correction


def estimate_marginal_entropy_H_Y(
        M: int,
        K: int,
        joint_mvn: MultivariateNormal,
        d: int,
        seed: int = None
):
    """
    Estimates the marginal entropy H(Y) using a nested MC scheme.

    H(Y) is the first term in the mutual information I(Y;Q) = H(Y) - H(Y|Q).
    This function's logic is analogous to the first term in BatchBALD.

    Args:
        M (int): Number of outer loop samples (for the expectation over Y).
        K (int): Number of inner loop samples (for marginalizing out R).
        joint_mvn (MultivariateNormal): The joint distribution p(R, Q).
        d (int): The dimension of the candidate batch R (and labels Y).
        seed (int, optional): Random seed for reproducibility.

    Returns:
        torch.Tensor: A scalar estimate of H(Y).
    """
    if seed is not None:
        torch.manual_seed(seed)

    probit_link = torch.distributions.Normal(0, 1).cdf

    # 1. SETUP: Extract the marginal distribution p(R)
    # --------------------------------------------------
    dim_R = d
    R_idxs = torch.arange(0, dim_R)

    mu_R = joint_mvn.mean[R_idxs]
    cov_RR = joint_mvn.covariance_matrix[R_idxs][:, R_idxs]

    # Add jitter to the covariance for stable sampling
    cov_RR_stable = 0.5 * (cov_RR + cov_RR.T) + 1e-6 * torch.eye(dim_R)
    p_R = MultivariateNormal(mu_R, cov_RR_stable)

    # 2. MAIN ESTIMATION LOOP
    # -----------------------
    total_entropy_estimate = 0.0
    log_k = torch.log(torch.tensor(K, dtype=torch.float32))

    # Outer loop: approximates E_{p(Y)}[...] with M samples
    for _ in range(M):
        # a) Sample K latent functions R ~ p(R)
        # R_samples shape: (K, d)
        R_samples = p_R.sample(torch.Size((K,)))

        # b) Generate ONE label sequence Y_sample from p(Y)
        # We do this by picking one of our R_samples (e.g., the first one)
        # and sampling a Y from the corresponding Bernoulli likelihood.
        p_y_one_for_sampling = probit_link(R_samples[0])
        Y_sample = Bernoulli(p_y_one_for_sampling).sample()  # Shape: (d,)

        # c) Estimate log p(Y_sample) by marginalizing out R
        # This requires the log-sum-exp trick over our K samples of R.

        # Calculate p(Y_i | R_i) for each Y_i in Y_sample and for each R_sample
        probs_one_given_R = probit_link(R_samples)  # Shape: (K, d)

        # Use torch.where to pick log(p) if Y_sample is 1, and log(1-p) if 0
        log_prob_Y_given_R = torch.where(
            Y_sample.bool(),
            torch.log(probs_one_given_R),
            torch.log(1.0 - probs_one_given_R)
        )

        # Sum the log probabilities over the 'd' dimension to get log p(Y_sample | R_j)
        # for each of the K samples of R.
        log_p_vector = log_prob_Y_given_R.sum(dim=1)  # Shape: (K,)

        # Use log-sum-exp to compute log( (1/K) * sum(exp(log_p_vector)) )
        log_p_Y = torch.logsumexp(log_p_vector, dim=0) - log_k

        # d) The entropy contribution for this Y_sample is -log p(Y_sample)
        total_entropy_estimate += -log_p_Y

    # 3. FINAL RESULT
    # ---------------
    # Average the estimates from the M outer loops
    final_H_Y = total_entropy_estimate / M
    return final_H_Y


def estimate_conditional_entropy_H_Y_given_Q(
        M: int,
        K: int,
        joint_mvn: MultivariateNormal,
        d: int,
        seed: int = None
):
    """
    Estimates the conditional entropy H(Y|Q) using a doubly-nested MC scheme.

    This corresponds to E_{p(Q)}[H(Y|Q)].

    Args:
        M (int): Number of outer loop samples (for the expectation over Q).
        K (int): Number of inner loop samples (for marginalizing out R).
        joint_mvn (MultivariateNormal): The joint distribution p(R, Q).
        d (int): The dimension of the candidate batch R (and labels Y).
        seed (int, optional): Random seed for reproducibility.

    Returns:
        torch.Tensor: A scalar estimate of H(Y|Q).
    """
    if seed is not None:
        torch.manual_seed(seed)

    # 1. SETUP & PRE-COMPUTATION
    # ---------------------------
    dim_R = d
    dim_Q = joint_mvn.mean.shape[0] - d
    n_total = dim_R + dim_Q

    R_idxs = torch.arange(0, dim_R)
    Q_idxs = torch.arange(dim_R, n_total)
    probit_link = torch.distributions.Normal(0, 1).cdf

    mean = joint_mvn.mean
    cov = joint_mvn.covariance_matrix

    # Extract marginal and conditional components from the joint covariance
    mu_R = mean[R_idxs]
    mu_Q = mean[Q_idxs]
    cov_RR = cov[R_idxs][:, R_idxs]
    cov_RQ = cov[R_idxs][:, Q_idxs]
    cov_QR = cov[Q_idxs][:, R_idxs]
    cov_QQ = cov[Q_idxs][:, Q_idxs]

    # Pre-compute expensive, Q-independent parts for the conditional distribution p(R|Q)
    cov_QQ_inv = torch.linalg.inv(cov_QQ)
    # The conditional covariance is the same for all samples of Q
    cond_cov_R_given_Q = cov_RR - cov_RQ @ cov_QQ_inv @ cov_QR
    # Symmetrize and add jitter for numerical stability
    cond_cov_R_given_Q = 0.5 * (cond_cov_R_given_Q + cond_cov_R_given_Q.T) + 1e-6 * torch.eye(dim_R)

    # Define the marginal distribution of Q to sample from
    # Add jitter to the covariance for stable sampling
    cov_QQ_stable = 0.5 * (cov_QQ + cov_QQ.T) + 1e-6 * torch.eye(dim_Q)
    p_Q = MultivariateNormal(mu_Q, cov_QQ_stable)

    # 2. MAIN ESTIMATION LOOP
    # -----------------------
    total_entropy_estimate = 0.0
    log_k = torch.log(torch.tensor(K, dtype=torch.float32))

    # Outer loop: approximates E_{p(Q)}[...] with M samples
    for _ in range(M):
        # a) Sample a single Q_sample ~ p(Q)
        Q_sample = p_Q.sample()

        # b) Define the conditional distribution p(R | Q = Q_sample)
        cond_mean_R_given_Q = mu_R + cov_RQ @ cov_QQ_inv @ (Q_sample - mu_Q)
        dist_R_given_Q = MultivariateNormal(cond_mean_R_given_Q, cond_cov_R_given_Q)

        # c) Sample K latent functions R ~ p(R | Q = Q_sample)
        # R_samples shape: (K, d)
        R_samples = dist_R_given_Q.sample(torch.Size((K,)))

        # --- Inner Estimation of H(Y | Q=Q_sample) ---
        # We estimate this "hard" entropy with a single sample of Y, Y_sample,
        # by computing -log p(Y_sample | Q=Q_sample).

        # d) Generate ONE label sequence Y_sample from p(Y | Q=Q_sample)
        # We do this by picking one of our R_samples (e.g., the first one)
        # and sampling a Y from it.
        p_y_one_for_sampling = probit_link(R_samples[0])
        Y_sample = Bernoulli(p_y_one_for_sampling).sample()  # Shape: (d,)

        # e) Estimate log p(Y_sample | Q=Q_sample) by marginalizing out R
        # This requires the log-sum-exp trick over our K samples of R.
        # This is a vectorized implementation.

        # Calculate p(Y_i | R_i) for each Y_i in Y_sample and for each R_sample
        probs_one_given_R = probit_link(R_samples)  # Shape: (K, d)

        # Use torch.where to pick log(p) if Y_sample is 1, and log(1-p) if 0
        log_prob_Y_given_R = torch.where(
            Y_sample.bool(),
            torch.log(probs_one_given_R),
            torch.log(1.0 - probs_one_given_R)
        )

        # Sum the log probabilities over the 'd' dimension to get log p(Y_sample | R_j)
        # for each of the K samples of R.
        log_p_vector = log_prob_Y_given_R.sum(dim=1)  # Shape: (K,)

        # Use log-sum-exp to compute log( (1/K) * sum(exp(log_p_vector)) )
        log_p_Y_given_Q = torch.logsumexp(log_p_vector, dim=0) - log_k

        # f) The entropy contribution for this Q_sample is -log p(Y_sample | Q=Q_sample)
        total_entropy_estimate += -log_p_Y_given_Q

    # 3. FINAL RESULT
    # ---------------
    # Average the estimates from the M outer loops
    final_H_Y_given_Q = total_entropy_estimate / M
    return final_H_Y_given_Q
