import numpy as np
from scipy.special import expit  # Numerically stable sigmoid


def bernoulli_likelihood(y, p):
    """Compute Bernoulli likelihood for binary observations."""
    # Clip probabilities to avoid numerical issues
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return p ** y * (1 - p) ** (1 - y)


def log_bernoulli_likelihood(y, p):
    """Compute log of Bernoulli likelihood for binary observations."""
    # Clip probabilities to avoid numerical issues
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return y * np.log(p) + (1 - y) * np.log(1 - p)


def sigmoid(x):
    """Numerically stable sigmoid function."""
    return expit(x)  # Use scipy's implementation for stability


def compute_log_marginal_likelihood(gpc, X_test, y_test, n_samples=100, random_state=None):
    """
    Compute mean log marginal likelihood using Monte Carlo integration.

    Parameters:
    -----------
    gpc : GaussianProcessClassifier
        Trained Gaussian Process Classifier
    X_test : array-like of shape (n_samples, n_features)
        Test input features
    y_test : array-like of shape (n_samples,)
        Test binary labels (0 or 1)
    n_samples : int, default=100
        Number of Monte Carlo samples
    random_state : int or None, default=None
        Random seed for reproducibility

    Returns:
    --------
    log_ml : float
        Estimated log marginal likelihood
    """
    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Get mean and covariance of latent predictive distribution
    f_star_mean, L_cov = gpc.latent_predictive_distribution(X_test)

    assert y_test.shape[0] == f_star_mean.shape[0], "Shapes of y_test and f_star_mean must match"

    # Initialize log likelihood accumulator
    log_ml_samples = np.zeros(n_samples)

    # Monte Carlo integration
    for i in range(n_samples):
        # Sample from latent predictive distribution
        f_sample = f_star_mean + L_cov @ np.random.normal(0, 1, size=f_star_mean.shape[0])

        # Transform to probability space
        p_sample = sigmoid(f_sample)

        # Compute log likelihood for this sample (sum of log likelihoods)
        log_ml_samples[i] = np.sum(log_bernoulli_likelihood(y_test, p_sample))

    # Use logsumexp trick for numerical stability when averaging in log space
    max_log_ml = np.max(log_ml_samples)
    log_ml = max_log_ml + np.log(np.mean(np.exp(log_ml_samples - max_log_ml)))

    return log_ml / len(y_test)

