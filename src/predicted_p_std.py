import numpy as np


def compute_pred_p_std(gpc, X_test, N: int = 100):
    # Get parameters of posterior distribution of f_star
    # Sample, and pass through link function
    # Compute std of samples

    f_star_mean, L_cov = gpc.latent_predictive_distribution(X_test)

    samples = []

    for _ in range(N):
        f_star_sample = f_star_mean + L_cov @ np.random.normal(0, 1, size=f_star_mean.shape)
        p_star_sample = 1 / (1 + np.exp(-f_star_sample))
        samples.append(p_star_sample)

    return np.std(samples, axis=0)


