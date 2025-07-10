import numpy as np
from scipy.stats import norm
from scipy.interpolate import interp1d
from tqdm import tqdm  # Use tqdm.auto for scripts
import time
import warnings

from src.models.laplace_approximation import laplace_approximation_probit


class N_Estimator:
    """
    An object that precomputes and inverts the mapping from (mu, sigma, N) to
    sigma_posterior, allowing for estimation of N required to reach a target
    posterior standard deviation.
    """

    def __init__(self, mu_grid, sigma_grid, n_grid):
        """
        Initializes and runs the precomputation.

        Args:
            mu_grid (np.ndarray): 1D array of prior means to precompute for.
            sigma_grid (np.ndarray): 1D array of prior std devs to precompute for.
            n_grid (np.ndarray): 1D array of N values to precompute for.
        """
        print("Starting precomputation of the N-estimator grid...")
        start_time = time.time()

        self.mu_grid = np.array(mu_grid)
        self.sigma_grid = np.array(sigma_grid)
        self.n_grid = np.array(n_grid)
        self.interpolators = {}

        # Use tqdm for a nice progress bar
        for mu_prior in tqdm(self.mu_grid, desc="mu_prior"):
            for sigma_prior in self.sigma_grid:

                computed_n = []
                computed_sigma_post = []

                for n_val in self.n_grid:
                    # Use the expected value of k, making the computation deterministic
                    k_expected = n_val * norm.cdf(mu_prior)

                    try:
                        _, sigma_post = laplace_approximation_probit(
                            mu_prior, sigma_prior, n_val, k_expected
                        )
                        # Only store valid, finite results
                        if np.isfinite(sigma_post) and sigma_post < sigma_prior:
                            computed_n.append(n_val)
                            computed_sigma_post.append(sigma_post)
                    except (ValueError, RuntimeError):
                        # Skip if the optimizer or inputs fail for some reason
                        continue

                # We need at least 2 points to create an interpolator
                if len(computed_n) < 2:
                    continue

                # To create the inverse interpolator (sigma_post -> N), the x-axis
                # (sigma_post) must be monotonically increasing.
                # So we sort by sigma_post.
                sort_indices = np.argsort(computed_sigma_post)
                sorted_sigma_post = np.array(computed_sigma_post)[sort_indices]
                sorted_n = np.array(computed_n)[sort_indices]

                # Remove duplicates which can cause issues for interp1d
                unique_indices = np.unique(sorted_sigma_post, return_index=True)[1]

                if len(unique_indices) < 2:
                    continue

                final_sigma_post = sorted_sigma_post[unique_indices]
                final_n = sorted_n[unique_indices]

                # Create the inverse interpolator and store it
                # fill_value="extrapolate" allows estimation outside the grid range
                # bounds_error=False prevents it from throwing an error
                inv_interpolator = interp1d(
                    final_sigma_post,
                    final_n,
                    kind='linear',
                    bounds_error=False,
                    fill_value='extrapolate'  # Fill with min/max N
                )
                self.interpolators[(mu_prior, sigma_prior)] = inv_interpolator

        end_time = time.time()
        print(f"Precomputation finished in {end_time - start_time:.2f} seconds.")

    def _get_closest_grid_point(self, mu_query, sigma_query):
        """Finds the nearest (mu, sigma) point on the precomputation grid."""
        mu_dist = np.abs(self.mu_grid - mu_query)
        sigma_dist = np.abs(self.sigma_grid - sigma_query)

        closest_mu_idx = np.argmin(mu_dist)
        closest_sigma_idx = np.argmin(sigma_dist)

        return self.mu_grid[closest_mu_idx], self.sigma_grid[closest_sigma_idx]

    def estimate_n(self, mu_prior, sigma_prior, sigma_posterior_target):
        """
        Estimates the number of samples N required to achieve a target posterior std dev.

        Args:
            mu_prior (float): The prior mean.
            sigma_prior (float): The prior standard deviation.
            sigma_posterior_target (float): The desired posterior standard deviation.

        Returns:
            float: The estimated number of samples N.
        """
        if sigma_posterior_target >= sigma_prior:
            warnings.warn("Target posterior sigma is not smaller than prior sigma. N=0 is assumed.")
            return 0.0

        # Find the interpolator for the closest grid point
        closest_mu, closest_sigma = self._get_closest_grid_point(mu_prior, sigma_prior)

        if (closest_mu, closest_sigma) not in self.interpolators:
            raise ValueError(
                f"No valid interpolator found for the closest grid point ({closest_mu}, {closest_sigma}). Try a denser grid.")

        interpolator = self.interpolators[(closest_mu, closest_sigma)]

        # Query the interpolator
        estimated_n = interpolator(sigma_posterior_target)

        return float(estimated_n)


if __name__ == '__main__':
    # --- 1. Define Grids ---
    # Choose grids that cover the expected range of your GP's predictive distribution

    # Grid for prior mean (mu_f)
    mu_grid = np.array([2, -1.5, -1, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])

    # Grid for prior standard deviation (sigma_f)
    sigma_grid = np.array([0.1, 0.5, 1.0, 2.0])

    # Grid for number of samples (N), log-spaced for better coverage
    n_grid = np.unique(np.logspace(0.5, 3, 15).astype(int))  # From 10 to 1000

    print("--- Grid Definitions ---")
    print(f"Mu Grid: {mu_grid}")
    print(f"Sigma Grid: {sigma_grid}")
    print(f"N Grid: {n_grid}")
    print("-" * 25)

    # --- 2. Create the Estimator Object ---
    # This will take some time as it runs the precomputation
    estimator = N_Estimator(mu_grid, sigma_grid, n_grid)

    print("\n--- Example Queries ---")

    # --- Case 1: On-grid query ---
    # My GP is fairly certain: mu=0.5, sigma=1.0.
    # How many samples to reduce sigma to 0.3?
    mu1, sigma1 = 0.5, 1.0
    target_sigma1 = 0.3
    est_n1 = estimator.estimate_n(mu1, sigma1, target_sigma1)
    print(f"Query 1: To go from (mu={mu1}, sigma={sigma1}) to sigma_post={target_sigma1}, we need N ≈ {est_n1:.0f}")

    # --- Case 2: Off-grid query ---
    # My GP is uncertain and gives mu=0.2, sigma=1.6
    # How many samples to reduce sigma to 0.5?
    # The tool will use the closest grid point: (mu=0.0, sigma=2.0)
    mu2, sigma2 = 0.2, 1.6
    target_sigma2 = 0.5
    closest_mu, closest_sigma = estimator._get_closest_grid_point(mu2, sigma2)
    est_n2 = estimator.estimate_n(mu2, sigma2, target_sigma2)
    print(f"Query 2: To go from (mu={mu2}, sigma={sigma2}) to sigma_post={target_sigma2}...")
    print(f"         (Using closest grid point mu={closest_mu}, sigma={closest_sigma})")
    print(f"         ...we need N ≈ {est_n2:.0f}")

    # --- Case 3: Ambitious target ---
    # Start with high uncertainty and aim for very low uncertainty
    mu3, sigma3 = 0.0, 2.0
    target_sigma3 = 0.1
    est_n3 = estimator.estimate_n(mu3, sigma3, target_sigma3)
    print(f"Query 3: To go from (mu={mu3}, sigma={sigma3}) to sigma_post={target_sigma3}, we need N ≈ {est_n3:.0f}")

    # --- Case 4: Impossible target ---
    # Target sigma is larger than prior sigma
    mu4, sigma4 = 0.5, 1.0
    target_sigma4 = 1.1
    try:
        est_n4 = estimator.estimate_n(mu4, sigma4, target_sigma4)
        print(f"Query 4: To go from (mu={mu4}, sigma={sigma4}) to sigma_post={target_sigma4}, we need N ≈ {est_n4:.0f}")
    except Exception as e:
        print(f"Query 4: Failed as expected. {e}")

    # Try halving sigma for various values of mu, sigma
    sigmas = [0.1, 0.2, 0.4, 0.8]
    mus = [-2, -1, 0, 1, 2]

    for mu in mus:
        for sigma in sigmas:
            target_sigma = sigma / 2
            est_n = estimator.estimate_n(mu, sigma, target_sigma)
            print(f"To go from (mu={mu}, sigma={sigma}) to sigma_post={target_sigma}, we need N approx {est_n:.3f}")
