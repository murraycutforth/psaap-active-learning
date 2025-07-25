import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, binom
from tqdm import tqdm

from src.models.laplace_approximation import laplace_approximation_probit


def run_convergence_experiment():
    """
    Runs an experiment to show how the Laplace approximation converges with N
    for different prior beliefs.
    """
    # --- Experiment Setup ---

    # 1. Define the "true" world
    f_true = 1.2  # The true underlying latent function value
    p_true = norm.cdf(f_true)  # The true probability of success (y=1)
    print(f"Ground Truth: f* = {f_true:.3f}, P(y=1|f*) = {p_true:.3f}")

    # 2. Define the different priors we want to test
    # A GPC model would provide these values as its predictive distribution.
    priors = {
        "Good Prior (Confident)": {"mu": 1.0, "sigma": 0.5},
        "Bad Prior (Confident)": {"mu": -1.0, "sigma": 0.5},
        "Vague Prior (Uncertain)": {"mu": 0.0, "sigma": 3.0},
    }

    # 3. Define the range of N values to test
    N_values = np.unique(np.logspace(0, 3.5, 30).astype(int))  # e.g., 1, 2, ..., 3162

    results = {}

    # --- Run Simulation ---
    for name, prior_params in priors.items():
        print(f"\nRunning simulation for: {name}")

        posterior_means = []
        posterior_vars = []

        for N in N_values:
            # Simulate new data Y* by drawing k from a Binomial distribution
            # For reproducibility, we use a fixed seed for each N, but you could
            # also average over multiple draws of k for a smoother plot.
            np.random.seed(N)
            k = binom.rvs(n=N, p=p_true)

            # Get the prior parameters
            mu_prior = prior_params["mu"]
            sigma_prior = prior_params["sigma"]

            # Compute the Laplace posterior
            mu_post, sigma_post = laplace_approximation_probit(mu_prior, sigma_prior, N, k)

            posterior_means.append(mu_post)
            posterior_vars.append(sigma_post ** 2)

        results[name] = {
            "means": np.array(posterior_means),
            "variances": np.array(posterior_vars)
        }

    # --- Plot Results ---
    #plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'text.latex.preamble': r'\usepackage{amsfonts}'
    })
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), dpi=200, sharex=True)

    # Plot 1: Convergence of the Posterior Mean
    for name, result_data in results.items():
        ax1.plot(N_values, result_data["means"], label=name, marker='.', linestyle='-')

    ax1.axhline(f_true, color='k', linestyle='--', label=f'True f* = {f_true:.2f}')
    ax1.set_ylabel("Posterior Mean $\\mathbb{E}[f* | Y*]$")
    #ax1.set_title("Convergence of Posterior Mean with Number of Observations (N)")
    ax1.set_xscale('log')
    ax1.legend()

    # Plot 2: Convergence of the Posterior Variance

    y_reference = 1.0 * N_values.astype(float)**(-1)

    for name, result_data in results.items():
        ax2.plot(N_values, result_data["variances"], label=name, marker='.', linestyle='-')

    ax2.plot(N_values, y_reference, color='k', linestyle='--', label="$\\mathcal{O}(N^{-1})$")

    ax2.set_xlabel("Number of New Observations (N)")
    ax2.set_ylabel("Posterior Variance Var$(f* | Y*)$")
    #ax2.set_title("Convergence of Posterior Variance with Number of Observations (N)")
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend()

    plt.tight_layout()
    plt.show()




# ==============================================================================
# New Visualization Function
# ==============================================================================

def visualize_convergence(mu_values, sigma_prior_values, n_grid):
    """
    Visualizes the convergence of the posterior standard deviation with N.

    Creates a multi-panel plot where each panel corresponds to a mu_prior value.
    Within each panel, lines show the convergence for different sigma_prior values.

    Args:
        mu_values (list or np.ndarray): A list of prior means to test.
        sigma_prior_values (list or np.ndarray): A list of prior std devs to test.
        n_grid (np.ndarray): 1D array of N values to plot against.
    """
    num_mus = len(mu_values)

    # Create a figure with one subplot per mu value
    fig, axes = plt.subplots(
        1, num_mus,
        dpi=200,
        figsize=(3 * num_mus, 3),
        sharey=True  # Share the Y-axis for easier comparison
    )

    # Define colors for the different sigma_prior lines
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(sigma_prior_values)))

    # Use tqdm to show progress over the outermost loop
    for i, mu_prior in enumerate(tqdm(mu_values, desc="Processing mu values")):
        ax = axes[i]

        for j, sigma_prior in enumerate(sigma_prior_values):
            posterior_sigmas = []

            for n_val in n_grid:
                # Use the expected value of k for a smooth curve
                k_expected = n_val * norm.cdf(mu_prior)

                _, sigma_post = laplace_approximation_probit(
                    mu_prior, sigma_prior, n_val, k_expected
                )
                posterior_sigmas.append(sigma_post)

            # Plot the convergence for this specific (mu, sigma) pair
            ax.plot(
                n_grid,
                posterior_sigmas,
                label=f'$\\sigma$_prior = {sigma_prior}',
                color=colors[j],
                marker='.',
                markersize=3
            )

        # Formatting for each subplot
        ax.set_title(f'$\\mu$ = {mu_prior}')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Number of Samples (N)')
        #ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        #ax.legend()

    # Set a common Y-label
    axes[0].set_ylabel('Posterior Standard Deviation')
    axes[1].legend()

    fig.suptitle('Convergence of Posterior Uncertainty with Sample Size (N)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
    plt.show()


def run_multi_convergence_experiment():

    # The 5 different mu values for the subplots
    mu_values = [-2.0, 0.0, 2.0]

    # The 3 different prior sigmas to show as lines in each plot
    sigma_prior_values = [1.0, 0.5, 0.1]

    # A log-spaced grid for N to clearly see the trend
    n_grid = np.unique(np.logspace(0, 4, 40).astype(int))  # From 1 to 10,000

    print("--- Visualization Parameters ---")
    print(f"Mu Values (Subplots): {mu_values}")
    print(f"Prior Sigma Values (Lines): {sigma_prior_values}")
    print(f"N Grid Range: {n_grid.min()} to {n_grid.max()}")
    print("-" * 30)

    # --- 2. Run the Visualization ---
    visualize_convergence(mu_values, sigma_prior_values, n_grid)


# Run the experiment
if __name__ == "__main__":
    #run_convergence_experiment()
    run_multi_convergence_experiment()