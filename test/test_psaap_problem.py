import unittest
import numpy as np
import torch
import matplotlib.pyplot as plt

# Import the functions and the trained model from the problem file
from src.problems.psaap_example import (
    train_model,
    sample_HF_outcomes,
    sample_LF_outcomes,
    get_HF_probs,
    get_LF_probs,
)


# --- 1. Reproducibility Test ---

def test_reproducibility():
    """
    Tests if training the model twice with the same seeds results in
    identical model parameters.
    """
    print("--- Running Reproducibility Test ---")

    # Train the first model
    print("Training model 1...")
    model1 = train_model()

    # Train the second model
    print("Training model 2...")
    model2 = train_model()

    # Compare the state dictionaries of the two models
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()

    all_match = True
    for param_name in state_dict1:
        if not torch.allclose(state_dict1[param_name], state_dict2[param_name]):
            print(f"Mismatch found in parameter: {param_name}")
            all_match = False
            break

    if all_match:
        print("SUCCESS: Model training is reproducible.")
    else:
        print("FAILURE: Model training is NOT reproducible.")

    assert all_match, "Reproducibility test failed."
    print("-" * 35 + "\n")


# --- 2. Unit Tests ---

class TestProblemFunctions(unittest.TestCase):
    """
    Unit tests for the sampling and probability functions.
    """

    def setUp(self):
        """Set up common test data before each test."""
        self.num_points = 20
        # Create a sample input array with 2 features
        self.X_test = np.random.rand(self.num_points, 2)

    def test_reproducibility(self):
        test_reproducibility()

    def test_visualisations(self):
        generate_visualizations()

    def test_get_probs(self):
        """Test the get_..._probs functions."""
        for func, name in zip([get_HF_probs, get_LF_probs], ["HF", "LF"]):
            with self.subTest(msg=f"Testing get_{name}_probs"):
                # 1. Get probabilities
                probs = func(self.X_test)

                # 2. Check type - should be a numpy array
                self.assertIsInstance(probs, np.ndarray, f"{name} probs should be a numpy array")

                # 3. Check shape - should be a 1D array of length num_points
                self.assertEqual(probs.shape, (self.num_points,), f"{name} probs have incorrect shape")

                # 4. Check range - all values must be between 0 and 1
                self.assertTrue(np.all((probs >= 0) & (probs <= 1)), f"{name} probs are not in [0, 1] range")

    def test_sample_outcomes(self):
        """Test the sample_..._outcomes functions."""
        for func, name in zip([sample_HF_outcomes, sample_LF_outcomes], ["HF", "LF"]):
            with self.subTest(msg=f"Testing sample_{name}_outcomes"):
                # 1. Get sampled outcomes
                outcomes = func(self.X_test)

                # 2. Check type
                self.assertIsInstance(outcomes, np.ndarray, f"{name} outcomes should be a numpy array")

                # 3. Check shape - should be (num_points,) as per problem spec
                self.assertEqual(outcomes.shape, (self.num_points,), f"{name} outcomes have incorrect shape")

                # 4. Check values - all outcomes must be either 0 or 1
                unique_values = np.unique(outcomes)
                self.assertTrue(np.all(np.isin(unique_values, [0, 1])),
                                f"{name} outcomes contain values other than 0 or 1")


# --- 3. Visualizations ---

def generate_visualizations():
    """
    Generates and displays plots for the probabilities and samples
    over a 2D grid of input points.
    """
    print("--- Generating Visualizations ---")

    # Create a grid of test points for visualization
    grid_size = 50
    x1 = np.linspace(0, 1, grid_size)
    x2 = np.linspace(0, 1, grid_size)
    X1, X2 = np.meshgrid(x1, x2)
    X_test_grid = np.vstack([X1.ravel(), X2.ravel()]).T

    # Get probabilities and samples for the grid
    hf_probs = get_HF_probs(X_test_grid).reshape(grid_size, grid_size)
    lf_probs = get_LF_probs(X_test_grid).reshape(grid_size, grid_size)

    # Note: Samples will be different each time this script is run
    hf_samples = sample_HF_outcomes(X_test_grid).reshape(grid_size, grid_size)
    lf_samples = sample_LF_outcomes(X_test_grid).reshape(grid_size, grid_size)

    # Create the plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    fig.suptitle("BFGPC Model Predictions and Samples", fontsize=16)

    # Plot labels and details
    x_label = "Normalized Feature 1 (xis_2)"
    y_label = "Normalized Feature 2 (xis_4)"

    # High-Fidelity Probability
    im_hf_prob = axes[0, 0].imshow(hf_probs, origin='lower', extent=(0, 1, 0, 1), cmap='viridis', vmin=0, vmax=1)
    fig.colorbar(im_hf_prob, ax=axes[0, 0], label="Probability")
    axes[0, 0].set_title("High-Fidelity: True Probabilities")
    axes[0, 0].set_ylabel(y_label)

    # High-Fidelity Samples
    axes[0, 1].scatter(X_test_grid[:, 0], X_test_grid[:, 1], c=hf_samples.ravel(), cmap='viridis', s=10, vmin=0, vmax=1)
    axes[0, 1].set_title("High-Fidelity: One Realization (Sample)")

    # Low-Fidelity Probability
    im_lf_prob = axes[1, 0].imshow(lf_probs, origin='lower', extent=(0, 1, 0, 1), cmap='viridis', vmin=0, vmax=1)
    fig.colorbar(im_lf_prob, ax=axes[1, 0], label="Probability")
    axes[1, 0].set_title("Low-Fidelity: True Probabilities")
    axes[1, 0].set_xlabel(x_label)
    axes[1, 0].set_ylabel(y_label)

    # Low-Fidelity Samples
    axes[1, 1].scatter(X_test_grid[:, 0], X_test_grid[:, 1], c=lf_samples.ravel(), cmap='viridis', s=10, vmin=0, vmax=1)
    axes[1, 1].set_title("Low-Fidelity: One Realization (Sample)")
    axes[1, 1].set_xlabel(x_label)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    print("Displaying plots. Close the plot window to exit.")
    plt.show()

