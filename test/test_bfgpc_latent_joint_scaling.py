import time
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.bfgpc import BFGPC_ELBO


class TestBFGPCSpeed(unittest.TestCase):
    """
    A test suite for measuring the performance of the BFGPC_ELBO model.
    """
    model = None  # Class attribute to hold the trained model

    @classmethod
    def setUpClass(cls):
        """
        Set up a trained BFGPC_ELBO model once for all tests in this class.
        This is the unittest equivalent of a module-scoped pytest fixture.
        It avoids retraining the model for each test, saving significant time.
        """
        print("\nSetting up a trained BFGPC_ELBO model for the speed test...")
        model = BFGPC_ELBO()

        # Use small dummy data for quick training
        X_L_train = torch.rand(50, 2, dtype=torch.float32)
        Y_L_train = torch.sin(X_L_train.sum(dim=1))
        X_H_train = torch.rand(25, 2, dtype=torch.float32)
        Y_H_train = torch.cos(X_H_train.sum(dim=1)) + torch.sin(X_H_train.sum(dim=1))

        # Train for a few epochs just to get the model into a valid, initialised state.
        # We assume train_model does not require a 'verbose' argument. If it does,
        # you can try to pass verbose=False.
        try:
            # We assume your train_model might not have a verbose flag.
            model.train_model(X_L_train, Y_L_train, X_H_train, Y_H_train, lr=0.1, n_epochs=100)
        except Exception as e:
            # This catch block helps diagnose issues during setup.
            print(f"An error occurred during model training in setUpClass: {e}")
            raise

        cls.model = model
        print("Model setup complete.")

    def test_predict_joint_speed_scaling(self):
        """
        Measures the execution time of predict_multi_fidelity_latent_joint
        as a function of input tensor sizes. It then plots the results and
        saves the raw timing data to a CSV file.
        """
        # The test method must run without errors to pass.
        # We access the pre-trained model via self.model.
        self.assertIsNotNone(self.model, "Model should have been trained in setUpClass")

        d = 2  # Input dimension of the model

        # Define the various input sizes to test.
        # We will scale N_L, N_H, and N_prime together for a clear trend.
        base_sizes = [10, 50, 100, 200, 400, 800, 1200]
        input_configs = [
            {'N_L': s, 'N_H': s // 2, 'N_prime': s} for s in base_sizes
        ]

        # Directory to save outputs. This will be created inside the 'tests' directory.
        output_dir = Path("output_speed_test_predict_joint")
        output_dir.mkdir(parents=True, exist_ok=True)

        timing_results = []

        print("\nRunning speed test for `predict_multi_fidelity_latent_joint`...")
        for config in input_configs:
            N_L, N_H, N_prime = config['N_L'], config['N_H'], config['N_prime']

            # --- Generate unique input data for the function call ---
            X_L = torch.rand(N_L, d, dtype=torch.float32)

            # The function asserts that there are no duplicate points between X_H and X_prime.
            # Generating a single larger tensor and splitting it is an easy way to ensure this.
            X_H_and_prime_combined = torch.rand(N_H + N_prime, d, dtype=torch.float32)
            X_H = X_H_and_prime_combined[:N_H]
            X_prime = X_H_and_prime_combined[N_H:]

            total_unique_points = N_L + N_H + N_prime
            print(f"Testing with N_L={N_L}, N_H={N_H}, N_prime={N_prime} (Total unique points: {total_unique_points})")

            # --- Time the function call ---
            # time.perf_counter() is suitable for measuring short durations.
            start_time = time.perf_counter()
            _ = self.model.predict_multi_fidelity_latent_joint(X_L, X_H, X_prime)
            end_time = time.perf_counter()

            duration = end_time - start_time

            timing_results.append({
                'N_L': N_L,
                'N_H': N_H,
                'N_prime': N_prime,
                'total_points': total_unique_points,
                'time_seconds': duration
            })

        # --- Process and Save Results ---

        # 1. Save timing data to a CSV file
        results_df = pd.DataFrame(timing_results)
        csv_path = output_dir / "speed_test_results.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")
        print(results_df)

        # 2. Create and save a plot of the results
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(results_df['total_points'], results_df['time_seconds'], marker='o', linestyle='-')
        ax.set_xlabel("Total Number of Input Points (N_L + N_H + N_prime)")
        ax.set_ylabel("Execution Time (seconds)")
        ax.set_title("`predict_multi_fidelity_latent_joint` Speed vs. Input Size")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # A log-log scale is useful for visualizing polynomial complexity
        ax.set_xscale('log')
        ax.set_yscale('log')

        plt.tight_layout()

        plot_path = output_dir / "speed_test_plot.png"
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")

        # If the test reaches this point without an error, it's considered successful.
        self.assertTrue(True, "Test completed and artifacts generated successfully.")


# This block allows the script to be run directly from the command line
if __name__ == '__main__':
    unittest.main()