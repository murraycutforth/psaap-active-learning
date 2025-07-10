import time
import unittest
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.models.bfgpc import BFGPC_ELBO


class TestBFGPCSpeed(unittest.TestCase):
    model = None  # Class attribute to hold the trained model

    @classmethod
    def setUpClass(cls):
        print("\nSetting up a trained BFGPC_ELBO model for the speed test...")
        cls.model = BFGPC_ELBO()

        # Use small dummy data for quick training
        # Training data set up as before
        X_L_train = torch.rand(50, 2, dtype=torch.float32)
        Y_L_train = torch.sin(X_L_train.sum(dim=1))
        X_H_train = torch.rand(25, 2, dtype=torch.float32)
        Y_H_train = torch.cos(X_H_train.sum(dim=1)) + torch.sin(X_H_train.sum(dim=1))

        try:
            cls.model.train_model(X_L_train, Y_L_train, X_H_train, Y_H_train, lr=0.1, n_epochs=100)
        except Exception as e:
            print(f"An error occurred during model training in setUpClass: {e}")
            raise

        print("Model setup complete.")

    def test_log_determinant_comparison(self):
        """
        Test to compare the log determinant values computed by both methods
        to ensure they are approximately equal.
        """
        self.assertIsNotNone(self.model, "Model should have been trained in setUpClass")

        N_reps = 100

        for _ in range(N_reps):

            d = 2  # Input dimension of the model
            N_L, N_H, N_prime = 100, 50, 100  # Example input sizes
            X_L = torch.rand(N_L, d, dtype=torch.float32)
            X_H_and_prime_combined = torch.rand(N_H + N_prime, d, dtype=torch.float32)
            X_H = X_H_and_prime_combined[:N_H]
            X_prime = X_H_and_prime_combined[N_H:]

            # Get log determinant from lazy method
            mvn_lazy = self.model.predict_multi_fidelity_latent_joint_lazy(X_L, X_H, X_prime)
            log_det_lazy = torch.logdet(mvn_lazy.covariance_matrix)

            # Get log determinant from regular method
            mvn_regular = self.model.predict_multi_fidelity_latent_joint(X_L, X_H, X_prime)
            log_det_regular = torch.logdet(mvn_regular.covariance_matrix)

            # Check if the log determinants are approximately equal
            self.assertAlmostEqual(log_det_lazy.item(), log_det_regular.item(), delta=0.2,
                                   msg="Log determinants should be approximately equal.")

    @unittest.skip
    def test_predict_speed_comparison(self):
        self.assertIsNotNone(self.model, "Model should have been trained in setUpClass")

        d = 2  # Input dimension of the model
        base_sizes = [10, 50, 100, 200, 400, 800, 1200]
        input_configs = [
            {'N_L': s, 'N_H': s // 2, 'N_prime': s} for s in base_sizes
        ]

        timing_results_lazy = []
        timing_results = []

        print("\nRunning speed tests for `predict_multi_fidelity_latent_joint` and its lazy variant...")
        for config in input_configs:
            N_L, N_H, N_prime = config['N_L'], config['N_H'], config['N_prime']
            X_L = torch.rand(N_L, d, dtype=torch.float32)
            X_H_and_prime_combined = torch.rand(N_H + N_prime, d, dtype=torch.float32)
            X_H = X_H_and_prime_combined[:N_H]
            X_prime = X_H_and_prime_combined[N_H:]

            # Timing the lazy method
            start_time = time.perf_counter()
            _ = self.model.predict_multi_fidelity_latent_joint_lazy(X_L, X_H, X_prime)
            end_time = time.perf_counter()
            timing_results_lazy.append(
                {'N_L': N_L, 'N_H': N_H, 'N_prime': N_prime, 'time_seconds': end_time - start_time})

            # Timing the regular method
            start_time = time.perf_counter()
            _ = self.model.predict_multi_fidelity_latent_joint(X_L, X_H, X_prime)
            end_time = time.perf_counter()
            timing_results.append({'N_L': N_L, 'N_H': N_H, 'N_prime': N_prime, 'time_seconds': end_time - start_time})

        # Process and Save Results
        self._save_results_and_plots(timing_results_lazy, timing_results, "predict_speed_comparison")

    def test_log_determinant_speed_comparison(self):
        self.assertIsNotNone(self.model, "Model should have been trained in setUpClass")

        d = 2  # Input dimension of the model
        base_sizes = [10, 50, 100, 200, 800, 1200, 2000, 4000]
        input_configs = [
            {'N_L': s, 'N_H': s // 2, 'N_prime': s} for s in base_sizes
        ]

        timing_results_lazy = []
        timing_results = []

        print("\nRunning speed tests for log determinant functions...")
        for config in input_configs:
            N_L, N_H, N_prime = config['N_L'], config['N_H'], config['N_prime']
            X_L = torch.rand(N_L, d, dtype=torch.float32)
            X_H_and_prime_combined = torch.rand(N_H + N_prime, d, dtype=torch.float32)
            X_H = X_H_and_prime_combined[:N_H]
            X_prime = X_H_and_prime_combined[N_H:]

            # Timing the lazy method log determinant
            start_time = time.perf_counter()
            mvn = self.model.predict_multi_fidelity_latent_joint_lazy(X_L, X_H,
                                                                            X_prime)
            torch.logdet(mvn.covariance_matrix)
            end_time = time.perf_counter()
            timing_results_lazy.append(
                {'N_L': N_L, 'N_H': N_H, 'N_prime': N_prime, 'time_seconds': end_time - start_time})

            # Timing the regular method log determinant
            start_time = time.perf_counter()
            mvn = self.model.predict_multi_fidelity_latent_joint(X_L, X_H,
                                                                       X_prime)
            torch.logdet(mvn.covariance_matrix)
            end_time = time.perf_counter()
            timing_results.append({'N_L': N_L, 'N_H': N_H, 'N_prime': N_prime, 'time_seconds': end_time - start_time})

        # Process and Save Results
        self._save_results_and_plots(timing_results_lazy, timing_results, "log_determinant_speed_comparison")

    def _save_results_and_plots(self, results_lazy, results, test_name):
        # Save timing data to CSV files
        results_df_lazy = pd.DataFrame(results_lazy)
        results_df = pd.DataFrame(results)

        output_dir = Path("output")
        output_dir.mkdir(parents=True, exist_ok=True)

        lazy_csv_path = Path(f"output/speed_test_{test_name}_lazy.csv")
        results_df_lazy.to_csv(lazy_csv_path, index=False)

        csv_path = Path(f"output/speed_test_{test_name}.csv")
        results_df.to_csv(csv_path, index=False)

        print(f"Results for lazy method saved to {lazy_csv_path}")
        print(f"Results for regular method saved to {csv_path}")

        # Create and save plots
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(results_df_lazy['N_L'], results_df_lazy['time_seconds'], marker='o', linestyle='-',
                label='Lazy Version')
        ax.plot(results_df['N_L'], results_df['time_seconds'], marker='x', linestyle='--', label='Regular Version')
        ax.set_xlabel("Number of Input Points (N_L)")
        ax.set_ylabel("Execution Time (seconds)")
        ax.set_title(f"Execution Time Comparison for {test_name}")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.loglog()
        ax.legend()

        plot_path = Path(f"output/speed_test_{test_name}.png")
        plt.savefig(plot_path)
        print(f"Plot for {test_name} saved to {plot_path}")


# This block allows the script to be run directly from the command line
if __name__ == '__main__':
    unittest.main()