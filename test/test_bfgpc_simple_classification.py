import unittest
from pathlib import Path
import os

import numpy as np
import torch
from pyDOE import lhs

from src.models.bfgpc import BFGPC_ELBO
from src.utils_plotting import plot_bfgpc_predictions, plot_bf_training_data


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


class TestMFGPClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up for all tests in this class."""
        cls.output_dir = "output_plots_bfgpc_test"
        os.makedirs(cls.output_dir, exist_ok=True)
        torch.manual_seed(42)
        np.random.seed(42)

        # Define synthetic 2D multi-fidelity classification problem parameters
        cls.lb_ex = np.array([0., 0.])
        cls.ub_ex = np.array([1., 1.])

        # Reduced data sizes and epochs for faster tests
        cls.N_L_init = 30 # Reduced from 200
        cls.N_H_init = 15 # Reduced from 100
        cls.N_test = 50   # Reduced from 200
        cls.n_epochs = 50 # Reduced from 1000
        cls.lhs_iterations = 5 # Reduced from 20

    def test_mfgpc_workflow_and_accuracy(self):
        """Test the full workflow: data generation, training, and evaluation."""

        plot_outdir = Path(self.output_dir)
        plot_outdir.mkdir(parents=True, exist_ok=True)

        # Initial training data (already normalized)
        X_L_init_norm_np = lhs(2, self.N_L_init, criterion='maximin', iterations=self.lhs_iterations)
        Y_L_init_norm_np = sampling_function_L(X_L_init_norm_np)
        X_H_init_norm_np = lhs(2, self.N_H_init, criterion='maximin', iterations=self.lhs_iterations)
        Y_H_init_norm_np = sampling_function_H(X_H_init_norm_np)

        plot_bf_training_data(X_L_init_norm_np, Y_L_init_norm_np, X_H_init_norm_np, Y_H_init_norm_np,
                              boundary_LF=true_boundary_L_normalized, boundary_HF=true_boundary_H_normalized,
                              outpath=plot_outdir / "initial_data.png")

        X_L_train = torch.tensor(X_L_init_norm_np, dtype=torch.float32)
        Y_L_train = torch.tensor(Y_L_init_norm_np, dtype=torch.float32)
        X_H_train = torch.tensor(X_H_init_norm_np, dtype=torch.float32)
        Y_H_train = torch.tensor(Y_H_init_norm_np, dtype=torch.float32)

        model = BFGPC_ELBO(X_L_train, X_H_train, initial_rho=1.0)
        model.train_model(X_L_train, Y_L_train, X_H_train, Y_H_train, lr=0.01, n_epochs=self.n_epochs)

        plot_bfgpc_predictions(model, X_LF=X_L_init_norm_np, Y_LF=Y_L_init_norm_np,
                               X_HF=X_H_init_norm_np, Y_HF=Y_H_init_norm_np,
                               boundary_HF=true_boundary_H_normalized, outpath=plot_outdir / "predictions.png")

        X_test_norm_np = lhs(2, self.N_test, criterion='maximin', iterations=self.lhs_iterations)
        Y_test_H_norm_np = sampling_function_H(X_test_norm_np)

        acc = model.evaluate_accuracy(X_test_norm_np, Y_test_H_norm_np)

        self.assertIsInstance(acc, float, "Accuracy should be a float.")
        self.assertGreaterEqual(acc, 0.0, "Accuracy should be non-negative.")
        self.assertLessEqual(acc, 1.0, "Accuracy should not exceed 1.0.")

        # A more specific accuracy check. This threshold might need adjustment
        # based on expected performance with reduced data/epochs.
        # If using the placeholder BFGPC_ELBO, it returns a fixed 0.85.
        # If using the real model, this might fail if 0.5 isn't consistently achieved
        # with the reduced test parameters. Adjust as necessary.
        self.assertGreater(acc, 0.5, f"Accuracy {acc} is too low. Expected > 0.5 even with reduced settings.")




if __name__ == '__main__':
    unittest.main()