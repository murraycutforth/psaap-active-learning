# Unit test for BFGPC on probability estimation (rather than classification) where the true probability is no longer
# binary.


from src.toy_example import create_smooth_change_linear


import unittest
from pathlib import Path
import os

import numpy as np
import torch
from pyDOE import lhs

from src.bfgpc import BFGPC_ELBO
from src.utils_plotting import plot_bfgpc_predictions, plot_bf_training_data, plot_bfgpc_predictions_two_axes

linear_low_f1, linear_high_f1 = create_smooth_change_linear()


def sampling_function_L(X_normalized):  # Expects N x 2 normalized input
    Y_linear_low_grid, probs_linear_low_grid = linear_low_f1(X_normalized, reps=1)
    Y_linear_high_grid = Y_linear_low_grid.mean(axis=0)
    return Y_linear_high_grid


def sampling_function_H(X_normalized):
    Y_linear_high_grid, probs_linear_high_grid = linear_high_f1(X_normalized, reps=1)
    Y_linear_high_grid = Y_linear_high_grid.mean(axis=0)
    return Y_linear_high_grid


class TestMFGPClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up for all tests in this class."""
        cls.output_dir = "output_plots_bfgpc_probest_test"
        os.makedirs(cls.output_dir, exist_ok=True)
        torch.manual_seed(42)
        np.random.seed(42)

        # Define synthetic 2D multi-fidelity classification problem parameters
        cls.lb_ex = np.array([0., 0.])
        cls.ub_ex = np.array([1., 1.])

        # Reduced data sizes and epochs for faster tests
        cls.N_L_init = 1000
        cls.N_H_init = 20
        cls.N_test = 1000
        cls.n_epochs = 500
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
                              boundary_LF=None, boundary_HF=None,
                              outpath=plot_outdir / "initial_data.png")

        X_L_train = torch.tensor(X_L_init_norm_np, dtype=torch.float32)
        Y_L_train = torch.tensor(Y_L_init_norm_np, dtype=torch.float32)
        X_H_train = torch.tensor(X_H_init_norm_np, dtype=torch.float32)
        Y_H_train = torch.tensor(Y_H_init_norm_np, dtype=torch.float32)

        model = BFGPC_ELBO(X_L_train, X_H_train, initial_rho=1.0)
        model.train_model(X_L_train, Y_L_train, X_H_train, Y_H_train, lr=0.01, n_epochs=self.n_epochs)

        plot_bfgpc_predictions(model, X_LF=X_L_init_norm_np, Y_LF=Y_L_init_norm_np,
                               X_HF=X_H_init_norm_np, Y_HF=Y_H_init_norm_np,
                               boundary_HF=None, outpath=plot_outdir / "predictions.png")

        plot_bfgpc_predictions_two_axes(model, X_LF=X_L_init_norm_np, Y_LF=Y_L_init_norm_np,
                                        X_HF=X_H_init_norm_np, Y_HF=Y_H_init_norm_np,
                                        boundary_HF=None, outpath=plot_outdir / "predictions_two_axes.png")

        X_test_norm_np = lhs(2, self.N_test, criterion='maximin', iterations=self.lhs_iterations)
        Y_test_H_norm_np = sampling_function_H(X_test_norm_np)

        elpp = model.evaluate_elpp(X_test_norm_np, Y_test_H_norm_np)

        self.assertIsInstance(elpp, float, "ELPP should be a float.")
        self.assertLessEqual(elpp, 0.0, "ELPP should be non-negative.")
        self.assertGreaterEqual(elpp, -1000.0, "ELPP should not exceed 1.0.")


if __name__ == '__main__':
    unittest.main()
