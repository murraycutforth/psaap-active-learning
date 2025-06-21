# test_mfgp_classifier_sweep.py
import unittest
from pathlib import Path
import os
import time

import numpy as np
import torch
from pyDOE import lhs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from src.bfgpc import BFGPC_ELBO


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


def plot_heatmap(data, title, xlabel, ylabel, xticklabels, yticklabels, figname, cmap="viridis", val_fmt="{x:.2f}"):
    """Helper function to plot a heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(data, cmap=cmap, aspect='auto')

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(xticklabels)))
    ax.set_yticks(np.arange(len(yticklabels)))
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(yticklabels)):
        for j in range(len(xticklabels)):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, val_fmt.format(x=val), ha="center", va="center",
                        color="w" if im.norm(val) < 0.5 else "black")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    plt.savefig(figname)
    plt.close(fig)
    print(f"Saved heatmap: {figname}")


class TestMFGPClassifierSweep(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up for all tests in this class."""
        cls.output_dir = Path("output_plots_bfgpc_sweep_test")
        cls.output_dir.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(42)
        np.random.seed(42)

        cls.N_L_values = [4, 8, 16, 32, 64, 128, 256, 512]
        cls.N_H_values = [4, 8, 16, 32, 64, 128, 256, 512]

        cls.N_test = 1000
        cls.n_epochs = 1000  # Can be increased for better convergence, e.g., 200-500
        cls.lhs_iterations = 10
        cls.lr = 0.01
        cls.initial_rho = 1.0

    def get_kernel_params(self, model):
        """
        Extracts kernel hyperparameters from the trained model.
        This function needs to be adapted to the specific structure of your BFGPC_ELBO.
        Returns a dictionary of parameters.
        """
        params = {}
        try:
            # Low-fidelity GP (gp_L)
            if hasattr(model, 'lf_model') and model.lf_model is not None:
                if hasattr(model.lf_model, 'covar_module'):
                    covar_L = model.lf_model.covar_module
                    if hasattr(covar_L, 'base_kernel') and hasattr(covar_L.base_kernel, 'lengthscale'):
                        ls_L = covar_L.base_kernel.lengthscale.detach().squeeze().cpu().numpy()
                        params['L_lengthscale_dim1'] = float(ls_L[0]) if ls_L.ndim > 0 else float(ls_L)
                        if ls_L.ndim > 0 and len(ls_L) > 1:
                            params['L_lengthscale_dim2'] = float(ls_L[1])
                    if hasattr(covar_L, 'outputscale'):
                        params['L_outputscale'] = float(covar_L.outputscale.detach().cpu().numpy())

            # Delta GP (gp_delta)
            if hasattr(model, 'delta_model') and model.delta_model is not None:
                if hasattr(model.delta_model, 'covar_module'):
                    covar_delta = model.delta_model.covar_module
                    if hasattr(covar_delta, 'base_kernel') and hasattr(covar_delta.base_kernel, 'lengthscale'):
                        # Delta lengthscale might be multi-dimensional (e.g., for X_H and for rho*Y_L(X_H))
                        ls_delta = covar_delta.base_kernel.lengthscale.detach().squeeze().cpu().numpy()
                        params['delta_lengthscale_dim1'] = float(ls_delta[0]) if ls_delta.ndim > 0 else float(ls_delta)
                        if ls_delta.ndim > 0 and len(ls_delta) > 1:
                            params['delta_lengthscale_dim2'] = float(ls_delta[1])
                        # Add more if delta GP input is higher dimensional
                    if hasattr(covar_delta, 'outputscale'):
                        params['delta_outputscale'] = float(covar_delta.outputscale.detach().cpu().numpy())

            # Rho parameter (if it's a learned scalar parameter)
            if hasattr(model, 'rho'):
                if isinstance(model.rho, torch.nn.Parameter):
                    params['rho'] = float(model.rho.item())
                elif isinstance(model.rho, torch.Tensor):
                    params['rho'] = float(model.rho.detach().cpu().numpy())  # if just a tensor
                # If rho is a GP itself, extraction would be more complex

        except AttributeError as e:
            print(f"Warning: Could not extract some hyperparameters: {e}")
        except Exception as e:
            print(f"Warning: Error extracting hyperparameters: {e}")
        return params

    def test_parameter_sweep_and_heatmaps(self):
        """
        Performs a sweep over N_L and N_H, trains models,
        evaluates accuracy, and extracts hyperparameters.
        Then, plots results as heatmaps.
        """
        num_nl = len(self.N_L_values)
        num_nh = len(self.N_H_values)

        accuracies = np.full((num_nl, num_nh), np.nan)
        elpps = np.full((num_nl, num_nh), np.nan)

        # Initialize dictionaries to store various hyperparameters
        # We'll populate keys dynamically based on what get_kernel_params returns for the first run
        hyperparam_results = {}
        first_run_params_collected = False

        # Generate common test data once
        X_test_norm_np = lhs(2, self.N_test, criterion='maximin', iterations=self.lhs_iterations)
        Y_test_H_norm_np = sampling_function_H(X_test_norm_np)

        total_runs = num_nl * num_nh
        current_run = 0

        for i, n_l in enumerate(self.N_L_values):
            for j, n_h in enumerate(self.N_H_values):
                current_run += 1
                print(f"\n--- Running sweep: N_L={n_l}, N_H={n_h} ({current_run}/{total_runs}) ---")

                start_time = time.time()

                # Set seeds for this specific run for reproducibility if an error occurs mid-sweep
                torch.manual_seed(42 + i * num_nh + j)
                np.random.seed(42 + i * num_nh + j)

                # Generate training data for this specific configuration
                X_L_init_norm_np = lhs(2, n_l, criterion='maximin', iterations=self.lhs_iterations)
                Y_L_init_norm_np = sampling_function_L(X_L_init_norm_np)

                # Ensure X_H is a subset of or sampled carefully w.r.t X_L if required by model
                # For this general test, we sample X_H independently.
                # If X_H must be a subset of X_L locations, this needs adjustment.
                X_H_init_norm_np = lhs(2, n_h, criterion='maximin', iterations=self.lhs_iterations)
                Y_H_init_norm_np = sampling_function_H(X_H_init_norm_np)

                X_L_train = torch.tensor(X_L_init_norm_np, dtype=torch.float32)
                Y_L_train = torch.tensor(Y_L_init_norm_np, dtype=torch.float32)
                X_H_train = torch.tensor(X_H_init_norm_np, dtype=torch.float32)
                Y_H_train = torch.tensor(Y_H_init_norm_np, dtype=torch.float32)

                # Initialize and train the model
                # Ensure X_H_train is passed to BFGPC_ELBO constructor if it's used for NARGP structure
                model = BFGPC_ELBO(X_L_train, X_H_train, initial_rho=self.initial_rho)
                try:
                    model.train_model(X_L_train, Y_L_train, X_H_train, Y_H_train,
                                      lr=self.lr, n_epochs=self.n_epochs)

                    # Evaluate accuracy
                    acc = model.evaluate_accuracy(X_test_norm_np, Y_test_H_norm_np)
                    accuracies[i, j] = acc
                    print(f"N_L={n_l}, N_H={n_h} -> Accuracy: {acc:.4f}")

                    # Evaluate elpp
                    elpp = model.evaluate_elpp(X_test_norm_np, Y_test_H_norm_np)
                    elpps[i, j] = elpp
                    print(f"N_L={n_l}, N_H={n_h} -> ELPP: {elpp:.4f}")

                    # Extract hyperparameters
                    current_params = self.get_kernel_params(model)
                    if not first_run_params_collected and current_params:
                        for key in current_params.keys():
                            hyperparam_results[key] = np.full((num_nl, num_nh), np.nan)
                        first_run_params_collected = True

                    for key, value in current_params.items():
                        if key in hyperparam_results:
                            hyperparam_results[key][i, j] = value
                        else:  # Should not happen if first_run_params_collected logic is right
                            print(
                                f"Warning: New hyperparameter key '{key}' found after first run. It might not be fully populated.")
                            hyperparam_results[key] = np.full((num_nl, num_nh), np.nan)
                            hyperparam_results[key][i, j] = value


                except Exception as e:
                    print(f"ERROR during training/evaluation for N_L={n_l}, N_H={n_h}: {e}")
                    # Accuracies and params will remain NaN for this cell

                elapsed_time = time.time() - start_time
                print(f"Time for this run: {elapsed_time:.2f} seconds")

        # Plotting the results
        nl_labels = [str(n) for n in self.N_L_values]
        nh_labels = [str(n) for n in self.N_H_values]

        plot_heatmap(accuracies, "Model Accuracy",
                     xlabel="Number of High-Fidelity Points (N_H)",
                     ylabel="Number of Low-Fidelity Points (N_L)",
                     xticklabels=nh_labels, yticklabels=nl_labels,
                     figname=self.output_dir / "heatmap_accuracy.png",
                     val_fmt="{x:.3f}")

        plot_heatmap(elpps, "ELPP ELBO",
                     xlabel="Number of High-Fidelity Points (N_H)",
                     ylabel="Number of Low-Fidelity Points (N_L)",
                     xticklabels=nl_labels, yticklabels=nl_labels,
                     figname=self.output_dir / "heatmap_elpp.png",
                     val_fmt="{x:.3f}")

        for param_name, param_data in hyperparam_results.items():
            plot_heatmap(param_data, f"{param_name.replace('_', ' ').title()}",
                         xlabel="Number of High-Fidelity Points (N_H)",
                         ylabel="Number of Low-Fidelity Points (N_L)",
                         xticklabels=nh_labels, yticklabels=nl_labels,
                         figname=self.output_dir / f"heatmap_{param_name}.png",
                         val_fmt="{x:.2e}" if "scale" in param_name else "{x:.2f}")  # scientific for scales

        # Basic assertion: check that at least some accuracies were computed (not all NaN)
        # This is a weak assertion, mainly ensuring the process ran.
        self.assertFalse(np.all(np.isnan(accuracies)), "All accuracy computations failed or were skipped.")
        if first_run_params_collected:
            for param_name, param_data in hyperparam_results.items():
                self.assertFalse(np.all(np.isnan(param_data)),
                                 f"All computations for {param_name} failed or were skipped.")


if __name__ == '__main__':
    unittest.main()