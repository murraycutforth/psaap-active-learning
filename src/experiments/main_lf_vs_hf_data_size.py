"""For linear and nonlinear toy problems, assess the performance as a function of (LF data size, HF data size)
Output is a heatmap.
"""
import time

import numpy as np
import torch
from matplotlib import colors
from pyDOE import lhs
import matplotlib.pyplot as plt

from src.problems.toy_example import create_smooth_change_nonlinear, create_smooth_change_linear
from src.models.bfgpc import BFGPC_ELBO
from src.paths import get_project_root

#DATASET = 'Toy Nonlinear'
DATASET = 'Toy Linear'

def main():
    if DATASET == 'Toy Linear':
        linear_low_f1, linear_high_f1, p_LF_toy, p_HF_toy = create_smooth_change_linear()
    elif DATASET == 'Toy Nonlinear':
        linear_low_f1, linear_high_f1, p_LF_toy, p_HF_toy = create_smooth_change_nonlinear()
    else:
        raise ValueError('Unknown dataset')

    def sampling_function_L(X_normalized):  # Expects N x 2 normalized input
        Y_linear_low_grid, _ = linear_low_f1(X_normalized, reps=1)
        Y_linear_high_grid = Y_linear_low_grid.mean(axis=0)
        return Y_linear_high_grid

    def sampling_function_H(X_normalized):
        Y_linear_high_grid, _ = linear_high_f1(X_normalized, reps=1)
        Y_linear_high_grid = Y_linear_high_grid.mean(axis=0)
        return Y_linear_high_grid

    output_dir = get_project_root() / 'output' / 'lf_vs_hf_data_size' / DATASET
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)

    N_L_values = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    N_H_values = [4, 8, 16, 32, 64, 128, 256, 512]

    N_test = 10_000
    lr = 0.1
    l2_reg = 0.01
    N_reps = 3

    num_nl = len(N_L_values)
    num_nh = len(N_H_values)

    elpps = np.full((num_nl, num_nh), np.nan)

    # Generate common test data once
    X_test_norm_np = lhs(2, N_test)
    Y_test_H_norm_np = sampling_function_H(X_test_norm_np)

    total_runs = num_nl * num_nh
    current_run = 0

    for i, n_l in enumerate(N_L_values):
        for j, n_h in enumerate(N_H_values):
            print(f"\n--- Running sweep: N_L={n_l}, N_H={n_h} ({current_run}/{total_runs}) ---")
            current_run += 1
            start_time = time.time()
            _elpps = []

            for k in range(N_reps):

                # Generate training data for this specific configuration
                X_L_init_norm_np = lhs(2, n_l)
                Y_L_init_norm_np = sampling_function_L(X_L_init_norm_np)

                X_H_init_norm_np = lhs(2, n_h)
                Y_H_init_norm_np = sampling_function_H(X_H_init_norm_np)

                X_L_train = torch.tensor(X_L_init_norm_np, dtype=torch.float32)
                Y_L_train = torch.tensor(Y_L_init_norm_np, dtype=torch.float32)
                X_H_train = torch.tensor(X_H_init_norm_np, dtype=torch.float32)
                Y_H_train = torch.tensor(Y_H_init_norm_np, dtype=torch.float32)

                model = BFGPC_ELBO(l2_reg_lambda=l2_reg, n_inducing_pts=max(len(X_L_train), len(X_H_train)) // 2)
                model.train_model(X_L_train, Y_L_train, X_H_train, Y_H_train,
                              lr=lr, n_epochs=max(len(X_L_train), len(X_H_train)) // 2)

                elpp = model.evaluate_elpp(X_test_norm_np, Y_test_H_norm_np)
                _elpps.append(elpp)

            elpps[i, j] = np.mean(_elpps)
            elapsed_time = time.time() - start_time
            print(f"Time for this run: {elapsed_time:.2f} seconds")

    # Plotting the results
    nl_labels = [str(n) for n in N_L_values]
    nh_labels = [str(n) for n in N_H_values]

    plot_heatmap(elpps, "ELPP",
             xlabel="Number of High-Fidelity Points",
             ylabel="Number of Low-Fidelity Points",
             xticklabels=nh_labels, yticklabels=nl_labels,
             figname=output_dir / f"elpp_varepochs_l2{l2_reg}.png",
             val_fmt="{x:.3f}")


def plot_heatmap(data, title, xlabel, ylabel, xticklabels, yticklabels, figname, cmap="viridis_r", val_fmt="{x:.2f}"):
    """Helper function to plot a heatmap."""
    data = data * -1

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(data, cmap=cmap, aspect='auto', norm=colors.LogNorm(vmin=data.min(), vmax=data.max()))

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(xticklabels)))
    ax.set_yticks(np.arange(len(yticklabels)))
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(yticklabels)):
        for j in range(len(xticklabels)):
            val = -1 * data[i, j]
            if not np.isnan(val):
                ax.text(j, i, val_fmt.format(x=val), ha="center", va="center")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    plt.savefig(figname)
    plt.close(fig)
    print(f"Saved heatmap: {figname}")

if __name__ == '__main__':
    main()

