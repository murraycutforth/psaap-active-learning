import logging

import numpy as np
import matplotlib.pyplot as plt
import gpytorch
import pandas as pd
import torch

import pdb

logger = logging.getLogger(__name__)



def plot_bfgpc_predictions(model, grid_res=100, X_LF=None, Y_LF=None, X_HF=None, Y_HF=None, boundary_HF=None, outpath=None):
    model.eval()

    xi = np.linspace(0, 1, grid_res)
    yi = np.linspace(0, 1, grid_res)
    xx, yy = np.meshgrid(xi, yi)
    grid_points_np = np.vstack([xx.ravel(), yy.ravel()]).T
    grid_points_torch = torch.tensor(grid_points_np, dtype=torch.float32)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predicted_probs_hf_grid = model(grid_points_torch)

    predicted_probs_hf_grid_reshaped = predicted_probs_hf_grid.numpy().reshape(grid_res, grid_res)

    # Plotting results
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(xx, yy, predicted_probs_hf_grid_reshaped, levels=np.linspace(0, 1, 11), cmap='coolwarm',
                           alpha=0.8)
    plt.colorbar(contour, label='Predicted P(Y_H=1)')

    # Plot decision boundary P(Y_H=1) = 0.5
    # plt.contour(xx, yy, predicted_probs_hf_grid_reshaped, levels=[0.5], colors='k', linewidths=2)

    # Plot training data
    if X_LF is not None:
        plt.scatter(X_LF[:, 0], X_LF[:, 1], c=Y_LF, cmap='viridis', marker='s', s=20,
                alpha=0.3, label='LF Training Data (Y_L)')
        plt.scatter(X_HF[:, 0], X_HF[:, 1], c=Y_HF, cmap='viridis', marker='o', s=80,
                edgecolors='k', label='HF Training Data (Y_H)')

    # Plot true HF boundary
    if boundary_HF is not None:
        x_boundary_plot = np.linspace(0, 1, 200)
        y_boundary_h_plot = boundary_HF(x_boundary_plot[:, None])
        plt.plot(x_boundary_plot, y_boundary_h_plot.squeeze(), 'g--', linewidth=2, label='True HF Boundary')

    plt.title('Bi-fidelity GPC Predictions')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend(loc='lower right')
    #plt.xlim(0, 1)
    #plt.ylim(0, 1)

    if outpath is not None:
        plt.savefig(outpath)
    else:
        plt.show()


def plot_bfgpc_predictions_two_axes(model, grid_res=100, X_LF=None, Y_LF=None, X_HF=None, Y_HF=None, true_p_LF=None, true_p_HF=None, outpath=None):
    """Major plot function
    """
    model.eval()

    xi = np.linspace(0, 1, grid_res)
    yi = np.linspace(0, 1, grid_res)
    xx, yy = np.meshgrid(xi, yi)
    grid_points_np = np.vstack([xx.ravel(), yy.ravel()]).T
    grid_points_torch = torch.tensor(grid_points_np, dtype=torch.float32)

    if true_p_LF is not None:
        scale = 20 * (1 - 0.75 * grid_points_np[:, 0])
        true_p_LF = true_p_LF(grid_points_np, scale).reshape(grid_res, grid_res)

    if true_p_HF is not None:
        scale = 20 * (1 - 0.75 * grid_points_np[:, 0])
        true_p_HF = true_p_HF(grid_points_np, scale).reshape(grid_res, grid_res)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # --- LF Predictions ---
        # If your model has a predict_lf method as defined above:
        predicted_probs_lf_grid = model.predict_lf_prob(grid_points_torch)
        # Alternatively, if you don't want to add predict_lf to the model class:
        # lf_latent_output = model.lf_model(grid_points_torch)
        # predicted_probs_lf_grid = model.lf_likelihood(lf_latent_output).mean

        # --- HF Predictions ---
        # This uses the forward method of BFGPC_ELBO as defined above
        predicted_probs_hf_grid = model(grid_points_torch) # model.__call__ which routes to model.forward

        # Predictions of latents
        predicted_f_L = model.predict_f_L(grid_points_torch)
        predicted_f_H = model.predict_f_H(grid_points_torch)

        if hasattr(model, 'predict_delta'):
            predicted_delta = model.predict_delta(grid_points_torch)
        else:
            predicted_delta = None
    
    predicted_probs_lf_grid_reshaped = predicted_probs_lf_grid.reshape(grid_res, grid_res)
    predicted_probs_hf_grid_reshaped = predicted_probs_hf_grid.reshape(grid_res, grid_res)

    predicted_f_L_mean = predicted_f_L.mean.numpy()
    predicted_f_H_mean = predicted_f_H.mean.numpy()
    predicted_f_L_var = predicted_f_L.variance.numpy()
    predicted_f_H_var = predicted_f_H.variance.numpy()


    if predicted_f_L_mean.ndim == 2:
        predicted_f_L_mean = predicted_f_L_mean.mean(axis=0)
    if predicted_f_H_mean.ndim == 2:
        predicted_f_H_mean = predicted_f_H_mean.mean(axis=0)
    if predicted_f_L_var.ndim == 2:
        predicted_f_L_var = predicted_f_L_var.mean(axis=0)
    if predicted_f_H_var.ndim == 2:
        predicted_f_H_var = predicted_f_H_var.mean(axis=0)

    predicted_f_L_mean_grid = predicted_f_L_mean.reshape(grid_res, grid_res)
    predicted_f_H_mean_grid = predicted_f_H_mean.reshape(grid_res, grid_res)
    predicted_f_L_var_grid = predicted_f_L_var.reshape(grid_res, grid_res)
    predicted_f_H_var_grid = predicted_f_H_var.reshape(grid_res, grid_res)

    if predicted_delta is not None:
        predicted_delta_mean_grid = predicted_delta.mean.numpy().reshape(grid_res, grid_res)
        predicted_delta_var_grid = predicted_delta.variance.numpy().reshape(grid_res, grid_res)
    else:
        predicted_delta_mean_grid = np.zeros((grid_res, grid_res))
        predicted_delta_var_grid = np.zeros((grid_res, grid_res))

    # Plotting results
    fig, axes = plt.subplots(3, 3, figsize=(11, 9), dpi=200, sharex=True, sharey=True) # sharex/sharey can be useful

    # --- Plot for LF Predictions (axes[0]) ---
    ax_lf = axes[0, 0]
    contour_lf = ax_lf.contourf(xx, yy, predicted_probs_lf_grid_reshaped, levels=np.linspace(0, 1, 11), cmap='coolwarm',
                                alpha=0.8)
    fig.colorbar(contour_lf, ax=ax_lf, label='  Predicted $P(Y_L=1)$') # Use fig.colorbar for subplots

    if X_LF is not None and Y_LF is not None:
        ax_lf.scatter(X_LF[:, 0], X_LF[:, 1], c=Y_LF, cmap='viridis', marker='s', s=30,
                      alpha=0.3, label='LF Training Data (Y_L)')
    if X_HF is not None and Y_HF is not None: # Also show HF points on LF plot for context
        ax_lf.scatter(X_HF[:, 0], X_HF[:, 1], c=Y_HF, cmap='viridis', marker='o', s=90,
                      edgecolors='grey', label='HF Training Data (Y_H)', alpha=0.4)



    # --- Plot for HF Predictions (axes[1]) ---
    ax_hf = axes[0, 1]
    contour_hf = ax_hf.contourf(xx, yy, predicted_probs_hf_grid_reshaped, levels=np.linspace(0, 1, 11), cmap='coolwarm',
                                alpha=0.8)
    fig.colorbar(contour_hf, ax=ax_hf, label='Predicted $P(Y_H=1)$')

    # Plot training data on HF plot
    if X_LF is not None and Y_LF is not None: # Show LF points on HF plot for context
        ax_hf.scatter(X_LF[:, 0], X_LF[:, 1], c=Y_LF, cmap='viridis', marker='s', s=20,
                      alpha=0.3, label='LF Training Data (Y_L)')
    if X_HF is not None and Y_HF is not None:
        ax_hf.scatter(X_HF[:, 0], X_HF[:, 1], c=Y_HF, cmap='viridis', marker='o', s=80,
                      edgecolors='k', label='HF Training Data (Y_H)', alpha=0.4)

    # Plot true HF boundary
    if true_p_HF is not None:
        ax_hf.contour(xx, yy, predicted_probs_hf_grid_reshaped, levels=[0.25, 0.5, 0.75], colors='k', linewidths=1, linestyles='-')
        ax_hf.contour(xx, yy, true_p_HF, levels=[0.25, 0.5, 0.75], colors='k', linewidths=1, linestyles='--', label='True P_HF')

    if true_p_LF is not None:
        # Plot decision boundary P(Y_L=1) = 0.5 for LF
        ax_lf.contour(xx, yy, predicted_probs_lf_grid_reshaped, levels=[0.25, 0.5, 0.75], colors='k', linewidths=1, linestyles='-')
        ax_lf.contour(xx, yy, true_p_LF, levels=[0.25, 0.5, 0.75], colors='k', linewidths=1, linestyles='--', label='True P_LF')

    # Plot mean of latents
    ax_f_L = axes[1, 0]
    contour = ax_f_L.contourf(xx, yy, predicted_f_L_mean_grid, cmap='coolwarm',alpha=0.8)
    fig.colorbar(contour, ax=ax_f_L, label='Mean $f_L$')
    ax_f_L.set_aspect('equal', adjustable='box')
    ax_f_L.set_title('Mean $f_L$')

    ax_f_H = axes[1, 1]
    contour = ax_f_H.contourf(xx, yy, predicted_f_H_mean_grid, cmap='coolwarm',alpha=0.8)
    fig.colorbar(contour, ax=ax_f_H, label='Mean $f_H$')
    ax_f_H.set_aspect('equal', adjustable='box')
    ax_f_H.set_title('Mean $f_H$')

    # Plot var of latents
    ax_f_L = axes[2, 0]
    contour = ax_f_L.contourf(xx, yy, predicted_f_L_var_grid, cmap='coolwarm',alpha=0.8)
    fig.colorbar(contour, ax=ax_f_L, label='Var $f_L$')
    ax_f_L.set_aspect('equal', adjustable='box')
    ax_f_L.set_title('Var $f_L$')

    ax_f_H = axes[2, 1]
    contour = ax_f_H.contourf(xx, yy, predicted_f_H_var_grid, cmap='coolwarm',alpha=0.8)
    fig.colorbar(contour, ax=ax_f_H, label='Var $f_H$')
    ax_f_H.set_aspect('equal', adjustable='box')
    ax_f_H.set_title('Var $f_H$')

    ax_delta = axes[1, 2]
    contour = ax_delta.contourf(xx, yy, predicted_delta_mean_grid, cmap='coolwarm', alpha=0.8)
    fig.colorbar(contour, ax=ax_delta, label='Mean $\\delta$')
    ax_delta.set_aspect('equal', adjustable='box')
    ax_delta.set_title('Mean $\\delta$')

    ax_delta = axes[2, 2]
    contour = ax_delta.contourf(xx, yy, predicted_delta_var_grid, cmap='coolwarm', alpha=0.8)
    fig.colorbar(contour, ax=ax_delta, label='Var $\\delta$')
    ax_delta.set_aspect('equal', adjustable='box')
    ax_delta.set_title('Var $\\delta$')

    axes[0, 2].axis('off')

    ax_hf.set_title('HF mean predicted probability')
    # ax_hf.set_ylabel('X2') # Only if not sharey
    # ax_hf.legend(loc='lower right')
    ax_hf.set_xlim(0, 1)
    ax_hf.set_ylim(0, 1)
    ax_hf.set_aspect('equal', adjustable='box')

    ax_lf.set_title('LF mean predicted probability')
    # ax_lf.legend(loc='lower right')
    ax_lf.set_xlim(0, 1)
    ax_lf.set_ylim(0, 1)
    ax_lf.set_aspect('equal', adjustable='box')

    plt.suptitle('Bi-fidelity GPC Predictions', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust rect to make space for suptitle

    if outpath:
        plt.savefig(outpath, dpi=300)
        logger.info(f"Plot saved to {outpath}")
        plt.close()
    else:
        plt.show()



def plot_bf_training_data(X_LF, Y_LF, X_HF, Y_HF, boundary_LF=None, boundary_HF=None, outpath=None):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 3), dpi=200)

    axs[0].scatter(X_LF[:, 0], X_LF[:, 1], c=Y_LF, cmap='viridis', marker='s', s=30,
                alpha=0.5, label='Low-fidelity Data')
    axs[1].scatter(X_HF[:, 0], X_HF[:, 1], c=Y_HF, cmap='viridis', marker='o', s=30,
                edgecolors='k', label='High-fidelity Data', alpha=0.5)

    x_boundary_plot = np.linspace(0, 1, 200)

    if boundary_LF is not None:
        y_boundary_l_plot = boundary_LF(x_boundary_plot[:, None])
        y_boundary_h_plot = boundary_HF(x_boundary_plot[:, None])
        plt.plot(x_boundary_plot, y_boundary_l_plot.squeeze(), 'b--', label='True LF Boundary')
        plt.plot(x_boundary_plot, y_boundary_h_plot.squeeze(), 'r--', label='True HF Boundary')

    for ax in axs:
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('$x1$')
        ax.set_ylabel('$x2$')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    if outpath is not None:
        plt.savefig(outpath)
        plt.close()
    else:
        plt.show()


def plot_active_learning_training_data(all_X_LF: list, all_X_HF: list, outpath=None):
    """
    Plots active learning training data across rounds, distinguishing rounds by color.

    Args:
        all_X_LF (list of np.array or torch.Tensor): List where each element is LF data (N_i, 2) for round i.
        all_X_HF (list of np.array or torch.Tensor): List where each element is HF data (M_i, 2) for round i.
        outpath (str, optional): Path to save the plot. If None, shows the plot.
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), dpi=200, sharex=True, sharey=True)  # Increased figsize slightly

    # Define a colormap for rounds
    # Using a perceptually uniform colormap is good. 'viridis', 'plasma', 'magma', 'cividis'
    # Or a qualitative one if you have few rounds: 'tab10', 'Set1', 'Pastel1'
    num_rounds_lf = len(all_X_LF)
    num_rounds_hf = len(all_X_HF)
    # Use the maximum number of rounds to define the color cycle to be consistent
    max_rounds = max(num_rounds_lf, num_rounds_hf, 1)  # Ensure at least 1 for colormap

    # Using a more distinct colormap for rounds
    colors = plt.cm.get_cmap('viridis', max_rounds)  # 'turbo' or 'jet' can be good for distinct colors

    # Plot Low-Fidelity Data
    axs[0].set_title('Low-Fidelity Data Evolution')
    for i, X_LF_round in enumerate(all_X_LF):
        if isinstance(X_LF_round, torch.Tensor):
            X_LF_round_np = X_LF_round.cpu().numpy()
        else:
            X_LF_round_np = X_LF_round

        if X_LF_round_np.shape[0] > 0:  # Only plot if there's data for this round
            # For LF, make initial data slightly different or less prominent if needed
            marker_size = 30 if i > 0 else 40  # Example: slightly larger initial round
            alpha_val = 0.5 if i > 0 else 0.7
            edge_color = None if i > 0 else 'grey'  # Example: initial round with edge

            axs[0].scatter(X_LF_round_np[:, 0], X_LF_round_np[:, 1],
                           color=colors(i / max(1, max_rounds - 1) if max_rounds > 1 else 0.0),  # Normalize color index
                           marker='s', s=marker_size, alpha=alpha_val, edgecolor=edge_color,
                           label=f'Round {i + 1}' if X_LF_round_np.shape[0] > 0 else None)

    axs[0].set_xlabel('X1')
    axs[0].set_ylabel('X2')
    axs[0].set_xlim(0, 1)
    axs[0].set_ylim(0, 1)
    axs[0].legend(fontsize='small', loc='upper right')
    axs[0].set_aspect('equal', adjustable='box')

    # Plot High-Fidelity Data
    axs[1].set_title('High-Fidelity Data Evolution')
    for i, X_HF_round in enumerate(all_X_HF):
        if isinstance(X_HF_round, torch.Tensor):
            X_HF_round_np = X_HF_round.cpu().numpy()
        else:
            X_HF_round_np = X_HF_round

        if X_HF_round_np.shape[0] > 0:  # Only plot if there's data for this round
            marker_size = 80 if i > 0 else 100  # Example
            alpha_val = 0.7 if i > 0 else 0.9

            axs[1].scatter(X_HF_round_np[:, 0], X_HF_round_np[:, 1],
                           color=colors(i / max(1, max_rounds - 1) if max_rounds > 1 else 0.0),  # Normalize color index
                           marker='o', s=marker_size, alpha=alpha_val,
                           edgecolors='k',  # Keep black edge for HF points for visibility
                           label=f'Round {i + 1}' if X_HF_round_np.shape[0] > 0 else None)

    axs[1].set_xlabel('X1')
    # axs[1].set_ylabel('X2') # Y-axis is shared
    axs[1].set_xlim(0, 1)
    axs[1].set_ylim(0, 1)
    axs[1].legend(fontsize='small', loc='upper right')
    axs[1].set_aspect('equal', adjustable='box')

    fig.suptitle('Active Learning Data Acquisition', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make space for suptitle

    if outpath is not None:
        plt.savefig(outpath, bbox_inches='tight', dpi=300)
        plt.close()  # Close the figure after saving
    else:
        plt.show()


def plot_al_summary_from_dataframe_mpl(results_df: pd.DataFrame, outpath: str = None):
    """
    Creates a summary plot from the active learning results dataframe using only Matplotlib.

    The plot will show:
    1. ELPP vs. Cumulative Cost.
    2. Number of HF/LF samples queried vs round

    Args:
        results_df (pd.DataFrame): DataFrame containing the active learning history
                                   with columns like "round", "cumulative_cost", "elpp",
                                   "lf_queried_this_round", "hf_queried_this_round",
                                   "total_lf_samples", "total_hf_samples".
        outpath (str, optional): Path to save the plot. If None, displays the plot.
    """
    if not isinstance(results_df, pd.DataFrame):
        raise TypeError("results_df must be a pandas DataFrame.")

    required_cols = ["repeat", "round", "cumulative_cost", "elpp",
                     "lf_queried_this_round", "hf_queried_this_round",
                     "total_lf_samples", "total_hf_samples"]
    for col in required_cols:
        if col not in results_df.columns:
            raise ValueError(f"DataFrame is missing required column: {col}")

    # Create a figure with multiple subplots
    fig, axs = plt.subplots(1, 2, figsize=(8, 4), dpi=200)
    fig.suptitle('Active Learning Summary', fontsize=18, y=0.995)  # y slightly adjusted

    # --- 1. ELPP vs. Cumulative Cost (Main Performance Metric) ---
    ax = axs[0]
    #ax.plot(grouped["cumulative_cost"].mean(), grouped["elpp"].mean(), marker='o', linestyle='-', color='dodgerblue', mec='b')
    ax.scatter(results_df["round"], results_df["elpp"], marker='o', color='dodgerblue')
    ax.set_xlabel("Round")
    ax.set_ylabel("ELPP")
    ax.set_title("ELPP vs. Round")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.spines['top'].set_visible(False)  # Remove top and right spines for cleaner look
    ax.spines['right'].set_visible(False)

    # --- 2. Number of LF/HF queried per round vs. Round ---
    ax = axs[1]
    ax.scatter(results_df["round"], results_df["lf_queried_this_round"], label='LF Queried')
    ax.scatter(results_df["round"], results_df["hf_queried_this_round"], label='HF Queried')
    ax.set_xlabel("Round")
    ax.set_ylabel("Number of Samples Queried")
    ax.set_title("LF Samples Queried per Round")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
    if len(results_df["round"].unique()) == len(results_df["round"]):
        ax.set_xticks(results_df["round"])
    ax.spines['top'].set_visible(False)  # Remove top and right spines for cleaner look
    ax.spines['right'].set_visible(False)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # rect to make space for suptitle

    if outpath is not None:
        plt.savefig(outpath, bbox_inches='tight', dpi=300)
        print(f"Summary plot saved to {outpath}")
        plt.close(fig)  # Close the figure to free memory
    else:
        plt.show()