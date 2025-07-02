"""
Just train and evaluate ELPP for a single model, no active learning
"""

import datetime
import logging
import os

from tqdm import tqdm
import pandas as pd
import pyDOE
import torch
from matplotlib import pyplot as plt

from src.active_learning.util_classes import BiFidelityDataset, ALExperimentConfig
from src.bfgpc import BFGPC_ELBO
from src.paths import get_project_root
from src.toy_example import create_smooth_change_linear, create_smooth_change_nonlinear
from src.utils_plotting import plot_bf_training_data, plot_bfgpc_predictions_two_axes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

linear_low_f1, linear_high_f1, p_LF_toy, p_HF_toy = create_smooth_change_nonlinear()

def sampling_function_L(X_normalized):  # Expects N x 2 normalized input
    Y_linear_low_grid, probs_linear_low_grid = linear_low_f1(X_normalized, reps=1)
    Y_linear_high_grid = Y_linear_low_grid.mean(axis=0)
    return Y_linear_high_grid


def sampling_function_H(X_normalized):
    Y_linear_high_grid, probs_linear_high_grid = linear_high_f1(X_normalized, reps=1)
    Y_linear_high_grid = Y_linear_high_grid.mean(axis=0)
    return Y_linear_high_grid


def main():
    seed = 42


    dataset = BiFidelityDataset(sample_LF=sampling_function_L, sample_HF=sampling_function_H,
                                true_p_LF=p_LF_toy, true_p_HF=p_HF_toy,
                                name='ToyNonLinear', c_LF=0.1, c_HF=1.0)

    base_config = ALExperimentConfig(
        N_L_init=500,
        N_H_init=500,
        train_epochs=100,
        train_lr=0.1,
        N_test=10_000,
        N_reps=10,
    )

    X_test = pyDOE.lhs(2, samples=base_config.N_test)
    Y_test = dataset.sample_HF(X_test)

    outdir = get_project_root() / "output" / "l2_vs_elpp" / f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{dataset.name}"
    outdir.mkdir(parents=True, exist_ok=True)

    metrics = {"ELPP": [], "l2": []}
    for rep in tqdm(range(base_config.N_reps)):
        models = [
            BFGPC_ELBO(l2_reg_lambda=1.0),
            BFGPC_ELBO(l2_reg_lambda=0.1),
            BFGPC_ELBO(l2_reg_lambda=0.01),
            BFGPC_ELBO(l2_reg_lambda=0.001),
            BFGPC_ELBO(l2_reg_lambda=0.0),
        ]

        for model in models:
            X_L_train = pyDOE.lhs(2, base_config.N_L_init)
            Y_L_train = dataset.sample_LF(X_L_train)

            X_H_train = pyDOE.lhs(2, base_config.N_H_init)
            Y_H_train = dataset.sample_HF(X_H_train)

            model.train_model(X_L_train, Y_L_train, X_H_train, Y_H_train, base_config.train_lr, base_config.train_epochs)
            elpp = model.evaluate_elpp(X_test, Y_test)

            logger.debug(f"ELPP = {elpp:.4f}")

            #plot_bfgpc_predictions_two_axes(model=model, true_p_LF=dataset.true_p_LF,
            #                                true_p_HF=dataset.true_p_HF,
            #                                outpath=outdir / f"model_predictions_{rep}.png")

            metrics["ELPP"].append(elpp)
            metrics["l2"].append(model.l2_reg_lambda)

    logger.info("All experiments finished.")

    df_metrics = pd.DataFrame(metrics)

    # Get the unique l2 values in a sorted order for the x-axis
    l2_values = sorted(df_metrics['l2'].unique())

    # Create a list of ELPP values for each l2 group
    # This is the data format that ax.boxplot() expects
    elpp_data_grouped = [df_metrics[df_metrics['l2'] == val]['ELPP'].values for val in l2_values]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=200)

    # Create the boxplot
    ax.boxplot(elpp_data_grouped)

    # Set the labels for the x-axis ticks
    ax.set_xticklabels([str(val) for val in l2_values], rotation=45, ha="right")

    # Set titles and labels
    ax.set_title(f"Model Performance vs. L2 Regularization Strength\n({base_config.N_reps} Repetitions)")
    ax.set_xlabel("L2 Regularization Lambda")
    ax.set_ylabel("Expected Log Predictive Probability (ELPP)")
    fig.tight_layout()

    # Save the figure to the output directory before showing
    plt.show()


    # Write results and config

    config_dict = {}
    config_dict["dataset_name"] = dataset.name
    config_dict["N_L_init"] = base_config.N_L_init
    config_dict["N_H_init"] = base_config.N_H_init
    config_dict["N_test"] = base_config.N_test
    config_dict["N_reps"] = base_config.N_reps
    config_dict["train_epochs"] = base_config.train_epochs
    config_dict["train_lr"] = base_config.train_lr
    pd.DataFrame([config_dict]).to_csv(os.path.join(outdir, "experiment_config.csv"), index=False)
    pd.DataFrame(metrics).to_csv(os.path.join(outdir, "metrics.csv"), index=False)


if __name__ == '__main__':
    main()
