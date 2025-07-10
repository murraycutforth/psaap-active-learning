"""
Just train and evaluate ELPP for a single model, no active learning
"""

import datetime
import logging

import pyDOE

from src.active_learning.util_classes import BiFidelityDataset, ALExperimentConfig
from src.models.bfgpc import BFGPC_ELBO
from src.paths import get_project_root
from src.problems.toy_example import create_smooth_change_nonlinear
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
        N_L_init=2000,
        N_H_init=1000,
        train_epochs=100,
        train_lr=0.1,
        N_test=10_000,
    )

    X_test = pyDOE.lhs(2, samples=base_config.N_test)
    Y_test = dataset.sample_HF(X_test)

    outdir = get_project_root() / "output" / "single_model" / f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{dataset.name}"
    outdir.mkdir(parents=True, exist_ok=True)

    model = BFGPC_ELBO()
    X_L_train = pyDOE.lhs(2, base_config.N_L_init)
    Y_L_train = dataset.sample_LF(X_L_train)

    X_H_train = pyDOE.lhs(2, base_config.N_H_init)
    Y_H_train = dataset.sample_HF(X_H_train)

    model.train_model(X_L_train, Y_L_train, X_H_train, Y_H_train, base_config.train_lr, base_config.train_epochs)
    elpp = model.evaluate_elpp(X_test, Y_test)

    print(f"ELPP = {elpp:.4f}")

    plot_bf_training_data(X_L_train, Y_L_train, X_H_train, Y_H_train, outpath=outdir / "data.png")

    plot_bfgpc_predictions_two_axes(model=model, true_p_LF=dataset.true_p_LF,
                                    true_p_HF=dataset.true_p_HF,
                                    outpath=outdir / f"model_predictions.png")


if __name__ == '__main__':
    main()
