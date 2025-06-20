import logging

import pyDOE
import torch

from src.active_learning.batch_active_learning_experiments import ALExperimentRunner
from src.active_learning.util_classes import BiFidelityDataset, ALExperimentConfig
from src.bfgpc import BFGPC_ELBO
from src.batch_al_strategies.random_strategy import RandomStrategy
from src.batch_al_strategies.mutual_information_strategy_bmfal import MutualInformationBMFALStrategy
from src.batch_al_strategies.mutual_information_strategy_grid_latents import MutualInformationGridStrategy
from src.batch_al_strategies.mutual_information_strategy_grid_observables import MutualInformationGridStrategyObservables
from src.toy_example import create_smooth_change_linear, create_smooth_change_nonlinear

logging.basicConfig(level=logging.INFO)

linear_low_f1, linear_high_f1, p_LF_toy, p_HF_toy = create_smooth_change_linear()


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

    inducing_points = pyDOE.lhs(2, 256, criterion='maximin', iterations=10)
    model = BFGPC_ELBO(train_x_lf=torch.tensor(inducing_points).float(), train_x_hf=torch.tensor(inducing_points).float())

    dataset = BiFidelityDataset(sample_LF=sampling_function_L, sample_HF=sampling_function_H,
                                true_p_LF=p_LF_toy, true_p_HF=p_HF_toy,
                                name='Toy', c_LF=0.5, c_HF=1.0)

    # strategy = RandomStrategy(model=model, dataset=dataset, seed=seed)
    strategy = MutualInformationBMFALStrategy(model=model, dataset=dataset, seed=seed, N_MC=50)
    # strategy = MutualInformationGridStrategy(model=model, dataset=dataset, seed=seed, N_MC=50)
    #strategy = MutualInformationGridStrategyObservables(model, dataset, seed=seed)

    config = ALExperimentConfig(
        N_L_init=1000,
        N_H_init=1000,
        cost_constraints=[20, 20, 20],
        N_cand_LF=250,
        N_cand_HF=250,
        train_epochs=1500,
    )

    experiment = ALExperimentRunner(model, dataset, strategy, config)
    experiment.run_experiment()


if __name__ == '__main__':
    main()