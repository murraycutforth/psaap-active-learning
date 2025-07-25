import datetime
import os
import logging

import numpy as np
import pandas as pd
import pyDOE
import torch

from src.active_learning.util_classes import BiFidelityModel, BiFidelityDataset, ALExperimentConfig
from src.batch_al_strategies.base import BiFidelityBatchALStrategy
from src.models.bfgpc import BFGPC_ELBO
from src.utils_plotting import plot_bf_training_data, plot_bfgpc_predictions_two_axes, plot_active_learning_training_data, plot_al_summary_from_dataframe_mpl
from src.paths import get_project_root


class ALExperimentRunner():
    """Orchestrate AL loop, record results
    """
    def __init__(self, dataset: BiFidelityDataset, al_strategy: BiFidelityBatchALStrategy, config: ALExperimentConfig):
        self.dataset = dataset
        self.al_strategy = al_strategy
        self.config = config
        self.strategy_name_str = str(self.al_strategy)

        self.outdir = get_project_root() / "output" / "active_learning" / self.dataset.name / f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.strategy_name_str}"
        self.outdir.mkdir(parents=True)

        # Setup logging
        self.logger = logging.getLogger(f"{__name__}")
        if not self.logger.hasHandlers():  # Avoid adding multiple handlers if re-instantiated
            self.logger.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            # File handler
            fh = logging.FileHandler(os.path.join(self.outdir, "experiment.log"))
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
            # Console handler
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

        self.logger.info(f"Experiment runner initialized. Output directory: {self.outdir}")
        self.logger.info(f"Config: {self.config}")
        self.logger.info(f"Dataset: {self.dataset.name}, LF cost: {self.dataset.c_LF}, HF cost: {self.dataset.c_HF}")
        self.logger.info(f"Strategy: {self.strategy_name_str}")

    def _generate_lhs_samples(self, n_samples: int) -> np.ndarray:
        assert n_samples > 0
        # Use the global seed for LHS if specific random_state is not used by all criteria.
        samples_unit_hypercube = pyDOE.lhs(2, samples=n_samples)
        scaled_samples = np.zeros_like(samples_unit_hypercube)
        for i in range(2):
            min_val, max_val = self.config.domain_bounds[i]
            scaled_samples[:, i] = samples_unit_hypercube[:, i] * (max_val - min_val) + min_val
        return scaled_samples

    def run_experiment(self):
        results_history = []

        self.logger.info("Starting experiment run...")

        # 1. Generate fixed Test Data for ELPP evaluation
        np.random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)

        X_test = self._generate_lhs_samples(self.config.N_test)
        Y_test = self.dataset.sample_HF(X_test)

        scale = 20 * (1 - 0.75 * X_test[:, 0])
        p_HF_test_true = self.dataset.true_p_HF(X_test, scale)
        self.logger.info(f"Generated {self.config.N_test} test points for ELPP evaluation.")

        # Outer loop, repeat for uncertainties in AL method comparison
        for run_iter in range(self.config.N_reps):
            rep_outdir = self.outdir / f"rep_{run_iter:02d}"
            rep_outdir.mkdir()

            all_X_LF = []
            all_X_HF = []

            # 2. Initial Data
            np.random.seed(run_iter)  # All methods get the same initial training data
            torch.manual_seed(run_iter)

            X_L_train = self._generate_lhs_samples(self.config.N_L_init)
            Y_L_train = self.dataset.sample_LF(X_L_train) if self.config.N_L_init > 0 else np.empty((0, 1))

            X_H_train = self._generate_lhs_samples(self.config.N_H_init)
            Y_H_train = self.dataset.sample_HF(X_H_train) if self.config.N_H_init > 0 else np.empty((0, 1))

            initial_cost = (self.config.N_L_init * self.dataset.c_LF +
                            self.config.N_H_init * self.dataset.c_HF)

            current_total_cost = initial_cost

            # 4. Plot initial data
            plot_bf_training_data(
                X_LF=X_L_train, Y_LF=Y_L_train,
                X_HF=X_H_train, Y_HF=Y_H_train,
                outpath=rep_outdir / f"data_plot_0.png")

            # 5. Initial Model Training and Evaluation (Round 0)
            self.logger.info("Training initial model...")
            model = BFGPC_ELBO(**self.config.model_args)
            model.train_model(X_L_train, Y_L_train, X_H_train, Y_H_train, self.config.train_lr, self.config.train_epochs)

            elpp_round_0 = model.evaluate_elpp(X_test, Y_test)
            self.logger.info(f"Round 0: ELPP = {elpp_round_0:.4f}, Cost = {current_total_cost:.2f}")

            pred_mean_p_HF = model.predict_hf_prob(X_test)
            mse = np.mean((pred_mean_p_HF - p_HF_test_true) ** 2)

            plot_bfgpc_predictions_two_axes(model=model, true_p_LF=self.dataset.true_p_LF, true_p_HF=self.dataset.true_p_HF,
                                            outpath=rep_outdir / f"model_predictions_0.png")

            results_history.append({
                "repeat": run_iter,
                "round": 0,
                "cumulative_cost": current_total_cost,
                "elpp": elpp_round_0,
                "mse": mse,
                "lf_queried_this_round": self.config.N_L_init,
                "hf_queried_this_round": self.config.N_H_init,
                "total_lf_samples": X_L_train.shape[0],
                "total_hf_samples": X_H_train.shape[0]
            })

            # 6. Active Learning Loop
            num_al_rounds = len(self.config.cost_constraints)
            for i_round in range(num_al_rounds):
                budget_for_this_step = self.config.cost_constraints[i_round]
                al_round_num = i_round + 1

                self.logger.info(f"\n--- Starting AL Round {al_round_num}/{num_al_rounds} ---")
                self.logger.info(f"Current total cost: {current_total_cost:.2f}")
                self.logger.info(f"Budget for this step: {budget_for_this_step:.2f}")


                # Get batch from strategy
                self.logger.info(f"Querying strategy for batch with budget {budget_for_this_step:.2f}...")
                new_X_L, new_X_H = self.al_strategy.select_batch(
                    config=self.config,
                    current_model_trained=model,  # Pass the currently trained model instance
                    budget_this_step=budget_for_this_step
                )
                self.logger.info(f"Strategy selected {len(new_X_L)} LF and {len(new_X_H)} HF points.")


                cost_this_batch = 0

                if len(new_X_L) > 0:
                    cost_this_batch += len(new_X_L) * self.dataset.c_LF
                    new_Y_L = self.dataset.sample_LF(new_X_L)

                    # Update training data
                    X_L_train = np.concatenate([X_L_train, new_X_L], axis=0) if X_L_train.size > 0 else new_X_L
                    Y_L_train = np.concatenate([Y_L_train, new_Y_L], axis=0) if Y_L_train.size > 0 else new_Y_L

                    all_X_LF.append(new_X_L)

                if len(new_X_H) > 0:
                    cost_this_batch += len(new_X_H) * self.dataset.c_HF
                    new_Y_H = self.dataset.sample_HF(new_X_H)

                    # Update training data
                    X_H_train = np.concatenate([X_H_train, new_X_H], axis=0) if X_H_train.size > 0 else new_X_H
                    Y_H_train = np.concatenate([Y_H_train, new_Y_H], axis=0) if Y_H_train.size > 0 else new_Y_H

                    all_X_HF.append(new_X_H)

                current_total_cost += cost_this_batch

                # Retrain model
                self.logger.info(f"Retraining model with new dataset of size {X_L_train.shape[0]} and {X_H_train.shape[0]}.")
                model = BFGPC_ELBO(**self.config.model_args)
                model.train_model(X_L_train, Y_L_train, X_H_train, Y_H_train, self.config.train_lr, self.config.train_epochs)

                # Evaluate ELPP
                current_elpp = model.evaluate_elpp(X_test, Y_test)
                self.logger.info(
                    f"Round {al_round_num}: ELPP = {current_elpp:.4f}, Cost Incurred this round = {cost_this_batch:.2f}, Cumulative Cost = {current_total_cost:.2f}")

                # Evaluate MSE of mean probability
                pred_mean_p_HF = model.predict_hf_prob(X_test)
                mse = np.mean((pred_mean_p_HF - p_HF_test_true)**2)
                self.logger.info(f"MSE = {mse:.4f}")

                results_history.append({
                    "repeat": run_iter,
                    "round": al_round_num,
                    "cumulative_cost": current_total_cost,
                    "elpp": current_elpp,
                    "mse": mse,
                    "lf_queried_this_round": len(new_X_L),
                    "hf_queried_this_round": len(new_X_H),
                    "total_lf_samples": X_L_train.shape[0],
                    "total_hf_samples": X_H_train.shape[0]
                })

                plot_bf_training_data(
                    X_LF=X_L_train, Y_LF=Y_L_train,
                    X_HF=X_H_train, Y_HF=Y_H_train,
                    outpath=rep_outdir / f"data_plot_{al_round_num}.png")

                # Plot current model predictions
                plot_bfgpc_predictions_two_axes(model=model, true_p_LF=self.dataset.true_p_LF, true_p_HF=self.dataset.true_p_HF,
                                                outpath=rep_outdir / f"model_predictions_{al_round_num}.png")

            self.logger.info("AL loop complete")

            # Plot history of training data
            plot_active_learning_training_data(all_X_LF, all_X_HF,
                                               outpath=self.outdir / f"training_data_{run_iter:02d}.png")

        self.logger.info("All repeats complete")

        # 7. Save results to CSV
        results_df = pd.DataFrame(results_history)
        csv_path = self.outdir / "results.csv"
        results_df.to_csv(csv_path, index=False)
        self.logger.info(f"Experiment finished. Results saved to {csv_path}")

        # Plot ELPP history
        plot_al_summary_from_dataframe_mpl(results_df, outpath=self.outdir / f"summary.png")

        # Log config
        config_dict = self.config.__dict__
        config_dict["dataset_name"] = self.dataset.name
        config_dict["dataset_c_LF"] = self.dataset.c_LF
        config_dict["dataset_c_HF"] = self.dataset.c_HF
        config_dict["model_name"] = str(model)
        config_dict["strategy_name"] = self.strategy_name_str
        pd.DataFrame([config_dict]).to_csv(os.path.join(self.outdir, "experiment_config.csv"), index=False)

        return results_df