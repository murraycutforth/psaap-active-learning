"""This is an example problem based on the current PSAAP data.

We train a BFGPC model and then use the posterior predictive probabilities from this as a test problem.

Functions are provided to sample an outcome at a given set of test points, and to get the true underlying probabilities
at a given set of test points.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch

from src.models.bfgpc import BFGPC_ELBO
from src.utils_plotting import plot_bfgpc_predictions_two_axes
from src.paths import get_project_root


def load_all_lf_data():
    filepaths = ['01_2M(1).npz', '02_2M(1).npz']
    all_xs = []
    all_ys = []
    for filepath in filepaths:
        data = np.load(get_project_root() / "data" / filepath)
        x = data["xis"][:, [2, 4]].astype(np.float32)
        y = data["chis"]
        all_xs.append(x)
        all_ys.append(y)

    xs = np.concatenate(all_xs, axis=0)
    ys = np.concatenate(all_ys, axis=0)

    return xs / np.array([[15, 100]]), ys


def load_all_hf_data():
    filepaths = ['01_15M(1).npz', 'test_01_15M.npz', 'test_02_15M.npz']
    all_xs = []
    all_ys = []
    for filepath in filepaths:
        data = np.load(get_project_root() / "data" / filepath)
        x = data["xis"][:, [2, 4]].astype(np.float32)
        y = data["chis"]
        all_xs.append(x)
        all_ys.append(y)

    xs = np.concatenate(all_xs, axis=0)
    ys = np.concatenate(all_ys, axis=0)

    return xs / np.array([[15, 100]]), ys


def train_model():
    np.random.seed(0)
    torch.manual_seed(0)

    X_lf, Y_lf = load_all_lf_data()
    X_hf, Y_hf = load_all_hf_data()

    X_lf = torch.from_numpy(X_lf).float()
    X_hf = torch.from_numpy(X_hf).float()
    Y_lf = torch.from_numpy(Y_lf).float()
    Y_hf = torch.from_numpy(Y_hf).float()

    model = BFGPC_ELBO(X_lf, X_hf)
    model.train_model(X_lf, Y_lf, X_hf, Y_hf, n_epochs=100)
    return model


_model = train_model()


def sample_HF_outcomes(X_test):
    probs = _model.predict_hf_prob(X_test)
    outcomes = np.random.binomial(n=1, p=probs)
    return outcomes


def sample_LF_outcomes(X_test):
    probs = _model.predict_lf_prob(X_test)
    outcomes = np.random.binomial(n=1, p=probs)
    return outcomes


def get_HF_probs(X_test):
    probs = _model.predict_hf_prob(X_test)
    return probs


def get_LF_probs(X_test):
    probs = _model.predict_lf_prob(X_test)
    return probs
