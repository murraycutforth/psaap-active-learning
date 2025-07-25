import numpy as np
from scipy.special import expit  # sigmoid function


def f1_low_decision_boundary(x):
    return (np.cos(x * np.pi * 0.5) + 1) / 3 - 0.1


################################################################################
# Linear transformation of low decision boundary
################################################################################
def linear_f1_high_decision_boundary(x):
    # linear transformation of low decision boundary
    return 0.8 * f1_low_decision_boundary(x) + 0.3


# Decision boundary
def create_sharp_change_linear():
    def low_f1(x):
        return x[:, 1] > f1_low_decision_boundary(x[:, 0])

    def high_f1(x):
        return x[:, 1] > linear_f1_high_decision_boundary(x[:, 0])

    return low_f1, high_f1


def create_smooth_change_linear(scale_factor=20):
    # logit_scale: scale of the sigmoid function
    # smooth_scale: scale of the smooth change along the decision boundary
    def low_f1_prob(x, scale):
        decision = f1_low_decision_boundary(x[:, 0])
        logits = scale * (x[:, 1] - decision)
        return expit(logits)

    def high_f1_prob(x, scale):
        decision = linear_f1_high_decision_boundary(x[:, 0])
        logits = scale * (x[:, 1] - decision)
        return expit(logits)

    # Stochastic label sampling (Bernoulli draw)
    def low_f1(x, reps=10):
        # change scale along the decision boundary
        scale = scale_factor * (1 - 0.75 * x[:, 0])
        probs = low_f1_prob(x, scale)
        return np.random.binomial(n=1, p=probs, size=(reps, x.shape[0])), probs

    def high_f1(x, reps=10):
        scale = scale_factor * (1 - 0.75 * x[:, 0])
        probs = high_f1_prob(x, scale)
        return np.random.binomial(n=1, p=probs, size=(reps, x.shape[0])), probs

    return low_f1, high_f1, low_f1_prob, high_f1_prob


################################################################################
# Nonlinear transformation of low decision boundary
################################################################################
def nonlinear_f1_high_decision_boundary(x):
    lf = f1_low_decision_boundary(x)
    return lf + 0.2 * np.sin(3 * np.pi * x) * (1 - x) + 0.1


def create_sharp_change_nonlinear():
    def low_f1(x):
        return x[:, 1] > f1_low_decision_boundary(x[:, 0])

    def high_f1(x):
        return x[:, 1] > nonlinear_f1_high_decision_boundary(x[:, 0])

    return low_f1, high_f1


def create_smooth_change_nonlinear(scale_factor=20):
    # logit_scale: scale of the sigmoid function
    # smooth_scale: scale of the smooth change along the decision boundary
    def low_f1_prob(x, scale):
        decision = f1_low_decision_boundary(x[:, 0])
        logits = scale * (x[:, 1] - decision)
        return expit(logits)

    def high_f1_prob(x, scale):
        decision = nonlinear_f1_high_decision_boundary(x[:, 0])
        logits = scale * (x[:, 1] - decision)
        return expit(logits)

    def low_f1(x, reps=10):
        scale = scale_factor * (1 - 0.75 * x[:, 0])
        probs = low_f1_prob(x, scale)
        return np.random.binomial(n=1, p=probs, size=(reps, x.shape[0])), probs

    def high_f1(x, reps=10):
        scale = scale_factor * (1 - 0.75 * x[:, 0])
        probs = high_f1_prob(x, scale)
        return np.random.binomial(n=1, p=probs, size=(reps, x.shape[0])), probs

    return low_f1, high_f1, low_f1_prob, high_f1_prob


