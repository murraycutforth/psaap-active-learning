#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 16:30:49 2018 (Updated for Python 3 and PyTorch)

@author: fsc
"""
import numpy as np
import sys
from matplotlib import pyplot as plt
import torch  # Added for PyTorch support


def invlogit(x, eps=sys.float_info.epsilon):
    """
    Computes the inverse logit (sigmoid) function, adding/subtracting epsilon
    for numerical stability to avoid exact 0 or 1.

    Handles torch.Tensor, np.ndarray, and Python scalars.
    The provided 'eps' is cast to the dtype of 'x' if 'x' is an array/tensor.
    """
    if isinstance(x, torch.Tensor):
        # Ensure eps is a tensor of the same dtype and device as x
        _eps = torch.tensor(eps, dtype=x.dtype, device=x.device)
        return (1.0 + 2.0 * _eps) / (1.0 + torch.exp(-x)) + _eps
    elif isinstance(x, np.ndarray):
        # Ensure eps is a numpy array of the same dtype as x
        _eps = np.array(eps, dtype=x.dtype)
        return (1.0 + 2.0 * _eps) / (1.0 + np.exp(-x)) + _eps
    else:  # Python scalar (float, int)
        # np.exp can handle scalars; eps is already a float
        return (1.0 + 2.0 * eps) / (1.0 + np.exp(-float(x))) + eps


def normalize(X, lb, ub):
    """
    Normalizes X to the range [0, 1] using lower bound (lb) and upper bound (ub).
    Handles torch.Tensor, np.ndarray, and Python scalars.
    """
    if isinstance(X, torch.Tensor):
        # Ensure lb and ub are tensors on the same device and dtype as X
        if not isinstance(lb, torch.Tensor):
            lb_tensor = torch.tensor(lb, dtype=X.dtype, device=X.device)
        else:
            lb_tensor = lb.to(dtype=X.dtype, device=X.device)

        if not isinstance(ub, torch.Tensor):
            ub_tensor = torch.tensor(ub, dtype=X.dtype, device=X.device)
        else:
            ub_tensor = ub.to(dtype=X.dtype, device=X.device)
        return (X - lb_tensor) / (ub_tensor - lb_tensor)
    elif isinstance(X, np.ndarray):
        # Ensure lb and ub are numpy arrays or broadcastable scalars of compatible dtype
        lb_array = np.asarray(lb, dtype=X.dtype)
        ub_array = np.asarray(ub, dtype=X.dtype)
        return (X - lb_array) / (ub_array - lb_array)
    else:  # Fallback for Python scalars
        return (float(X) - float(lb)) / (float(ub) - float(lb))


def denormalize(X, lb, ub):
    """
    Denormalizes X from the range [0, 1] back to its original scale
    using lower bound (lb) and upper bound (ub).
    Handles torch.Tensor, np.ndarray, and Python scalars.
    """
    if isinstance(X, torch.Tensor):
        # Ensure lb and ub are tensors on the same device and dtype as X
        if not isinstance(lb, torch.Tensor):
            lb_tensor = torch.tensor(lb, dtype=X.dtype, device=X.device)
        else:
            lb_tensor = lb.to(dtype=X.dtype, device=X.device)

        if not isinstance(ub, torch.Tensor):
            ub_tensor = torch.tensor(ub, dtype=X.dtype, device=X.device)
        else:
            ub_tensor = ub.to(dtype=X.dtype, device=X.device)
        return lb_tensor + X * (ub_tensor - lb_tensor)
    elif isinstance(X, np.ndarray):
        # Ensure lb and ub are numpy arrays or broadcastable scalars of compatible dtype
        lb_array = np.asarray(lb, dtype=X.dtype)
        ub_array = np.asarray(ub, dtype=X.dtype)
        return lb_array + X * (ub_array - lb_array)
    else:  # Fallback for Python scalars
        return float(lb) + float(X) * (float(ub) - float(lb))


def prettyplot(xlabel, ylabel, xlabelpad=-10, ylabelpad=-20, minXticks=True, minYticks=True):
    """
    Applies custom styling to a Matplotlib plot.
    """
    plt.xlabel(xlabel, labelpad=xlabelpad)
    plt.ylabel(ylabel, labelpad=ylabelpad)

    if minXticks:
        current_xlim = plt.xlim()  # Get current limits
        plt.xticks(current_xlim)  # Set ticks at these limits (min and max of current view)

        # Get the locations and labels of these newly set ticks
        # Note: plt.xticks() called after setting them returns the new state.
        tick_locs, labels = plt.xticks()

        if labels:  # Check if any labels were created
            labels[0].set_horizontalalignment("left")
            if len(labels) > 1:  # Ensure there's a distinct last label
                labels[-1].set_horizontalalignment("right")
            elif len(labels) == 1:  # If only one label, it's both first and last
                labels[0].set_horizontalalignment("center")  # Or keep as 'left' as per original logic

    if minYticks:
        current_ylim = plt.ylim()
        plt.yticks(current_ylim)

        tick_locs, labels = plt.yticks()

        if labels:
            labels[0].set_verticalalignment("bottom")
            if len(labels) > 1:
                labels[-1].set_verticalalignment("top")
            elif len(labels) == 1:
                labels[0].set_verticalalignment("center")  # Or keep 'bottom'


if __name__ == '__main__':
    # Example Usage (Python 3)
    print("--- Testing invlogit ---")
    # NumPy
    x_np = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
    print(f"NumPy input: {x_np}")
    print(f"invlogit(NumPy): {invlogit(x_np)}")
    print(f"invlogit(NumPy, eps=1e-3): {invlogit(x_np, eps=1e-3)}")

    # PyTorch
    x_torch = torch.tensor([-10.0, -1.0, 0.0, 1.0, 10.0], dtype=torch.float32)
    print(f"PyTorch tensor input: {x_torch}")
    print(f"invlogit(PyTorch tensor): {invlogit(x_torch)}")
    x_torch_double = torch.tensor([-10.0, -1.0, 0.0, 1.0, 10.0], dtype=torch.float64)
    print(f"invlogit(PyTorch tensor double): {invlogit(x_torch_double)}")

    # Scalar
    x_scalar = 0.5
    print(f"Scalar input: {x_scalar}")
    print(f"invlogit(Scalar): {invlogit(x_scalar)}")
    print("-" * 20)

    print("\n--- Testing normalize/denormalize ---")
    lb, ub = 0, 10
    # NumPy
    data_np = np.array([0., 2.5, 5., 7.5, 10.])
    norm_np = normalize(data_np, lb, ub)
    denorm_np = denormalize(norm_np, lb, ub)
    print(f"Original NumPy: {data_np}")
    print(f"Normalized NumPy: {norm_np}")
    print(f"Denormalized NumPy: {denorm_np}")

    # PyTorch
    data_torch = torch.tensor([0., 2.5, 5., 7.5, 10.], dtype=torch.float32)
    norm_torch = normalize(data_torch, lb, ub)
    denorm_torch = denormalize(norm_torch, lb, ub)
    print(f"Original PyTorch: {data_torch}")
    print(f"Normalized PyTorch: {norm_torch}")
    print(f"Denormalized PyTorch: {denorm_torch}")

    # PyTorch with tensor bounds
    lb_t = torch.tensor(0.0, dtype=torch.float32)
    ub_t = torch.tensor(10.0, dtype=torch.float32)
    norm_torch_tb = normalize(data_torch, lb_t, ub_t)
    print(f"Normalized PyTorch (tensor bounds): {norm_torch_tb}")

    # Scalar
    data_scalar = 7.0
    norm_scalar = normalize(data_scalar, lb, ub)
    denorm_scalar = denormalize(norm_scalar, lb, ub)
    print(f"Original Scalar: {data_scalar}")
    print(f"Normalized Scalar: {norm_scalar}")
    print(f"Denormalized Scalar: {denorm_scalar}")
    print("-" * 20)

    print("\n--- Testing prettyplot (visual check) ---")
    plt.figure(figsize=(6, 4))
    x_plot = np.linspace(0, 10, 100)
    y_plot = np.sin(x_plot)
    plt.plot(x_plot, y_plot)
    plt.xlim(0, 10)
    plt.ylim(-1, 1)
    prettyplot("X-axis", "Y-axis")
    plt.title("Prettyplot Test")
    plt.show()