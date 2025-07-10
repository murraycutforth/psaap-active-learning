import pyDOE
import torch
import numpy as np
from tqdm import tqdm
from gpytorch.distributions import MultivariateNormal
from collections import Counter

from src.active_learning.util_classes import BiFidelityDataset, ALExperimentConfig
from src.models.bfgpc import BFGPC_ELBO
from src.problems.toy_example import create_smooth_change_linear

#from src.batch_al_strategies.entropy_functions import calculate_entropy_from_samples_miller_madow, calculate_entropy_from_samples
from src.batch_al_strategies.entropy_functions import estimate_marginal_entropy_H_Y, estimate_conditional_entropy_H_Y_given_Q


linear_low_f1, linear_high_f1, p_LF_toy, p_HF_toy = create_smooth_change_linear()


def sampling_function_L(X_normalized):  # Expects N x 2 normalized input
    Y_linear_low_grid, probs_linear_low_grid = linear_low_f1(X_normalized, reps=1)
    Y_linear_high_grid = Y_linear_low_grid.mean(axis=0)
    return Y_linear_high_grid


def sampling_function_H(X_normalized):
    Y_linear_high_grid, probs_linear_high_grid = linear_high_f1(X_normalized, reps=1)
    Y_linear_high_grid = Y_linear_high_grid.mean(axis=0)
    return Y_linear_high_grid


def calculate_entropy_from_samples_miller_madow(y_samples):
    # y_samples: list of tuples, each is a bitstring Y
    # computes empirical entropy (base e)
    total = len(y_samples)
    counts = Counter(y_samples)
    probs = np.array([v/float(total) for v in counts.values()])
    naive_entropy = -np.sum(probs * np.log(probs))
    K = len(counts)
    bias = (K-1) / (2*total)
    return naive_entropy + bias


def sample_bernoulli(R: torch.Tensor) -> torch.Tensor:
    # R: [batch, d]
    probit_link = torch.distributions.Normal(0, 1).cdf
    probs = probit_link(R)
    return torch.bernoulli(probs)



def estimate_MI_MC_true_likelihood(M: int, K: int, joint_mvn: MultivariateNormal, d:int, seed=None):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    assert joint_mvn.mean.shape[0] == 100 + d
    n = joint_mvn.mean.shape[0]

    H_Y = estimate_marginal_entropy_H_Y(M, K, joint_mvn, d)
    H_Y_given_Q = estimate_conditional_entropy_H_Y_given_Q(M, K, joint_mvn, d)
    MI = H_Y - H_Y_given_Q

    return MI, H_Y, H_Y_given_Q




def estimate_MI_MC_empirical_likelihood(M, N, joint_mvn: MultivariateNormal, d, use_miller_madow=True, seed=None):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    assert joint_mvn.mean.shape[0] == 100 + d
    n = joint_mvn.mean.shape[0]

    Q_idxs = torch.arange(d, n)
    R_idxs = torch.arange(0, d)

    # Build marginal over Q (the last 100 dimensions):
    Q_mean = joint_mvn.mean[Q_idxs]
    Q_covar = joint_mvn.covariance_matrix[Q_idxs][:, Q_idxs]
    Q_dist = MultivariateNormal(Q_mean, Q_covar)

    # Build conditional over R given Q (for each sampled Q)
    mean = joint_mvn.mean
    cov = joint_mvn.covariance_matrix
    # block extraction
    mu_R = mean[R_idxs]
    mu_Q = mean[Q_idxs]
    cov_RR = cov[R_idxs][:, R_idxs]
    cov_RQ = cov[R_idxs][:, Q_idxs]
    cov_QR = cov[Q_idxs][:, R_idxs]
    cov_QQ = cov[Q_idxs][:, Q_idxs]

    Y_all_samples = []
    Y_by_Q = []

    # For conditional entropy: store a list of Y samples for each Q
    for outer in range(N):

        # 1. Sample Q ~ P(Q)
        Q = Q_dist.sample()

        # 2. Compute parameters of P(R|Q)
        # Cond formula: mu_R|Q = mu_R + Sigma_RQ @ Sigma_QQ^{-1} (Q - mu_Q)
        cov_QQ_inv = torch.linalg.inv(cov_QQ)
        cond_mean = mu_R + cov_RQ @ cov_QQ_inv @ (Q - mu_Q)
        cond_covar = cov_RR - cov_RQ @ cov_QQ_inv @ cov_QR
        cond_covar = 0.5 * (cond_covar + cond_covar.T) + 1e-4 * torch.eye(cond_covar.shape[0])
        R_given_Q_dist = MultivariateNormal(cond_mean, cond_covar, validate_args=True)

        # store list of all M Y samples for this Q (for conditional entropy)
        Y_for_this_Q = []
        for inner in range(M):
            R = R_given_Q_dist.sample()
            Y = sample_bernoulli(R).int().numpy()
            Y_tuple = tuple(Y.tolist())
            Y_all_samples.append(Y_tuple)         # for marginal entropy
            Y_for_this_Q.append(Y_tuple)          # for conditional entropy

        Y_by_Q.append(Y_for_this_Q)

    # --- Marginal entropy ---
    if use_miller_madow:
        H_Y = calculate_entropy_from_samples_miller_madow(Y_all_samples)
    else:
        # Simple empirical entropy
        total = len(Y_all_samples)
        counts = Counter(Y_all_samples)
        probs = np.array([v/float(total) for v in counts.values()])
        H_Y = -np.sum(probs * np.log(probs))

    # --- Conditional entropy ---
    H_Y_given_Qs = []
    for Ylist in Y_by_Q:
        if use_miller_madow:
            H_y_given_q = calculate_entropy_from_samples_miller_madow(Ylist)
        else:
            total = len(Ylist)
            counts = Counter(Ylist)
            probs = np.array([v/float(total) for v in counts.values()])
            H_y_given_q = -np.sum(probs * np.log(probs))
        H_Y_given_Qs.append(H_y_given_q)
    H_Y_given_Q = np.mean(H_Y_given_Qs)

    MI = H_Y - H_Y_given_Q
    return MI, H_Y, H_Y_given_Q


def MI_convergence_experiment(joint_mvn, M_vals, N_vals, d, n_repeats=1):
    # Initialize 2D grids for MI, H(Y), H(Y|Q) [mean and std]
    MI_grid = np.zeros((len(M_vals), len(N_vals)))
    MI_std_grid = np.zeros((len(M_vals), len(N_vals)))
    HY_grid = np.zeros((len(M_vals), len(N_vals)))
    HYgQ_grid = np.zeros((len(M_vals), len(N_vals)))

    for i, M in enumerate(tqdm(M_vals, desc="M grid")):
        for j, N in enumerate(tqdm(N_vals, desc="N grid", leave=False)):
            MI_reps = []
            HY_reps = []
            HYgQ_reps = []
            for rep in range(n_repeats):
                #MI, H_Y, H_Y_given_Q = estimate_MI_MC_empirical_likelihood(M, N, joint_mvn, seed=rep, d=d)
                MI, H_Y, H_Y_given_Q = estimate_MI_MC_true_likelihood(M, N, joint_mvn, seed=rep, d=d)
                MI_reps.append(MI)
                HY_reps.append(H_Y)
                HYgQ_reps.append(H_Y_given_Q)
            MI_grid[i, j] = np.mean(MI_reps)
            MI_std_grid[i, j] = np.std(MI_reps)
            HY_grid[i, j] = np.mean(HY_reps)
            HYgQ_grid[i, j] = np.mean(HYgQ_reps)
    return MI_grid, MI_std_grid, HY_grid, HYgQ_grid, M_vals, N_vals

def plot_mi_lines(MI_grid, MI_std_grid, M_vals, N_vals):
    """
    Plots MI vs N for fixed M (one line per M, colored by M)
    and MI vs M for fixed N (one line per N, colored by N).
    Shows fill_between regions for std deviation from repeats.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # MI vs N (each line: one value of M)
    norm = plt.Normalize(M_vals[0], M_vals[-1])
    cmap = cm.viridis

    for i, M in enumerate(M_vals):
        color = cmap(norm(M))
        y = MI_grid[i, :]
        std = MI_std_grid[i, :]
        axes[0].plot(N_vals, y, '-o', label=f'M={M}', color=color)
        axes[0].fill_between(N_vals, y - std, y + std, color=color, alpha=0.25, linewidth=0)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=axes[0], label="M value (lines)")
    axes[0].set_xlabel("N (Outer MC samples)")
    axes[0].set_ylabel("Estimated MI")
    axes[0].set_title("MI vs N (fixed M, color=M)")
    axes[0].grid(True)

    # MI vs M (each line: one value of N)
    norm2 = plt.Normalize(N_vals[0], N_vals[-1])
    cmap2 = cm.plasma

    for j, N in enumerate(N_vals):
        color = cmap2(norm2(N))
        y = MI_grid[:, j]
        std = MI_std_grid[:, j]
        axes[1].plot(M_vals, y, '-o', label=f'N={N}', color=color)
        axes[1].fill_between(M_vals, y - std, y + std, color=color, alpha=0.25, linewidth=0)

    sm2 = plt.cm.ScalarMappable(cmap=cmap2, norm=norm2)
    plt.colorbar(sm2, ax=axes[1], label="N value (lines)")
    axes[1].set_xlabel("M (Inner MC samples)")
    axes[1].set_ylabel("Estimated MI")
    axes[1].set_title("MI vs M (fixed N, color=N)")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


def plot_convergence(MI_grid, HY_grid, HYgQ_grid, M_vals, N_vals):
    import matplotlib.pyplot as plt

    Mv, Nv = np.meshgrid(M_vals, N_vals, indexing='ij')
    levels_MI = np.linspace(np.nanmin(MI_grid), np.nanmax(MI_grid), 20)
    levels_HY = np.linspace(np.nanmin(HY_grid), np.nanmax(HY_grid), 20)
    levels_HYgQ = np.linspace(np.nanmin(HYgQ_grid), np.nanmax(HYgQ_grid), 20)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    cs0 = axes[0].contourf(Mv, Nv, MI_grid, levels=levels_MI, cmap='viridis')
    c0 = axes[0].contour(Mv, Nv, MI_grid, levels=levels_MI, colors='k', linewidths=0.7)
    fig.colorbar(cs0, ax=axes[0])
    axes[0].clabel(c0, inline=True, fontsize=8)
    axes[0].set_title('Mutual Information $I(Y;F)$')
    axes[0].set_xlabel('M (inner MC samples)')
    axes[0].set_ylabel('N (outer MC samples)')

    cs1 = axes[1].contourf(Mv, Nv, HY_grid, levels=levels_HY, cmap='viridis')
    c1 = axes[1].contour(Mv, Nv, HY_grid, levels=levels_HY, colors='k', linewidths=0.7)
    fig.colorbar(cs1, ax=axes[1])
    axes[1].clabel(c1, inline=True, fontsize=8)
    axes[1].set_title('Marginal Entropy $H(Y)$')
    axes[1].set_xlabel('M (inner MC samples)')
    axes[1].set_ylabel('N (outer MC samples)')

    cs2 = axes[2].contourf(Mv, Nv, HYgQ_grid, levels=levels_HYgQ, cmap='viridis')
    c2 = axes[2].contour(Mv, Nv, HYgQ_grid, levels=levels_HYgQ, colors='k', linewidths=0.7)
    fig.colorbar(cs2, ax=axes[2])
    axes[2].clabel(c2, inline=True, fontsize=8)
    axes[2].set_title('Conditional Entropy $H(Y|Q)$')
    axes[2].set_xlabel('M (inner MC samples)')
    axes[2].set_ylabel('N (outer MC samples)')

    plt.tight_layout()
    plt.show()


# Example integration with your main()
def main():
    dataset = BiFidelityDataset(sample_LF=sampling_function_L, sample_HF=sampling_function_H,
                                true_p_LF=p_LF_toy, true_p_HF=p_HF_toy,
                                name='ToyLinear', c_LF=0.1, c_HF=1.0)

    base_config = ALExperimentConfig(
        N_L_init=500,
        N_H_init=500,
        train_epochs=100,
        train_lr=0.1,
    )

    X_L_train = pyDOE.lhs(2, base_config.N_L_init)
    Y_L_train = dataset.sample_LF(X_L_train)

    X_H_train = pyDOE.lhs(2, base_config.N_H_init)
    Y_H_train = dataset.sample_HF(X_H_train)

    model = BFGPC_ELBO()
    model.train_model(X_L_train, Y_L_train, X_H_train, Y_H_train, base_config.train_lr, base_config.train_epochs)

    X_test = torch.tensor(pyDOE.lhs(2, samples=100)).float()
    X_cand_LF = torch.tensor(pyDOE.lhs(2, samples=50)).float()
    X_cand_HF = torch.tensor(pyDOE.lhs(2, samples=50)).float()

    joint_mvn = model.predict_multi_fidelity_latent_joint(X_cand_LF, X_cand_HF, X_test, extra_assertions=True)

    # MI convergence
    M_vals = [5, 10, 20, 40, 80, 160, 320]
    N_vals = [5, 10, 20, 40, 80, 160, 320]
    MI_grid, MI_grid_std, HY_grid, HYgQ_grid, M_vals, N_vals = MI_convergence_experiment(joint_mvn, M_vals, N_vals, d=100, n_repeats=5)
    plot_convergence(MI_grid, HY_grid, HYgQ_grid, M_vals, N_vals)
    plot_mi_lines(MI_grid, MI_grid_std, M_vals, N_vals)


if __name__ == '__main__':
    main()