import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.log_marginal_likelihood_metric import compute_log_marginal_likelihood
from src.predicted_p_std import compute_pred_p_std

# Set random seed for reproducibility
np.random.seed(42)


# Define the low-fidelity function (diagonal linear interface)
def low_fidelity(x):
    """
    Low fidelity function with a diagonal linear interface
    x: array of shape (n_samples, 2) with each point in [0,1] × [0,1]
    returns: probability values in [0,1]
    """
    # Linear interface: x[0] + x[1] - 1 = 0
    # Points above this line will be class 1, below will be class 0
    decision_value = x[:, 0] + x[:, 1] - 1.0

    # Apply sigmoid function to get probabilities
    probs = 1.0 / (1.0 + np.exp(-10 * decision_value))

    return probs


# Define the high-fidelity function (shifted interface)
def high_fidelity(x):
    """
    High fidelity function with a shifted diagonal linear interface
    x: array of shape (n_samples, 2) with each point in [0,1] × [0,1]
    returns: probability values in [0,1]
    """
    # Shifted linear interface: x[0] + x[1] - 0.8 = 0
    decision_value = x[:, 0] + x[:, 1] - 0.8

    # Apply sigmoid function to get probabilities
    probs = 1.0 / (1.0 + np.exp(-10 * decision_value))

    return probs


# Generate training and testing data
def generate_data(n_samples, fidelity_func):
    # Generate uniform random points in [0,1] × [0,1]
    X = np.random.rand(n_samples, 2)

    # Get probabilities from the fidelity function
    probs = fidelity_func(X)

    # Sample data from bernoulli distribution (p depends on x)
    y = np.random.binomial(n=1, p=probs)

    return X, y, probs


# Create a mesh grid for visualization
def create_mesh_grid(resolution=100):
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    xx, yy = np.meshgrid(x, y)
    return xx, yy


# Visualize the classifier results
def visualize_results(xx, yy, true_func, pred_func, X_train, y_train, title):
    plt.figure(figsize=(12, 12))

    # Plot the true function
    plt.subplot(2, 2, 1)
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    true_z = true_func(mesh_points).reshape(xx.shape)
    plt.contourf(xx, yy, true_z, cmap=plt.cm.RdBu)
    plt.title(f'True Probability - {title}')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.colorbar()

    plt.subplot(2, 2, 2)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, alpha=0.5)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.colorbar()

    # Plot the predicted probability
    plt.subplot(2, 2, 3)
    if pred_func is not None:
        pred_z = pred_func(mesh_points).reshape(xx.shape)
        plt.contourf(xx, yy, pred_z, cmap=plt.cm.RdBu)
        plt.title(f'Predicted Probability - {title}')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.colorbar()

    # Plot the std of predicted probability
    plt.subplot(2, 2, 4)
    pred_z_std = compute_pred_p_std(gpc_low, mesh_points, N=100).reshape(xx.shape)
    plt.contourf(xx, yy, pred_z_std, cmap=plt.cm.RdBu)
    plt.title(f'Std dev of predicted probability - {title}')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.colorbar()

    plt.tight_layout()
    plt.show()


# Create a function that returns the GP predicted probabilities
def get_gp_predict_func(gp_model):
    def predict_func(X):
        return gp_model.predict_proba(X)[:, 1]

    return predict_func


def log_likelihood(y_true, p_pred):
    """The mean log likelihood of observing the given y_true binary outcomes under the predicted probabilities
    Each y_true is Bernoulli distributed, using p_pred
    """
    assert y_true.shape == p_pred.shape
    assert len(y_true.shape) == 1

    ll = 0
    for y, p in zip(y_true, p_pred):
        ll += y * np.log(p) + (1 - y) * np.log(1 - p)

    return ll / len(y_true)


def marginal_log_likelihood_test(y_true, p_pred, gpc, X_pred):
    N = 100
    f_samples = gpc.f_samples(X_pred, 100)


# Main execution
if __name__ == "__main__":
    # Generate training and testing data for low fidelity
    n_samples = 500
    X, y, probs = generate_data(n_samples, low_fidelity)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Create and train a Gaussian Process Classifier on low-fidelity data
    kernel = 1.0 * RBF(length_scale=0.1)
    gpc_low = GaussianProcessClassifier(kernel=kernel, random_state=42)
    gpc_low.fit(X_train, y_train)

    # Evaluate the low-fidelity model using accuracy of mean, log likelihood of posterior mode, and log marginal likelihood

    y_pred_low = gpc_low.predict(X_test)
    accuracy_low = accuracy_score(y_test, y_pred_low)

    p_pred_low = gpc_low.predict_proba(X_test)[:, 1]
    ll_low = log_likelihood(y_test, p_pred_low)


    lml_low = compute_log_marginal_likelihood(gpc_low, X_test, y_test)

    print(f"Low-fidelity model accuracy: {accuracy_low:.4f}")
    print(f"Low-fidelity log likelihood: {ll_low:.4f}")
    print(f"Low-fidelity log marginal likelihood: {lml_low:.4f}")

    # Create a mesh grid for visualization
    xx, yy = create_mesh_grid(resolution=100)

    # Visualize low-fidelity results
    visualize_results(xx, yy, low_fidelity, get_gp_predict_func(gpc_low),
                      X_train, y_train, "Low Fidelity")

    # # Generate training and testing data for high fidelity
    # X_high, y_high, probs_high = generate_data(n_samples, high_fidelity)
    # X_train_high, X_test_high, y_train_high, y_test_high = train_test_split(
    #     X_high, y_high, test_size=0.3, random_state=42)

    # # Create and train a Gaussian Process Classifier on high-fidelity data
    # gpc_high = GaussianProcessClassifier(kernel=kernel, random_state=42)
    # gpc_high.fit(X_train_high, y_train_high)

    # # Evaluate the high-fidelity model
    # y_pred_high = gpc_high.predict(X_test_high)
    # accuracy_high = accuracy_score(y_test_high, y_pred_high)
    # print(f"High-fidelity model accuracy: {accuracy_high:.4f}")

    # # Visualize high-fidelity results
    # visualize_results(xx, yy, high_fidelity, get_gp_predict_func(gpc_high),
    #                   X_train_high, y_train_high, "High Fidelity")