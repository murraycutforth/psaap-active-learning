# Multi-fidelity DGP classification model
# x->f_low->f_high->logit(f_high)->y_high
# |->logit(f_low)->y_low

import numpy as np
import torch
import gpytorch
from gpytorch.kernels import RBFKernel, ScaleKernel, ProductKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.mlls import VariationalELBO,DeepApproximateMLL
from gpytorch.likelihoods import _OneDimensionalLikelihood
from torch.distributions import Bernoulli
from torch.distributions.normal import Normal

from src.models.dgp_layers import GPlayer_Cholesky, GPlayer_NN

from src.active_learning.util_classes import BiFidelityModel

def squeeze_mtmvn(mtmvn):
    """
    Converts MultitaskMultivariateNormal with num_tasks=1 to MultivariateNormal
    by removing the task dimension.
    """
    mean = mtmvn.mean.squeeze(-1)  # [n]
    covar = mtmvn.lazy_covariance_matrix.squeeze(-3).squeeze(-1)  # [n, n]
    return MultivariateNormal(mean, covar)


class ProbitLikelihood(_OneDimensionalLikelihood):
    def __init__(self):
        super().__init__()
        self.standard_normal = Normal(0, 1)

    def forward(self, function_samples, **kwargs):
        probs = self.standard_normal.cdf(function_samples)
        return Bernoulli(probs=probs)


class BFDGPC(torch.nn.Module, BiFidelityModel):
    def __init__(self, input_dims, num_inducing_low, num_inducing_high, l2_reg_lambda=1):
        super(BFDGPC, self).__init__()
        self.num_inducing_low = num_inducing_low
        self.num_inducing_high = num_inducing_high
        self.input_dims = input_dims
        self.l2_reg_lambda = l2_reg_lambda

        gp_low = GPlayer_Cholesky(input_dims=input_dims, 
                                  output_dims=1,
                                  num_inducing=num_inducing_low, 
                                  mean_type="constant")
        
        covar_module = ProductKernel(
            ScaleKernel(RBFKernel(batch_shape=torch.Size([]), active_dims=[0]), 
                                  batch_shape=torch.Size([])),
            ScaleKernel(RBFKernel(batch_shape=torch.Size([]), active_dims=[1,2], ard_num_dims=2), 
                                  batch_shape=torch.Size([])),
        ) + ScaleKernel(RBFKernel(batch_shape=torch.Size([]), active_dims=[1,2], ard_num_dims=2), 
                                  batch_shape=torch.Size([]))
        gp_high = GPlayer_Cholesky(input_dims=3, 
                                output_dims=None, 
                                num_inducing=num_inducing_high,
                                covar_module=covar_module,
                                mean_type="constant")
        
        self.gp_low = gp_low
        self.gp_high = gp_high
        # self.gp_high_residual = gp_high_residual
        # self.low_likelihood = ProbitLikelihood()
        # self.high_likelihood = ProbitLikelihood()
        self.low_likelihood = BernoulliLikelihood()
        self.high_likelihood = BernoulliLikelihood()

        print("[INFO] Initializing BFDGPC model")
    
    def predict_f_L(self, x_predict, num_samples=1):
        if isinstance(x_predict, np.ndarray):
            x_predict = torch.from_numpy(x_predict).float()
        q_f_L = squeeze_mtmvn(self.gp_low(x_predict))
        if num_samples > 1:
            q_f_L = q_f_L.rsample(sample_shape=torch.Size([num_samples]))
        return q_f_L
    
    def predict_f_H(self, x_predict, num_samples=1):
        if isinstance(x_predict, np.ndarray):
            x_predict = torch.from_numpy(x_predict).float()
        q_f_L = self.gp_low(x_predict)
        q_f_H = self.gp_high(q_f_L, x_predict) # + self.gp_high_residual(inputs)
        if num_samples > 1:
            q_f_H = q_f_H.rsample(sample_shape=torch.Size([num_samples]))
        return q_f_H

    def forward(self, test_x, num_samples=1, return_lf=False):
        self.eval()
        with torch.no_grad():
            low_output = self.predict_f_L(test_x, num_samples)
            high_output = self.predict_f_H(test_x, num_samples)
            # low_pred = self.low_likelihood(low_output)
            # high_pred = self.high_likelihood(high_output)

        if return_lf:
            if num_samples > 1:
                results = {"hf_samples": self.high_likelihood(high_output),
                           "lf_samples": self.low_likelihood(low_output),
                           "hf_mean": self.high_likelihood(high_output).probs.mean(dim=0),
                           "lf_mean": self.low_likelihood(low_output).probs.mean(dim=0)}
            else:
                results = {"hf_mean": self.high_likelihood(high_output).probs.mean(dim=0),
                           "lf_mean": self.low_likelihood(low_output).probs.mean(dim=0)}
            return results
        else:
            if num_samples > 1:
                results = {"hf_samples": self.high_likelihood(high_output),
                           "hf_mean": self.high_likelihood(high_output).probs.mean(dim=0)}
            else:
                results = self.high_likelihood(high_output).probs.mean(dim=0)
            return results

    def __repr__(self):
        return f"BFDGPC"
    
    def predict_hf_prob(self, x_predict, num_samples=1):
        hf_probs = self.forward(x_predict, 
                            num_samples=num_samples, 
                            return_lf=False)
        return hf_probs.mean(dim=0).detach().numpy()
    
    def predict_lf_prob(self, x_predict, num_samples=1):
        if num_samples > 1:
            return self.low_likelihood(self.predict_f_L(x_predict, num_samples=num_samples)).probs.mean(dim=0).detach().numpy()
        else:
            return self.low_likelihood(self.predict_f_L(x_predict)).probs.mean(dim=0).detach().numpy()
    
    def predict_lf(self, x_predict, num_samples=1):
        if num_samples > 1:
            return self.predict_f_L(x_predict, num_samples=num_samples).mean(dim=0).detach().numpy()
        else:
            return self.predict_f_L(x_predict).mean(dim=0).detach().numpy()

    
    def train_model(self, X_LF, Y_LF, X_HF, Y_HF, lr, n_epochs):

        update_epochs = n_epochs // 3  # Update the learning rate every 1/3 of the epochs
        power = 0 

        if isinstance(X_LF, np.ndarray):
            X_LF = torch.from_numpy(X_LF).float()
        if isinstance(Y_LF, np.ndarray):
            Y_LF = torch.from_numpy(Y_LF).float()
        if isinstance(X_HF, np.ndarray):
            X_HF = torch.from_numpy(X_HF).float()
        if isinstance(Y_HF, np.ndarray):
            Y_HF = torch.from_numpy(Y_HF).float()
        
        self.gp_low.train()
        self.gp_high.train()

        # initialize the optimizer and mll
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        mll_low = DeepApproximateMLL(VariationalELBO(self.low_likelihood, 
                                                     self.gp_low, 
                                                     num_data=X_LF.shape[0], 
                                                     beta=self.l2_reg_lambda))
        mll_high = DeepApproximateMLL(VariationalELBO(self.high_likelihood, 
                                                      self.gp_high, 
                                                      num_data=X_HF.shape[0], 
                                                      beta=self.l2_reg_lambda))

        print(f"[INFO] Low data shape: {X_LF.shape}")
        print(f"[INFO] Low data target shape: {Y_LF.shape}")
        print(f"[INFO] High data shape: {X_HF.shape}")
        print(f"[INFO] High data target shape: {Y_HF.shape}")
        print(f"[INFO] Training BFDGPC model with {n_epochs} epochs and learning rate {lr}")
        print(f"[INFO] L2 regularization weight: {self.l2_reg_lambda}")

        for epoch in range(n_epochs):
            optimizer.zero_grad()
            low_output = self.predict_f_L(X_LF)
            low_loss = mll_low(low_output, Y_LF)

            high_output = self.predict_f_H(X_HF)
            high_loss = mll_high(high_output, Y_HF)
            loss = -low_loss -high_loss
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print(f"[INFO] Epoch {epoch}, Low Loss: {-low_loss.item()}, High Loss: {-high_loss.item()}")

            if (epoch+1) % update_epochs == 0:
                power += 1
                optimizer = torch.optim.Adam(self.parameters(), lr=lr*0.1**power)
                mll_low = DeepApproximateMLL(VariationalELBO(self.low_likelihood, 
                                                             self.gp_low, 
                                                             num_data=X_LF.shape[0], 
                                                             beta=self.l2_reg_lambda*0.1**power))
                mll_high = DeepApproximateMLL(VariationalELBO(self.high_likelihood, 
                                                              self.gp_high, 
                                                              num_data=X_HF.shape[0], 
                                                              beta=self.l2_reg_lambda*0.1**power))
                
                print(f"[INFO] Updated L2 regularization weight to {self.l2_reg_lambda*0.1**power} at epoch {epoch}")
                print(f"[INFO] Updated learning rate to {lr*0.1**power} at epoch {epoch}")
    
    def predict_hf_prob_var(self, x_predict, num_samples=1):
        self.eval()
        q_f_H = self.predict_f_H(x_predict, num_samples)
        hf_probs = self.high_likelihood(q_f_H)
        return hf_probs.probs.var(dim=0).detach().numpy()

    def predict_multi_fidelity_latent_joint(self, 
                                            X_L: torch.tensor, 
                                            X_H: torch.tensor, 
                                            X_prime: torch.tensor,
                                            num_samples=1,
                                            extra_assertions=False):
        pass

            
    def evaluate_elpp(self, X_HF_test: np.ndarray, Y_HF_test: np.ndarray, num_samples=1):
        """The expected log predictive probability is a standard metric in the VI literature.
        This metric is suitable for probability estimation, as we have in the PSAAP problem.
        """
        pred_probs_p1 = self.forward(X_HF_test, 
                                     num_samples=num_samples, 
                                     return_lf=False)

        # Ensure Y_HF_test is a torch tensor for calculations
        if not torch.is_tensor(Y_HF_test):
            Y_HF_test = torch.tensor(Y_HF_test).float()

        # 2. Calculate the log probability of the TRUE class for each sample.
        #    This is the log-likelihood of the Bernoulli distribution.
        #    - If Y_HF_test is 1, this becomes: 1 * log(p1) + 0 * log(1-p1) = log(p1)
        #    - If Y_HF_test is 0, this becomes: 0 * log(p1) + 1 * log(1-p1) = log(1-p1)
        #    We add a small epsilon for numerical stability to avoid log(0).
        epsilon = 1e-8
        log_probs_of_true_class = (
                Y_HF_test * torch.log(pred_probs_p1 + epsilon) +
                (1 - Y_HF_test) * torch.log(1 - pred_probs_p1 + epsilon)
        )

        # 3. The ELPP is the average of these log probabilities.
        elpp = torch.mean(log_probs_of_true_class)

        return elpp.item()


    def evaluate_accuracy(self, X_HF_test, Y_HF_test, num_samples=1):
        """If we have a classification problem (rather than probability estimation).
        """
        self.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_probs = self.forward(torch.tensor(X_HF_test, dtype=torch.float), 
                                      num_samples=num_samples, 
                                      return_lf=False)
            test_probs = test_probs.mean(dim=0)
            test_labels = (test_probs > 0.5).float()
        accuracy = (test_labels == torch.tensor(Y_HF_test, dtype=torch.float32)).float().mean().item()

        return accuracy
    
    





