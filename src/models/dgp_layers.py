from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution, MeanFieldVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.models.deep_gps import DeepGPLayer
from gpytorch.variational.nearest_neighbor_variational_strategy import NNVariationalStrategy

import torch
import gpytorch

# Cholesky Variational Strategy
class GPlayer_Cholesky(DeepGPLayer):
    def __init__(self, 
                 input_dims, 
                 output_dims,
                 num_inducing, 
                 covar_module=None,
                 mean_type='constant'):
        
        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(output_dims, num_inducing, input_dims)
            batch_shape = torch.Size([output_dims])
        
        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )
        super(GPlayer_Cholesky, self).__init__(variational_strategy, input_dims, output_dims)

        if mean_type == 'constant':
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        else:
            self.mean_module = LinearMean(input_dims)
        
        if covar_module is None:
            self.covar_module = ScaleKernel(
                RBFKernel(batch_shape=batch_shape),
                batch_shape=batch_shape,
                ard_num_dims=input_dims
            )
        else:
            self.covar_module = covar_module
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
    
    def __call__(self, x, *other_inputs, **kwargs):
        """
        Overriding __call__ isn't strictly necessary, but it lets us add concatenation based skip connections
        easily. For example, hidden_layer2(hidden_layer1_outputs, inputs) will pass the concatenation of the first
        hidden layer's outputs and the input data to hidden_layer2.
        """
        if len(other_inputs):
            if isinstance(x, gpytorch.distributions.MultitaskMultivariateNormal):
                x = x.rsample()
            
            
            processed_inputs = [
                inp.unsqueeze(0).expand(gpytorch.settings.num_likelihood_samples.value(), *inp.shape)
                for inp in other_inputs
            ]
                
            x = torch.cat([x] + processed_inputs, dim=-1)

        return super().__call__(x, are_samples=bool(len(other_inputs)))



# Nearest Neighbor Variational Strategy
# TODO: Not compatible with the GPlayer_Cholesky as the high_level GP right now
# TODO: the contour looks NN blobs.
class GPlayer_NN(DeepGPLayer):
    def __init__(self, 
                 input_dims, 
                 output_dims,
                 num_inducing, 
                 covar_module=None,
                 mean_type='constant'):
        
        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(output_dims, num_inducing, input_dims)
            batch_shape = torch.Size([output_dims])


        # diagnoal covariance matrix
        variational_distribution = MeanFieldVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )

        variational_strategy = NNVariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            k = min(num_inducing, 128),
            training_batch_size=16
        )

        super(GPlayer_NN, self).__init__(variational_strategy, input_dims, output_dims)

        if mean_type == 'constant':
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        else:
            self.mean_module = LinearMean(input_dims)
        
        if covar_module is None:
            self.covar_module = ScaleKernel(
                RBFKernel(batch_shape=batch_shape),
                batch_shape=batch_shape,
                ard_num_dims=input_dims
            )
        else:
            self.covar_module = covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
    
    def __call__(self, x, prior=False, **kwargs):
        if x is not None:
            if x.dim() == 1:
                x = x.unsqueeze(-1)
        q_normal_dist = self.variational_strategy(x=x, prior=prior, **kwargs)
        return q_normal_dist