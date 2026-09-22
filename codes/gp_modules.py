"""
Gaussian Process (GP) modules for spatial priors.
"""
import warnings

from gpytorch.kernels import Kernel
from gpytorch.means import Mean
from linear_operator.utils.cholesky import psd_safe_cholesky
import torch
from torch import Tensor, nn


class GP(nn.Module):
    """
    Base GP class over the latent space.
    Uses spatial coordinates as inputs and latent variables as outputs.
    """

    def __init__(self, output_dims: int, kernel: Kernel, mean: Mean):
        super(GP, self).__init__()
        self.output_dims = output_dims

        kernel_batch_shape = kernel.batch_shape
        if len(kernel_batch_shape) == 0:
            warnings.warn("Got a kernel without batch dim, we added an extra batch dim.")
            kernel = kernel.expand_batch(torch.Size([1]))
        elif len(kernel_batch_shape) > 1:
            raise ValueError(f"Only support one batch dim, but got batch shape {kernel_batch_shape}.")
        elif len(kernel_batch_shape) == 1 and kernel_batch_shape[0] != 1:
            assert kernel_batch_shape[0] == output_dims, \
                f"kernel batch doesn't match the GP output dims {output_dims}."

        assert len(mean.batch_shape) == 1, \
            f"Mean function of GP must have 1-D batch dim but got {mean.batch_shape}."
        assert mean.batch_shape[0] == output_dims, \
            f"The batch dim of Mean must be equal to output_dims {output_dims}."

        self.kernel = kernel
        self.mean = mean

    def prior(self, x: Tensor, are_neighbors: bool = False):
        """
        Compute GP prior based on spatial coordinates.

        Args:
            x: spatial coordinates.
            are_neighbors: whether x represents nearest neighbors.

        Returns:
            prior_mean, prior_cov.
        """
        x = x.unsqueeze(-3)
        prior_mean = self.mean(x)
        prior_cov = self.kernel(x).to_dense()
        if are_neighbors:
            prior_mean = prior_mean.transpose(-2, -3)
            prior_cov = prior_cov.transpose(-3, -4)
        return prior_mean, prior_cov

