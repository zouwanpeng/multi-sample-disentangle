"""
Disentangled GPVAE with spatial structure support.
"""
import warnings

from linear_operator.utils.cholesky import psd_safe_cholesky
import torch
from torch import Tensor, nn
from torch.distributions import Normal

from .disentangle import LD
from .gcn_layers import build_neighbor_graphs
from .gp_modules import GP


class DisentangledGPVAE(nn.Module):
    """
    Disentangled GPVAE with spatial GP priors and per-sample GCN decoder.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        gp_c: GP,
        gp_u: GP,
        k_neighbors: int,
        sample_id: int,
        use_ld: bool = True,
        ld_n_round: int = 3,
        search_device: str = "cpu",
        data_device: str = "cpu",
        module_device: str = "cpu",
        jitter: float = 1e-6,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.gp_c = gp_c
        self.gp_u = gp_u

        self.sample_id = sample_id
        self.k_neighbors = k_neighbors
        self.dim_c = self.gp_c.output_dims
        self.dim_u = self.gp_u.output_dims

        self.search_device = search_device
        self.data_device = data_device
        self.module_device = module_device
        self.jitter = jitter
        self.use_ld = use_ld

        # LD module for noise extraction from u
        if use_ld:
            self.ld_module = LD(dim=self.dim_u, n_round=ld_n_round, dropout=0.3)
        else:
            self.ld_module = None

        self.train_dataset = None
        self.nn_util = None

    def get_nn_data(self, test_x: Tensor, k_neighbors: int, **kwargs):
        """Find k nearest neighbors and return their data."""
        nn_indices = self.train_dataset.nn_util.find_nn_idx(test_x, k=k_neighbors)[0]
        return self.train_dataset.gather(nn_indices)

    def forward(self, y_batch: Tensor, adj: Tensor, batch_id: Tensor, adj_c: Tensor = None):
        """
        Forward pass through disentangled encoder and decoder.

        Args:
            y_batch: input gene expression [B, G_sample].
            adj: adjacency matrix for mini-batch [B, B] (for u encoding).
            batch_id: batch ID [B].
            adj_c: unified adjacency matrix for c encoding [B_total, B_total] (optional).

        Returns:
            c_mean, c_std, u_mean, u_std:
                posterior parameters for c and u.
            c, u, u_cleaned, aleatoric:
                sampled latent variables, cleaned u, extracted noise.
            x_recon_c, x_recon_u, x_recon_cu:
                three reconstructions: from c only, from u only, and from c+u.
        """
        # Encode (c uses unified graph if provided, u uses sample-specific graph)
        c_mean, c_std, u_mean, u_std = self.encoder(
            y_batch, adj, sample_id=self.sample_id, adj_c=adj_c
        )

        # Sample c and u
        c = Normal(c_mean, c_std).rsample()
        u = Normal(u_mean, u_std).rsample()

        # Apply LD module to extract noise from u
        if self.use_ld and self.ld_module is not None:
            u_cleaned, aleatoric = self.ld_module(u)
        else:
            u_cleaned = u
            aleatoric = torch.zeros_like(u)

        # Decode: three reconstruction modes (c only, u only, c+u)
        # This allows c and u to both contribute to reconstruction separately
        x_recon_c, x_recon_u, x_recon_cu = self.decoder(
            c, u_cleaned, adj, batch_id, sample_id=self.sample_id
        )

        return c_mean, c_std, u_mean, u_std, c, u, u_cleaned, aleatoric, x_recon_c, x_recon_u, x_recon_cu

    def kl_divergence_sws(self, x_batch: Tensor, batch_id: Tensor, **kwargs):
        """
        Compute KL divergence with spatial GP priors for c+u (combined).

        For neighbor encoding, build adjacency matrices from neighbor coordinates.
        Only computes KL for c+u, not separately for c and u.
        """
        # Get nearest neighbor data
        y_nn, x_nn = self.get_nn_data(
            x_batch, k_neighbors=self.k_neighbors, **kwargs
        )[:2]
        y_nn, x_nn = y_nn.to(self.module_device), x_nn.to(self.module_device)

        v, n = self.train_dataset.series_shape.numel(), x_batch.size(-2)

        # Build adjacency matrices for neighbor sets
        adj_nn = build_neighbor_graphs(x_nn)

        # Encode neighbors with GCN encoder
        c_mean_q, c_std_q, u_mean_q, u_std_q = self.encoder(
            y_nn, adj_nn, sample_id=self.sample_id
        )

        # Re-arrange dimensions for KL computation
        c_mean_q = c_mean_q.permute(*range(c_mean_q.ndim - 3), -1, -3, -2)
        c_std_q = c_std_q.permute(*range(c_std_q.ndim - 3), -1, -3, -2)
        u_mean_q = u_mean_q.permute(*range(u_mean_q.ndim - 3), -1, -3, -2)
        u_std_q = u_std_q.permute(*range(u_std_q.ndim - 3), -1, -3, -2)

        # GP priors for c and u, then combine for c+u
        c_mean_p, c_cov_p = self.gp_c.prior(x_nn, are_neighbors=True)
        u_mean_p, u_cov_p = self.gp_u.prior(x_nn, are_neighbors=True)

        # KL for c and u separately
        kl_c = self._compute_kl_component(c_mean_q, c_std_q, c_mean_p, c_cov_p)
        kl_u = self._compute_kl_component(u_mean_q, u_std_q, u_mean_p, u_cov_p)

        kl = (kl_c + kl_u) / (v * n)
        return kl

    def _compute_kl_component(self, mean_q, std_q, mean_p, cov_p):
        """Compute KL divergence for one latent component (c or u)."""
        L_chol = psd_safe_cholesky(
            cov_p + self.jitter * torch.eye(self.k_neighbors, device=self.module_device)
        )

        mean_diff = (mean_q - mean_p).unsqueeze(-1)
        mahalanobis = torch.linalg.solve_triangular(L_chol, mean_diff, upper=False)
        mahalanobis = mahalanobis.square().sum(dim=(-1, -2))

        L_inv = torch.linalg.solve_triangular(
            L_chol,
            torch.eye(L_chol.size(-1), device=self.module_device),
            upper=False,
        )
        tmp = L_inv * std_q.unsqueeze(-2)
        trace = tmp.square().sum(dim=(-1, -2))

        log_det_cov_p = L_chol.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
        log_det_cov_q = std_q.log().sum(dim=-1)
        log_det = log_det_cov_p - log_det_cov_q

        kl = 0.5 * (mahalanobis + trace - self.k_neighbors) + log_det
        return kl.sum()


class DisentangledGPVAESpatial(DisentangledGPVAE):
    """
    Disentangled GPVAE with spatial structure support.
    
    Extends DisentangledGPVAE to handle spatial transcriptomics data with:
        - Gene expression (count) data
        - Spatial coordinate information
        - Disentangled latent variables c (shared) and u (sample-specific)
        - Single reconstruction path: from c+u_cleaned only
    """

    def expected_log_prob(self, y_batch: Tensor, y_rec: Tensor, **kwargs) -> Tensor:
        """
        Compute the expected log-likelihood term in the ELBO loss.
        """
        out_dist = self.decoder.output_distribution

        if out_dist == 'normal':
            sigma2_y = self.decoder.get_sigma2_y()
            loss = (y_batch - y_rec).square() / sigma2_y + torch.log(2 * torch.pi * sigma2_y)
            expected_lk = -0.5 * loss.sum() / len(self.train_dataset)
        elif out_dist == 'bernoulli':
            expected_lk = -nn.functional.binary_cross_entropy(
                input=y_rec, target=y_batch, reduction='sum'
            )
        else:
            raise NotImplementedError(f'Unrecognized output distribution: {out_dist}')

        scale = (y_batch.shape[:len(self.train_dataset.series_shape) + 1]).numel()
        scale = len(self.train_dataset) / scale

        return expected_lk * scale

    def loss_sws(self, y_batch: Tensor, x_batch: Tensor, 
                 y_rec: Tensor, batch_id: Tensor, beta=1.):
        """
        Compute negative ELBO = -log_likelihood + β * KL.
        
        If beta=0, skip KL computation entirely to save computation time.
        """
        lik = self.expected_log_prob(y_batch, y_rec)
        if beta == 0:
            # Skip KL computation when beta=0 to save computation time
            kl = torch.tensor(0.0, device=self.module_device)
        else:
            kl = self.kl_divergence_sws(x_batch, batch_id) / len(self.train_dataset)
        return lik, beta * kl

    @torch.no_grad()
    def sample_latent(self, y_batch: Tensor, adj: Tensor, batch_id: Tensor, sample=True):
        """
        Sample latent variables c and u, with LD noise extraction.
        
        Args:
            y_batch: Input data [B, G].
            adj: Adjacency matrix [B, B].
            batch_id: Batch ID [B].
            sample: Whether to sample or use mean.
            
        Returns:
            c, u_cleaned, aleatoric, c_mean, c_std, u_mean, u_std.
        """
        # Encode to disentangled latent distributions
        c_mean, c_std, u_mean, u_std = self.encoder(y_batch, adj, sample_id=self.sample_id)
        
        if sample:
            c = Normal(c_mean, c_std).sample()
            u = Normal(u_mean, u_std).sample()
        else:
            c = c_mean
            u = u_mean
        
        # Apply LD module to extract noise from u
        if self.use_ld and self.ld_module is not None:
            u_cleaned, aleatoric = self.ld_module(u)
        else:
            u_cleaned = u
            aleatoric = torch.zeros_like(u)
        
        return c, u_cleaned, aleatoric, c_mean, c_std, u_mean, u_std

