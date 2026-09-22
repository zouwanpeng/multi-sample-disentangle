"""
Disentangled GCN Decoder module.
"""
import torch
import numpy as np
from torch import nn, Tensor
import torch.nn.functional as F

from .gcn_layers import GCNLayer


class DisentangledGCNDecoder(nn.Module):
    """
    Per-sample independent GCN decoders.

    Design:
        - Each sample has its own decoder (parameters are NOT shared).
        - Three reconstruction modes:
          1. c-only: decode from shared latent c
          2. u-only: decode from sample-specific latent u (after LD cleaning)
          3. c+u: decode from fused representation using scNiche-style fusion
        - Output: reconstructed gene expression for each sample.
    """

    def __init__(
        self,
        dim_c: int,
        dim_u: int,
        n_batches: int,
        hidden_dims: list,
        output_dims_list: list,
        sigma2_y: float = 1.0,
        fix_variance: bool = False,
        activation: str = 'elu',
        norm_type: str = 'rms',
        use_norm: bool = True,
    ):
        """
        Args:
            dim_c (int): Dimension of shared latent variable c.
            dim_u (int): Dimension of sample-specific latent variable u (should equal dim_c).
            n_batches (int): Number of batches/samples (kept for compatibility, not used).
            hidden_dims (list): Hidden dimensions for GCN decoder [h1, h2, ...].
            output_dims_list (list): Output dimensions for each sample [G1, G2, ...].
            sigma2_y (float): Observation noise variance.
            fix_variance (bool): Whether to fix the variance (True) or learn it (False).
            activation (str): Activation function for hidden layers ('elu', 'relu', 'gelu', 'tanh', or None).
        """
        super().__init__()

        assert dim_c == dim_u, "scNiche-style fusion requires dim_c == dim_u for consistent decoder input dimensions."

        self.dim_c = dim_c
        self.dim_u = dim_u
        self.n_batches = n_batches
        self.n_samples = len(output_dims_list)
        self.output_distribution = "normal"
        self.output_dims_list = output_dims_list
        self.activation = activation
        self.norm_type = norm_type
        self.use_norm = use_norm

        # --------------------------------------------------------------------
        # Input dimension: unified as dim_c for all decoder branches
        # --------------------------------------------------------------------
        input_dim = dim_c

        # --------------------------------------------------------------------
        # scNiche-style fusion for c+u joint decoding
        # Simple concatenation + MLP fusion (like scNiche)
        # --------------------------------------------------------------------
        
        # Fusion MLP to combine c and u (scNiche style)
        fusion_hidden_dim = max(64, (dim_c + dim_u) // 2)
        
        if self.n_samples == 1:
            self.fusion_mlp = nn.Sequential(
                nn.Linear(dim_c + dim_u, fusion_hidden_dim),
                nn.BatchNorm1d(fusion_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(fusion_hidden_dim, dim_c),
                nn.BatchNorm1d(dim_c),
                nn.ReLU()
            )
            self.fusion_mlp_list = None
        else:
            self.fusion_mlp_list = nn.ModuleList()
            for _ in range(self.n_samples):
                fusion_mlp = nn.Sequential(
                    nn.Linear(dim_c + dim_u, dim_c),
                    nn.BatchNorm1d(dim_c),
                    nn.ReLU()
                )
                self.fusion_mlp_list.append(fusion_mlp)
            self.fusion_mlp = None

        # --------------------------------------------------------------------
        # Separate decoders for c and u (matching encoder structure):
        #   - c decoder: shared across all samples (corresponds to common encoder)
        #   - u decoder: sample-specific (corresponds to specific encoder)
        #   - cu decoder: can use shared or sample-specific (we use sample-specific for flexibility)
        # Input dimension is unified as dim_c for all decoder branches:
        #   - c branch: directly uses c [B, dim_c]
        #   - u branch: directly uses u [B, dim_u] (assumes dim_u == dim_c)
        #   - c+u branch: uses fused representation from scNiche-style fusion [B, dim_c]
        # --------------------------------------------------------------------

        if self.n_samples == 1:
            # Single sample: create separate decoders for c and u
            layer_dims = [input_dim] + hidden_dims
            
            # Shared decoder for c (even for single sample, keep structure consistent)
            c_gcn_layers = nn.ModuleList()
            for i in range(len(layer_dims) - 1):
                c_gcn_layers.append(
                    GCNLayer(layer_dims[i], layer_dims[i + 1],
                             use_norm=self.use_norm, norm_type=self.norm_type, activation=activation)
                )
            c_output_layer = GCNLayer(layer_dims[-1], output_dims_list[0], 
                                     use_norm=self.use_norm, norm_type=self.norm_type, activation=None)
            
            # Sample-specific decoder for u
            u_gcn_layers = nn.ModuleList()
            for i in range(len(layer_dims) - 1):
                u_gcn_layers.append(
                    GCNLayer(layer_dims[i], layer_dims[i + 1],
                             use_norm=self.use_norm, norm_type=self.norm_type, activation=activation)
                )
            u_output_layer = GCNLayer(layer_dims[-1], output_dims_list[0], 
                                     use_norm=self.use_norm, norm_type=self.norm_type, activation=None)
            
            # Decoder for c+u (use sample-specific for flexibility)
            cu_gcn_layers = nn.ModuleList()
            for i in range(len(layer_dims) - 1):
                cu_gcn_layers.append(
                    GCNLayer(layer_dims[i], layer_dims[i + 1],
                             use_norm=self.use_norm, norm_type=self.norm_type, activation=activation)
                )
            cu_output_layer = GCNLayer(layer_dims[-1], output_dims_list[0], 
                                      use_norm=self.use_norm, norm_type=self.norm_type, activation=None)

            self.decoder_c = nn.ModuleDict({
                "gcn_layers": c_gcn_layers,
                "output_layer": c_output_layer,
            })
            self.decoder_u = nn.ModuleDict({
                "gcn_layers": u_gcn_layers,
                "output_layer": u_output_layer,
            })
            self.decoder_cu = nn.ModuleDict({
                "gcn_layers": cu_gcn_layers,
                "output_layer": cu_output_layer,
            })
            self.decoders_c = None
            self.decoders_u = None
            self.decoders_cu = None
        else:
            # Multiple samples: 
            # - Shared GCN layers for c, but sample-specific output layers (since gene numbers differ)
            # - Sample-specific decoders for u (each sample has its own)
            # - Sample-specific decoders for cu (each sample has its own)
            
            # Shared GCN layers for c (all samples share the hidden layers)
            layer_dims = [input_dim] + hidden_dims
            c_gcn_layers = nn.ModuleList()
            for i in range(len(layer_dims) - 1):
                c_gcn_layers.append(
                    GCNLayer(layer_dims[i], layer_dims[i + 1],
                             use_norm=self.use_norm, norm_type=self.norm_type, activation=activation)
                )
            # Sample-specific output layers for c (since each sample has different number of genes)
            c_output_layers = nn.ModuleList()
            for sample_id in range(self.n_samples):
                c_output_layer = GCNLayer(layer_dims[-1], output_dims_list[sample_id], 
                                         use_norm=self.use_norm, norm_type=self.norm_type, activation=None)
                c_output_layers.append(c_output_layer)
            
            self.decoder_c = nn.ModuleDict({
                "gcn_layers": c_gcn_layers,  # Shared
                "output_layers": c_output_layers,  # Sample-specific
            })
            
            # Sample-specific decoders for u
            self.decoders_u = nn.ModuleList()
            for sample_id in range(self.n_samples):
                layer_dims = [input_dim] + hidden_dims
                u_gcn_layers = nn.ModuleList()
                for i in range(len(layer_dims) - 1):
                    u_gcn_layers.append(
                        GCNLayer(layer_dims[i], layer_dims[i + 1],
                                 use_norm=self.use_norm, norm_type=self.norm_type, activation=activation)
                    )
                u_output_layer = GCNLayer(layer_dims[-1], output_dims_list[sample_id], 
                                         use_norm=self.use_norm, norm_type=self.norm_type, activation=None)
                decoder_u = nn.ModuleDict({
                    "gcn_layers": u_gcn_layers,
                    "output_layer": u_output_layer,
                })
                self.decoders_u.append(decoder_u)
            
            # Sample-specific decoders for cu
            self.decoders_cu = nn.ModuleList()
            for sample_id in range(self.n_samples):
                layer_dims = [input_dim] + hidden_dims
                cu_gcn_layers = nn.ModuleList()
                for i in range(len(layer_dims) - 1):
                    cu_gcn_layers.append(
                        GCNLayer(layer_dims[i], layer_dims[i + 1],
                                 use_norm=self.use_norm, norm_type=self.norm_type, activation=activation)
                    )
                cu_output_layer = GCNLayer(layer_dims[-1], output_dims_list[sample_id], 
                                         use_norm=self.use_norm, norm_type=self.norm_type, activation=None)
                decoder_cu = nn.ModuleDict({
                    "gcn_layers": cu_gcn_layers,
                    "output_layer": cu_output_layer,
                })
                self.decoders_cu.append(decoder_cu)
            
            # In multi-sample case, decoder_c is shared, but decoder_u and decoder_cu are sample-specific
            # So decoder_c is already created above, but decoder_u and decoder_cu should be None
            self.decoder_u = None
            self.decoder_cu = None

        # --------------------------------------------------------------------
        # Observation noise variance (optional, if you need it in the loss)
        # --------------------------------------------------------------------
        self.fix_variance = fix_variance
        if fix_variance:
            self.register_buffer("sigma2_y", torch.tensor(sigma2_y))
        else:
            self.log_sigma2_y = nn.Parameter(torch.tensor(np.log(sigma2_y)))

    def get_sigma2_y(self) -> Tensor:
        """Return observation noise variance."""
        if self.fix_variance:
            return self.sigma2_y
        else:
            return torch.exp(self.log_sigma2_y)

    def forward(
        self,
        c: Tensor,
        u: Tensor,
        adj: Tensor,
        batch_id: Tensor,
        sample_id: int,
    ) -> tuple:
        """
        Forward pass: decode with three reconstruction modes for a given sample.

        Args:
            c (Tensor): Shared latent variable, shape [B, dim_c].
            u (Tensor): Sample-specific latent variable (should be u_cleaned after LD noise removal), 
                       shape [B, dim_u].
            adj (Tensor): Normalized adjacency matrix, shape [B, B].
            batch_id (Tensor): Batch indices (kept for compatibility, not used).
            sample_id (int): Sample index (0, 1, ...).

        Returns:
            x_recon_c (Tensor): Reconstruction from c only, shape [B, G_sample].
            x_recon_u (Tensor): Reconstruction from u only, shape [B, G_sample].
            x_recon_cu (Tensor): Reconstruction from c+u, shape [B, G_sample].
        """
        def decode_from_c(z: Tensor) -> Tensor:
            """Decode from shared latent c using shared GCN layers and sample-specific output layer."""
            decoder = self.decoder_c
            
            h = z
            for gcn_layer in decoder["gcn_layers"]:
                h = gcn_layer(h, adj)
            
            # Use sample-specific output layer
            if self.n_samples == 1:
                output_layer = decoder["output_layer"]
            else:
                output_layer = decoder["output_layers"][sample_id]
            
            x_recon = output_layer(h, adj)  # [B, G_sample]
            return x_recon

        def decode_from_u(z: Tensor) -> Tensor:
            """Decode from sample-specific latent u using sample-specific decoder."""
            if self.n_samples == 1:
                decoder = self.decoder_u
            else:
                decoder = self.decoders_u[sample_id]
            
            h = z
            for gcn_layer in decoder["gcn_layers"]:
                h = gcn_layer(h, adj)
            x_recon = decoder["output_layer"](h, adj)  # [B, G_sample]
            return x_recon

        def decode_with_scniche_fusion(c_input: Tensor, u_input: Tensor) -> Tensor:
            """
            Decode using scNiche-style simple concatenation + MLP fusion.
            
            scNiche fusion mechanism:
            1. Simple concatenation: [c, u]
            2. MLP transformation to get fused representation
            3. Decode from fused representation using sample-specific decoder
            
            Args:
                c_input: Shared latent variable c [B, dim_c]
                u_input: Sample-specific latent variable u [B, dim_u]
            
            Returns:
                Reconstructed output [B, G_sample]
            """
            # Get fusion MLP for this sample
            if self.n_samples == 1:
                fusion_mlp = self.fusion_mlp
                decoder = self.decoder_cu
            else:
                fusion_mlp = self.fusion_mlp_list[sample_id]
                decoder = self.decoders_cu[sample_id]
            
            # scNiche-style fusion: simple concatenation + MLP
            cu_concat = torch.cat([c_input, u_input], dim=-1)  # [B, dim_c + dim_u]
            fused_z = fusion_mlp(cu_concat)  # [B, dim_c]
            
            # Decode the fused representation using sample-specific decoder
            h = fused_z
            for gcn_layer in decoder["gcn_layers"]:
                h = gcn_layer(h, adj)
            x_recon = decoder["output_layer"](h, adj)  # [B, G_sample]
            return x_recon

        # --------------------------------------------------------------------
        # Three reconstruction modes:
        #   1. c-only: shared information only, uses shared decoder (corresponds to common encoder)
        #   2. u-only: sample-specific information, uses sample-specific decoder (corresponds to specific encoder)
        #   3. c+u: joint decoding using scNiche-style fusion, uses sample-specific decoder
        # --------------------------------------------------------------------

        # 1. Reconstruction from c only (shared information, batch-free)
        # Uses shared decoder (matching shared encoder)
        x_recon_c = decode_from_c(c)

        # 2. Reconstruction from u only (sample-specific information)
        # Uses sample-specific decoder (matching sample-specific encoder)
        x_recon_u = decode_from_u(u)

        # 3. Reconstruction from c+u using scNiche-style fusion
        # Fusion process: concat([c, u]) -> MLP -> fused_z -> sample-specific GCN decoder
        # x_recon_cu = decode_with_scniche_fusion(c, u)  # 注释掉：不需要整合嵌入

        # return x_recon_c, x_recon_u, x_recon_cu
        x_recon_cu = torch.zeros_like(x_recon_c)  # 占位符，保持返回值格式
        return x_recon_c, x_recon_u, x_recon_cu
    
    def get_fusion_embedding(
        self,
        c: Tensor,
        u: Tensor,
        adj: Tensor,
        batch_id: Tensor,
        sample_id: int,
    ) -> Tensor:
        """
        Extract scNiche-style fusion embedding for c+u (for downstream analysis).
        
        Uses simple concatenation + MLP fusion to combine c and u representations,
        returning the fused latent representation directly.
        
        Note: This should use u_cleaned (after LD noise removal) to be consistent with
        the training procedure where decoder receives u_cleaned.
        
        Args:
            c (Tensor): Shared latent variable, shape [B, dim_c].
            u (Tensor): Sample-specific latent variable (should be u_cleaned), shape [B, dim_u].
            adj (Tensor): Normalized adjacency matrix, shape [B, B] (kept for compatibility, not used).
            batch_id (Tensor): Batch indices (kept for compatibility, not used).
            sample_id (int): Sample index (0, 1, ...).
        
        Returns:
            fused_z (Tensor): scNiche-style fusion latent representation, shape [B, dim_c].
        """
        # Select fusion MLP for this sample
        if self.n_samples == 1:
            fusion_mlp = self.fusion_mlp
        else:
            fusion_mlp = self.fusion_mlp_list[sample_id]
        
        # scNiche-style fusion: simple concatenation + MLP
        cu_concat = torch.cat([c, u], dim=-1)  # [B, dim_c + dim_u]
        fused_z = fusion_mlp(cu_concat)  # [B, dim_c]
        
        return fused_z
