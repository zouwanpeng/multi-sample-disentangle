"""
Disentangled GCN Encoder module.
"""
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Tuple, Optional

from .gcn_layers import GCNLayer


class DisentangledGCNEncoder(nn.Module):
    """
    Disentangled GCN Encoder with Mini-batch Support
    Handles both regular mini-batch data and neighbor data for KL computation
    """
    
    def __init__(self, dim_c, dim_u, input_dims_list, proj_dim,
                 hidden_dims_common, hidden_dims_specific,
                 activation: Optional[str] = 'elu',
                 norm_type: Optional[str] = 'rms',
                 use_norm: bool = True,
                 separate_mean_std: bool = False):
        """
        Args:
            separate_mean_std: If True, use separate output layers for mean and std.
                              If False (default), use single layer outputting 2*dim (more efficient).
        """
        super().__init__()
        
        self.dim_c = dim_c
        self.dim_u = dim_u
        self.n_samples = len(input_dims_list)
        self.proj_dim = proj_dim
        self.activation = activation
        self.norm_type = norm_type
        self.use_norm = use_norm
        self.separate_mean_std = separate_mean_std
        
        # Sample-specific projection layers
        self.projection_layers = nn.ModuleList([
            nn.Linear(input_dim, proj_dim, bias=False) for input_dim in input_dims_list
        ])
        
        for layer in self.projection_layers:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        
        # Shared common GCN encoder for c
        if separate_mean_std:
            # Separate branches: shared base + separate mean/std heads
            # Base layers (shared)
            layer_dims_base = [proj_dim] + hidden_dims_common
            self.common_gcn_base = nn.ModuleList()
            for i in range(len(layer_dims_base) - 1):
                self.common_gcn_base.append(
                    GCNLayer(layer_dims_base[i], layer_dims_base[i + 1],
                             use_norm=self.use_norm, norm_type=self.norm_type, activation=activation)
                )
            # Separate output heads
            last_hidden = layer_dims_base[-1]
            self.common_mean_head = GCNLayer(last_hidden, dim_c, use_norm=self.use_norm, 
                                             norm_type=self.norm_type, activation=None)
            self.common_std_head = GCNLayer(last_hidden, dim_c, use_norm=self.use_norm,
                                            norm_type=self.norm_type, activation=None)
        else:
            # Single output layer (original design)
            layer_dims_common = [proj_dim] + hidden_dims_common + [2 * dim_c]
            self.common_gcn_layers = nn.ModuleList()
            for i in range(len(layer_dims_common) - 1):
                # Last layer: no activation (output layer)
                # Other layers: use specified activation
                use_activation = activation if i < len(layer_dims_common) - 2 else None
                self.common_gcn_layers.append(
                    GCNLayer(layer_dims_common[i], layer_dims_common[i + 1],
                             use_norm=self.use_norm, norm_type=self.norm_type, activation=use_activation)
                )
        
        # Sample-specific GCN encoders for u
        if separate_mean_std:
            # Separate branches for each sample
            layer_dims_base = [proj_dim] + hidden_dims_specific
            self.specific_gcn_encoders = nn.ModuleList()
            for _ in range(self.n_samples):
                # Base layers (shared)
                gcn_base = nn.ModuleList()
                for i in range(len(layer_dims_base) - 1):
                    gcn_base.append(
                        GCNLayer(layer_dims_base[i], layer_dims_base[i + 1],
                                 use_norm=self.use_norm, norm_type=self.norm_type, activation=activation)
                    )
                # Separate output heads
                last_hidden = layer_dims_base[-1]
                mean_head = GCNLayer(last_hidden, dim_u, use_norm=self.use_norm,
                                     norm_type=self.norm_type, activation=None)
                std_head = GCNLayer(last_hidden, dim_u, use_norm=self.use_norm,
                                    norm_type=self.norm_type, activation=None)
                self.specific_gcn_encoders.append(nn.ModuleDict({
                    'base': gcn_base,
                    'mean_head': mean_head,
                    'std_head': std_head
                }))
        else:
            # Single output layer (original design)
            layer_dims_specific = [proj_dim] + hidden_dims_specific + [2 * dim_u]
            self.specific_gcn_encoders = nn.ModuleList()
            for _ in range(self.n_samples):
                gcn_layers = nn.ModuleList()
                for i in range(len(layer_dims_specific) - 1):
                    # Last layer: no activation (output layer)
                    # Other layers: use specified activation
                    use_activation = activation if i < len(layer_dims_specific) - 2 else None
                    gcn_layers.append(
                        GCNLayer(layer_dims_specific[i], layer_dims_specific[i + 1],
                                 use_norm=self.use_norm, norm_type=self.norm_type, activation=use_activation)
                    )
                self.specific_gcn_encoders.append(gcn_layers)
    
    def forward(self, x: Tensor, adj: Tensor, sample_id: int, adj_c: Tensor = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Forward pass - handles both 2D and 3D inputs
        
        Args:
            x (Tensor): Input data
                       [B, G_sample] for regular mini-batch
                       [B, k, G_sample] for neighbor data
            adj (Tensor): Normalized adjacency matrix for u encoding
                         [B, B] for regular mini-batch
                         [B, k, k] for neighbor data
            sample_id (int): Sample index
            adj_c (Tensor, optional): Unified adjacency matrix for c encoding
                                    If None, uses adj for both c and u
        
        Returns:
            c_mean, c_std, u_mean, u_std
        """
        # Projection
        if x.ndim == 2:
            # Regular: [B, G]
            h = self.projection_layers[sample_id](x)
        elif x.ndim == 3:
            # Neighbors: [B, k, G] or [v, n, k, G]
            shape = x.shape
            x_flat = x.reshape(-1, shape[-1])  # [B*k, G] or [v*n*k, G]
            h_flat = self.projection_layers[sample_id](x_flat)
            h = h_flat.reshape(*shape[:-1], self.proj_dim)  # [B, k, proj_dim] or [v, n, k, proj_dim]
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")
        
        # Apply activation after projection
        if self.activation == 'elu':
            h = F.elu(h)
        elif self.activation == 'relu':
            h = F.relu(h)
        elif self.activation == 'gelu':
            h = F.gelu(h)
        # If None, no activation
        
        # Encode with GCN
        # Determine adjacency matrices for c and u encoding
        adj_for_c = adj_c if adj_c is not None else adj  # c uses unified graph if provided, otherwise sample-specific graph
        adj_for_u = adj  # u always uses sample-specific graph
        
        if self.separate_mean_std:
            # Separate mean and std branches
            # For c (uses unified graph or sample-specific graph)
            h_c_base = self._forward_gcn(h, adj_for_c, self.common_gcn_base)
            c_mean = self.common_mean_head(h_c_base, adj_for_c)
            c_std_logit = self.common_std_head(h_c_base, adj_for_c)
            c_std = F.softplus(c_std_logit)
            c_std = torch.clamp(c_std, min=1e-4)
            
            # For u (uses sample-specific graph)
            encoder_u = self.specific_gcn_encoders[sample_id]
            h_u_base = self._forward_gcn(h, adj_for_u, encoder_u['base'])
            u_mean = encoder_u['mean_head'](h_u_base, adj_for_u)
            u_std_logit = encoder_u['std_head'](h_u_base, adj_for_u)
            u_std = F.softplus(u_std_logit)
            u_std = torch.clamp(u_std, min=1e-4)
        else:
            # Single output layer (original)
            c_params = self._forward_gcn(h, adj_for_c, self.common_gcn_layers)
            u_params = self._forward_gcn(h, adj_for_u, self.specific_gcn_encoders[sample_id])
            
            # Extract parameters
            c_mean = c_params[..., :self.dim_c]
            c_std = F.softplus(c_params[..., self.dim_c:])
            c_std = torch.clamp(c_std, min=1e-4)
            
            u_mean = u_params[..., :self.dim_u]
            u_std = F.softplus(u_params[..., self.dim_u:])
            u_std = torch.clamp(u_std, min=1e-4)
        
        return c_mean, c_std, u_mean, u_std
    
    def _forward_gcn(self, h: Tensor, adj: Tensor, gcn_layers: nn.ModuleList) -> Tensor:
        """Forward through GCN layers, handles both 2D and 3D inputs"""
        # Activation is now handled inside GCNLayer, so no need to add it here
        for gcn_layer in gcn_layers:
            h = gcn_layer(h, adj)
        return h
    
    def encode_to_c(self, x: Tensor, adj: Tensor, sample_id: int, adj_c: Tensor = None) -> Tuple[Tensor, Tensor]:
        """
        Encode to c using GCN
        
        Args:
            x (Tensor): Input data [B, G_sample]
            adj (Tensor): Normalized adjacency matrix [B, B] (fallback)
            sample_id (int): Sample index
            adj_c (Tensor, optional): Unified adjacency matrix for c encoding
        
        Returns:
            c_mean (Tensor): [B, dim_c]
            c_std (Tensor): [B, dim_c]
        """
        h = self.projection_layers[sample_id](x)
        if self.activation == 'elu':
            h = F.elu(h)
        elif self.activation == 'relu':
            h = F.relu(h)
        elif self.activation == 'gelu':
            h = F.gelu(h)
        
        # Use unified graph or sample-specific graph for c encoding
        adj_for_c = adj_c if adj_c is not None else adj
        
        if self.separate_mean_std:
            h_c_base = self._forward_gcn(h, adj_for_c, self.common_gcn_base)
            c_mean = self.common_mean_head(h_c_base, adj_for_c)
            c_std_logit = self.common_std_head(h_c_base, adj_for_c)
            c_std = F.softplus(c_std_logit)
            c_std = torch.clamp(c_std, min=1e-4)
        else:
            c_params = self._forward_gcn(h, adj_for_c, self.common_gcn_layers)
            c_mean = c_params[..., :self.dim_c]
            c_std = F.softplus(c_params[..., self.dim_c:])
            c_std = torch.clamp(c_std, min=1e-4)
        
        return c_mean, c_std
    
    def encode_to_u(self, x: Tensor, adj: Tensor, sample_id: int) -> Tuple[Tensor, Tensor]:
        """
        Encode to u using GCN
        
        Args:
            x (Tensor): Input data [B, G_sample]
            adj (Tensor): Normalized adjacency matrix [B, B]
            sample_id (int): Sample index
        
        Returns:
            u_mean (Tensor): [B, dim_u]
            u_std (Tensor): [B, dim_u]
        """
        h = self.projection_layers[sample_id](x)
        if self.activation == 'elu':
            h = F.elu(h)
        elif self.activation == 'relu':
            h = F.relu(h)
        elif self.activation == 'gelu':
            h = F.gelu(h)
        
        if self.separate_mean_std:
            encoder_u = self.specific_gcn_encoders[sample_id]
            h_u_base = self._forward_gcn(h, adj, encoder_u['base'])
            u_mean = encoder_u['mean_head'](h_u_base, adj)
            u_std_logit = encoder_u['std_head'](h_u_base, adj)
            u_std = F.softplus(u_std_logit)
            u_std = torch.clamp(u_std, min=1e-4)
        else:
            u_params = self._forward_gcn(h, adj, self.specific_gcn_encoders[sample_id])
            u_mean = u_params[..., :self.dim_u]
            u_std = F.softplus(u_params[..., self.dim_u:])
            u_std = torch.clamp(u_std, min=1e-4)
        
        return u_mean, u_std

