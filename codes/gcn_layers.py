"""
GCN layers and graph construction utilities.
"""
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    Faster and simpler than LayerNorm (no mean subtraction)
    
    Formula: output = input / sqrt(mean(input^2) + eps) * scale
    """
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor [..., dim]
        Returns:
            normalized tensor [..., dim]
        """
        # Compute RMS (Root Mean Square)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        return x / rms * self.scale


class GCNLayer(nn.Module):
    """
    Simple GCN layer for spatial transcriptomics
    Implements: h_out = activation(A_norm * h_in * W)
    Uses RMSNorm for normalization (faster and more stable than LayerNorm)
    """
    def __init__(self, in_dim: int, out_dim: int, use_norm: bool = True, 
                 norm_type: str = 'rms', activation: str = 'elu'):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.kaiming_normal_(self.linear.weight, nonlinearity='relu')
        
        # Add normalization (RMSNorm by default, can use LayerNorm)
        self.use_norm = use_norm
        self.norm_type = norm_type
        if use_norm:
            if norm_type == 'rms':
                self.norm = RMSNorm(out_dim)
            elif norm_type == 'layer':
                self.norm = nn.LayerNorm(out_dim)
            else:
                raise ValueError(f"Unsupported norm_type: {norm_type}. Use 'rms' or 'layer'.")
        
        # Add activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'none' or activation is None:
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Unsupported activation: {activation}. Use 'relu', 'gelu', 'tanh', or 'none'.")
    
    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        """
        Args:
            x: Node features [N, in_dim] or [B, N, in_dim]
            adj: Normalized adjacency matrix [N, N] or [B, N, N]
        Returns:
            out: Aggregated features [N, out_dim] or [B, N, out_dim]
        """
        # Spatial aggregation: A * X
        if x.ndim == 2:
            h = torch.mm(adj, x)  # [N, in_dim]
        elif x.ndim == 3:
            h = torch.bmm(adj, x)  # [B, N, in_dim]
        else:
            raise ValueError(f"Unsupported input dimension: {x.ndim}")
        
        # Linear transformation
        out = self.linear(h)
        
        # Normalization (before activation)
        if self.use_norm:
            out = self.norm(out)
        
        # Activation function
        out = self.activation(out)
        
        return out


def build_spatial_graph(spatial_coords: Tensor, k_neighbors: int = 10) -> Tensor:
    """
    Build normalized adjacency matrix from spatial coordinates using k-NN
    
    Args:
        spatial_coords (Tensor): Spatial coordinates [N, 2]
        k_neighbors (int): Number of nearest neighbors
    
    Returns:
        adj_norm (Tensor): Normalized adjacency matrix [N, N]
    """
    N = spatial_coords.size(0)
    device = spatial_coords.device
    
    # Compute pairwise distances
    dist_matrix = torch.cdist(spatial_coords, spatial_coords, p=2)
    
    # Build k-NN graph
    _, indices = torch.topk(dist_matrix, k=k_neighbors + 1, largest=False, dim=1)
    indices = indices[:, 1:]  # Remove self
    
    # Build adjacency matrix
    adj = torch.zeros(N, N, device=device)
    for i in range(N):
        adj[i, indices[i]] = 1.0
    
    # Make symmetric
    adj = (adj + adj.t()) / 2.0
    
    # Add self-loops
    adj = adj + torch.eye(N, device=device)
    
    # Normalize: D^{-1/2} A D^{-1/2}
    degree = adj.sum(dim=1)
    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
    
    D_inv_sqrt = torch.diag(degree_inv_sqrt)
    adj_norm = torch.mm(torch.mm(D_inv_sqrt, adj), D_inv_sqrt)
    
    return adj_norm


def build_neighbor_graphs(spatial_neighbors: Tensor) -> Tensor:
    """
    Build adjacency matrices for neighbor sets
    Used when encoding k-NN neighbors in KL divergence computation
    
    Args:
        spatial_neighbors (Tensor): Neighbor coordinates
                                   [B, k, 2] or [v, n, k, 2]
    
    Returns:
        adj_neighbors (Tensor): Adjacency matrices [B, k, k] or [v, n, k, k]
    """
    original_shape = spatial_neighbors.shape[:-2]  # [B] or [v, n]
    k = spatial_neighbors.shape[-2]
    device = spatial_neighbors.device
    
    # Flatten batch dimensions
    spatial_flat = spatial_neighbors.reshape(-1, k, 2)  # [num_groups, k, 2]
    num_groups = spatial_flat.size(0)
    
    adj_list = []
    
    for i in range(num_groups):
        coords = spatial_flat[i]  # [k, 2]
        
        # Compute pairwise distances
        dist = torch.cdist(coords, coords, p=2)  # [k, k]
        
        # Build adjacency with Gaussian similarity
        adj = torch.exp(-dist ** 2 / (2 * 1.0 ** 2))
        
        # Add self-loops
        adj = adj + torch.eye(k, device=device)
        
        # Normalize
        degree = adj.sum(dim=1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
        
        D_inv_sqrt = torch.diag(degree_inv_sqrt)
        adj_norm = torch.mm(torch.mm(D_inv_sqrt, adj), D_inv_sqrt)
        
        adj_list.append(adj_norm)
    
    # Stack and reshape
    adj_neighbors = torch.stack(adj_list, dim=0)  # [num_groups, k, k]
    adj_neighbors = adj_neighbors.reshape(*original_shape, k, k)
    
    return adj_neighbors


def extract_subgraph(indices: Tensor, full_adj: Tensor) -> Tensor:
    """
    Extract subgraph adjacency matrix for mini-batch
    
    Args:
        indices (Tensor): Node indices [B]
        full_adj (Tensor): Full adjacency [N, N]
    
    Returns:
        subgraph_adj (Tensor): Subgraph adjacency [B, B]
    """
    subgraph_adj = full_adj[indices][:, indices]
    
    # Renormalize
    degree = subgraph_adj.sum(dim=1)
    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
    degree_inv_sqrt[degree == 0] = 0.0
    
    D_inv_sqrt = torch.diag(degree_inv_sqrt)
    subgraph_adj_norm = torch.mm(torch.mm(D_inv_sqrt, subgraph_adj), D_inv_sqrt)
    
    return subgraph_adj_norm

