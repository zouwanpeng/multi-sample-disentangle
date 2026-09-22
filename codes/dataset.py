"""
Dataset classes for spatial transcriptomics data.
"""
import torch
import numpy as np
import faiss
from torch.utils.data import Dataset


class NNUtil_Spatial(torch.nn.Module):
    """
    Spatial Nearest Neighbor Search Utility using FAISS
    
    Provides fast k-NN search over 2D spatial coordinates for spatial transcriptomics data.
    Used to find spatial neighbors for GP prior computation in GPVAE.
    """

    def __init__(self, k: int, dim: int, search_device: str = 'cpu', 
                 preferred_nnlib: str = 'faiss', metric: str = 'L2'):
        """
        Initialize the nearest neighbor search module
        
        Args:
            k (int): Number of nearest neighbors to retrieve
            dim (int): Dimensionality of spatial coordinates (typically 2 for spatial transcriptomics)
            search_device (str): Device for search, 'cpu' or 'cuda'
            preferred_nnlib (str): NN library to use (currently only 'faiss' supported)
            metric (str): Distance metric (currently only 'L2' supported)
        """
        super().__init__()
        self.k = k
        self.D = dim
        self.metric = metric
        self.search_device = search_device
        self.nnlib = preferred_nnlib

        # Initialize FAISS index
        if preferred_nnlib == 'faiss':
            if search_device == 'cpu':
                if metric == 'L2':
                    self.index = faiss.IndexFlatL2(self.D)
                else:
                    raise NotImplementedError("Only 'L2' metric is implemented for CPU.")
            elif search_device == 'cuda':
                res = faiss.StandardGpuResources()
                if metric == 'L2':
                    self.index = faiss.GpuIndexFlatL2(res, self.D)
                else:
                    raise NotImplementedError("Only 'L2' metric is implemented for GPU.")
            else:
                raise ValueError(f"Unsupported search_device: {search_device}. Use 'cpu' or 'cuda'.")
        else:
            raise ValueError(f"Unsupported NN library: {preferred_nnlib}. Currently only 'faiss' is supported.")

    def build_nn_idx(self, coordinates: torch.Tensor):
        """
        Build the nearest neighbor index from spatial coordinates
        Should be called once with all training spatial coordinates
        
        Args:
            coordinates (Tensor): Spatial coordinates, shape [N, D]
                                 For spatial transcriptomics, typically [N, 2]
        """
        assert coordinates.shape[-1] == self.D, \
            f"Expected {self.D}-dimensional coordinates, got {coordinates.shape[-1]}"
        coords_np = coordinates.cpu().numpy().astype(np.float32)
        self.index.add(coords_np)

    def find_nn_idx(self, query_coordinates: torch.Tensor, k: int = None):
        """
        Perform k-NN search for given query coordinates
        
        Args:
            query_coordinates (Tensor): Query spatial coordinates, shape [M, D]
            k (int, optional): Number of neighbors to retrieve. If None, uses self.k
            
        Returns:
            indices (Tensor): Neighbor indices, shape [M, k]
            distances (Tensor): Neighbor distances, shape [M, k]
        """
        k = self.k if k is None else k
        assert k > 0, f'k must be greater than 0, got {k}.'
        query_np = query_coordinates.cpu().numpy().astype(np.float32)
        distances, indices = self.index.search(query_np, k)
        return torch.from_numpy(indices).long(), torch.from_numpy(distances).float()


class NNDataset(Dataset):
    """
    Spatial omics dataset for DisentangledGPVAE training
    
    Includes:
        - Gene expression count matrix (or features)
        - Spatial coordinates
        - Efficient k-NN lookup via NNUtil_Spatial
    """

    def __init__(self, data_dict: dict, series_shape=torch.Size([]), data_device='cuda:0',
                 search_device: str = 'cpu', k_neighbors=None):
        """
        Args:
            data_dict (dict): Dictionary containing:
                - 'count': Gene expression matrix [N, G]
                - 'spatial': Spatial coordinates [N, 2]
            series_shape (torch.Size): Optional extra shape prefix for batching
            data_device (str): Device to store actual data (usually 'cuda')
            search_device (str): Device for FAISS k-NN search ('cpu' or 'cuda')
            k_neighbors (int): Number of spatial neighbors to retrieve for GP prior
        """
        super().__init__()
        self.Y = torch.as_tensor(data_dict['count'], dtype=torch.get_default_dtype(), device=data_device)
        self.X = torch.as_tensor(data_dict['spatial'], dtype=torch.float32, device=data_device).contiguous()
        self.series_shape = series_shape

        # Input validation
        assert self.X.shape[:-2] == torch.Size([]) or self.series_shape == self.X.shape[:-2], \
            f"Inconsistent spatial dimensions."
        assert self.series_shape == self.Y.shape[:len(self.series_shape)], \
            f"Inconsistent batch shape between count and spatial data."

        self.data_device = data_device
        self.search_device = None if search_device is None else search_device

        # Initialize k-NN structure for spatial GP prior
        self.nn_util = NNUtil_Spatial(k=k_neighbors, dim=2, search_device=self.search_device)
        self.nn_util.build_nn_idx(self.X)
        
        # Precompute sequential k-NN indices (for all data points)
        self.seq_nn_idx = self.nn_util.find_nn_idx(self.X, k=self.nn_util.k)[0]

    def __len__(self):
        """Return dataset size along the sample dimension"""
        return self.Y.size(len(self.series_shape))

    def __getitem__(self, idx):
        """
        Returns data, spatial coordinates, AND index
        """
        if self.series_shape == torch.Size([]):
            return self.Y[idx], self.X[idx], idx
        
        slices_y = [slice(None)] * len(self.series_shape) + [idx]
        sample_y = self.Y[slices_y]
        sample_x = self.X[idx] if self.X.ndim == 2 else self.X[slices_y]
        return sample_y, sample_x, idx

    def gather(self, nn_idx):
        """
        Gather nearest neighbor features for each sample using precomputed indices
        Used in DisentangledGPVAE.kl_divergence_sws() for GP prior computation
        
        Args:
            nn_idx (Tensor): Neighbor indices, shape [..., k]
            
        Returns:
            y_gather (Tensor): Gathered gene expression [..., k, G]
            x_gather (Tensor): Gathered spatial coordinates [..., k, 2]
        """
        def _gather(idx, tensor):
            idx_ndim = idx.ndim
            new_shape = idx.shape + torch.Size([1] * (tensor.ndim + 1 - idx_ndim))
            idx_expand = idx.reshape(new_shape)
            data_expand = tensor.unsqueeze(idx_ndim - 2)
            return torch.take_along_dim(data_expand, idx_expand, dim=idx_ndim - 1)

        nn_idx = nn_idx.to(self.data_device)
        nn_idx_expand = nn_idx.expand(*self.series_shape, -1, -1)
        y_gather = _gather(nn_idx_expand, self.Y.to(self.data_device))
        x_gather = _gather(nn_idx, self.X.to(self.data_device))
        return y_gather, x_gather

