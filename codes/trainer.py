"""
Joint training framework for disentangled models.
"""
from gpytorch.kernels import RBFKernel
from gpytorch.means import ZeroMean
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import NNDataset
from .decoder import DisentangledGCNDecoder
from .encoder import DisentangledGCNEncoder
from .gcn_layers import build_spatial_graph, extract_subgraph
from .gp_modules import GP
from .gpvae import DisentangledGPVAESpatial
from .losses import (
    HSIC,
    MNNDiscriminator,
    compute_batch_mnn_pairs,
    enhanced_mnn_mutual_information_loss,
    mnn_mutual_information_loss,
)
from .mnn import evaluate_mnn_matches, find_mnn_pairs_from_c, find_mnn_pairs_global


class DisentangledJointTrainer(nn.Module):
    """
    Joint training framework for disentangled models.
    Supports multiple samples (2 or more).
    """
    
    def __init__(self, adata_concat,
                batch_name: str = 'batch',
                dim_c: int = 20,
                dim_u: int = 20,
                proj_dim: int = 1024,
                hidden_dims_enc_common: list = [512, 256],
                hidden_dims_enc_specific: list = [512, 256],
                hidden_dims_dec: list = [256, 512],
                k_neighbors: int = 10,
                activation: str = 'elu',
                enc_norm_type: str = 'rms',
                dec_norm_type: str = 'rms',
                enc_use_norm: bool = True,
                dec_use_norm: bool = True,
                separate_mean_std: bool = False,
                mnn_k: int = 5,
                n_top_genes: int = 200,
                use_ld: bool = True,
                ld_n_round: int = 3,
                lambda_batch_cls: float = 1.0,
                warmup_epochs: int = 50,
                cell_type_key: str = 'cell_type',
                device='cuda:0',
                print_mnn_accuracy: bool = True):
        super().__init__()
        
        # Validate input - accept concatenated adata
        import anndata as ad
        if not isinstance(adata_concat, ad.AnnData):
            raise ValueError("adata_concat must be an AnnData object (concatenated)")
        
        if batch_name not in adata_concat.obs.columns:
            raise ValueError(f"batch_name '{batch_name}' not found in adata_concat.obs")
        
        # Store concatenated adata
        self.adata_concat = adata_concat
        self.batch_name = batch_name
        
        # Split adata into batches
        batch_list = adata_concat.obs[batch_name]
        self.section_ids = np.array(batch_list.unique())
        self.n_samples = len(self.section_ids)
        self.n_batches = self.n_samples
        
        # Split adata into list for processing
        self.adata_list = []
        for section_id in self.section_ids:
            adata_batch = adata_concat[adata_concat.obs[batch_name] == section_id].copy()
            self.adata_list.append(adata_batch)
        
        self.device = device
        self.dim_c = dim_c
        self.dim_u = dim_u
        self.k_neighbors = k_neighbors
        self.use_ld = use_ld
        self.mnn_k = mnn_k
        self.n_top_genes = n_top_genes
        self.lambda_batch_cls = lambda_batch_cls
        self.warmup_epochs = warmup_epochs
        self.cell_type_key = cell_type_key
        self.print_mnn_accuracy = print_mnn_accuracy
        self.MIN_STD_CLAMP = 1e-4
        
        # Loss history for plotting
        self.loss_history = {
            'epoch': [],
            'total': [],
            'recon_avg': [],
            'kl_avg': [],
            'hsic': [],
            'batch_cls': [],
            'aleatoric_reg': [],
            'mnn_mi': [],
        }
        # Add per-sample loss tracking
        for i in range(self.n_samples):
            self.loss_history[f'recon_c{i}'] = []
            self.loss_history[f'recon_u{i}'] = []
            self.loss_history[f'recon{i}'] = []
            self.loss_history[f'kl{i}'] = []
        
        # Extract gene dimensions and build datasets
        G_list = []
        self.datasets = []
        self.adj_full_list = []
        
        print(f"Processing {self.n_samples} samples...")
        for i, adata in enumerate(self.adata_list):
            count = adata.obsm['feat'].toarray() if hasattr(adata.obsm['feat'], 'toarray') else adata.obsm['feat']
            G_list.append(count.shape[1])
            self.datasets.append(self._build_dataset(adata))
            
            spatial = torch.tensor(adata.obsm['spatial'], dtype=torch.float32).to(device)
            adj_full = build_spatial_graph(spatial, k_neighbors=k_neighbors)
            self.adj_full_list.append(adj_full)
            print(f"  Sample {i+1}: {count.shape[0]} cells, {count.shape[1]} genes")
        
        # Shared encoder
        self.encoder = DisentangledGCNEncoder(
            dim_c=dim_c, dim_u=dim_u, input_dims_list=G_list, proj_dim=proj_dim,
            hidden_dims_common=hidden_dims_enc_common,
            hidden_dims_specific=hidden_dims_enc_specific,
            activation=activation,
            norm_type=enc_norm_type,
            use_norm=enc_use_norm,
            separate_mean_std=separate_mean_std,
        )
        
        # Shared decoder instance with sample-specific decoders inside
        self.decoder = DisentangledGCNDecoder(
            dim_c=dim_c, dim_u=dim_u, n_batches=self.n_batches,
            hidden_dims=hidden_dims_dec,
            output_dims_list=G_list,
            activation=activation,
            norm_type=dec_norm_type,
            use_norm=dec_use_norm,
            sigma2_y=1.0,
            fix_variance=False,
        )
        
        # GP Priors for each sample
        self.gp_c_list = nn.ModuleList([
            GP(output_dims=dim_c, kernel=RBFKernel(batch_shape=torch.Size([dim_c])),
               mean=ZeroMean(batch_shape=torch.Size([dim_c])))
            for _ in range(self.n_samples)
        ])
        
        self.gp_u_list = nn.ModuleList([
            GP(output_dims=dim_u, kernel=RBFKernel(batch_shape=torch.Size([dim_u])),
               mean=ZeroMean(batch_shape=torch.Size([dim_u])))
            for _ in range(self.n_samples)
        ])
        
        # Create GPVAE instances for each sample
        self.gpvae_list = nn.ModuleList()
        for i in range(self.n_samples):
            gpvae = DisentangledGPVAESpatial(
                encoder=self.encoder, decoder=self.decoder,
                gp_c=self.gp_c_list[i], gp_u=self.gp_u_list[i],
                k_neighbors=k_neighbors, sample_id=i, module_device=device,
                use_ld=use_ld, ld_n_round=ld_n_round
            )
            gpvae.train_dataset = self.datasets[i]
            self.gpvae_list.append(gpvae)

        # Batch classifier to ensure u retains batch-specific information
        self.batch_classifier = nn.Sequential(
            nn.Linear(dim_u, 32),
            nn.LayerNorm(32),
            nn.ELU(),
            nn.Linear(32, self.n_batches)
        )

        # MNN Discriminator for mutual information maximization
        self.mnn_discriminator = MNNDiscriminator(
            latent_dim=dim_c, 
            hidden_dim=max(64, dim_c)
        )

        # MNN cache: store pairs for all sample pairs (i, j) where i < j
        # Format: {(i, j): (mnn_idx_i, mnn_idx_j)}
        self.mnn_pairs = {}
        
        self.to(self.device)
        
        # Compute MNN pairs once at initialization (based on gene expression)
        self.update_mnn_pairs(k=self.mnn_k, n_top_genes=n_top_genes, 
                              cell_type_key=self.cell_type_key, 
                              verbose=self.print_mnn_accuracy)
        
        # Build unified graph for shared c encoding (STAligner style)
        # This will be rebuilt when MNN pairs are updated
        self._build_unified_graph()
    
    @staticmethod
    def _to_dense_array(X):
        """Convert sparse matrix to dense numpy array if needed."""
        if hasattr(X, 'toarray'):
            return X.toarray()
        return np.asarray(X, dtype=np.float32)
    
    def _apply_activation(self, x):
        """Apply activation function based on encoder settings."""
        if self.encoder.activation == 'elu':
            return F.elu(x)
        elif self.encoder.activation == 'relu':
            return F.relu(x)
        elif self.encoder.activation == 'gelu':
            return F.gelu(x)
        return x
    
    def _build_unified_graph(self):
        """
        Build unified graph (block diagonal + MNN pairs connections) for shared c encoding.
        Similar to STAligner's approach.
        """
        # Collect all sample sizes and compute start indices
        sample_sizes = []
        sample_start_indices = [0]
        for i in range(self.n_samples):
            adj_i = self.adj_full_list[i]
            sample_size = adj_i.size(0) if adj_i.is_sparse else adj_i.shape[0]
            sample_sizes.append(sample_size)
            sample_start_indices.append(sample_start_indices[-1] + sample_size)
        
        total_size = sum(sample_sizes)
        
        # Collect all non-zero elements from each adjacency matrix
        row_indices = []
        col_indices = []
        values = []
        
        # Fill block diagonal: each sample's spatial graph
        start_idx = 0
        for i, adj_i in enumerate(self.adj_full_list):
            end_idx = start_idx + sample_sizes[i]
            
            # Extract non-zero elements
            if adj_i.is_sparse:
                adj_i_coo = adj_i.coalesce()
                rows_i = adj_i_coo.indices()[0] + start_idx
                cols_i = adj_i_coo.indices()[1] + start_idx
                vals_i = adj_i_coo.values()
            else:
                rows_i, cols_i = torch.nonzero(adj_i, as_tuple=True)
                rows_i = rows_i + start_idx
                cols_i = cols_i + start_idx
                vals_i = adj_i[rows_i - start_idx, cols_i - start_idx]
            
            row_indices.append(rows_i.to(self.device))
            col_indices.append(cols_i.to(self.device))
            values.append(vals_i.to(self.device))
            
            start_idx = end_idx
        
        # Add MNN pairs connections (cross-sample edges)
        for i in range(self.n_samples):
            for j in range(i + 1, self.n_samples):
                pair_key = (i, j)
                if pair_key not in self.mnn_pairs:
                    continue
                
                mnn_idx_i, mnn_idx_j = self.mnn_pairs[pair_key]
                if len(mnn_idx_i) == 0:
                    continue
                
                # Convert to global indices in unified graph
                mnn_global_i = torch.tensor(mnn_idx_i, dtype=torch.long, device=self.device) + sample_start_indices[i]
                mnn_global_j = torch.tensor(mnn_idx_j, dtype=torch.long, device=self.device) + sample_start_indices[j]
                
                # Add bidirectional edges (symmetric)
                row_indices.append(mnn_global_i)
                col_indices.append(mnn_global_j)
                values.append(torch.ones(len(mnn_global_i), device=self.device))
                
                row_indices.append(mnn_global_j)
                col_indices.append(mnn_global_i)
                values.append(torch.ones(len(mnn_global_j), device=self.device))
        
        # Build sparse COO tensor
        if len(row_indices) > 0:
            rows = torch.cat(row_indices)
            cols = torch.cat(col_indices)
            vals = torch.cat(values)
            
            self.adj_unified = torch.sparse_coo_tensor(
                torch.stack([rows, cols]),
                vals,
                size=(total_size, total_size),
                device=self.device
            ).coalesce()
        else:
            self.adj_unified = torch.sparse_coo_tensor(
                torch.empty((2, 0), dtype=torch.long, device=self.device),
                torch.empty((0,), device=self.device),
                size=(total_size, total_size),
                device=self.device
            )
        
        self.sample_sizes = sample_sizes
        self.sample_start_indices = sample_start_indices
        
        return self.adj_unified, sample_sizes
    
    def _build_unified_graph_batch(self, idx_list):
        """
        Build unified graph for batch (block diagonal + MNN pairs connections).
        
        Args:
            idx_list: List of global indices for each sample's batch
        
        Returns:
            adj_unified_batch: Unified adjacency matrix for batch [B_total, B_total]
            batch_sizes: List of batch sizes for each sample
        """
        batch_sizes = [len(idx) for idx in idx_list]
        total_batch_size = sum(batch_sizes)
        
        # Collect non-zero elements
        row_indices = []
        col_indices = []
        values = []
        
        # Compute batch start indices in unified graph
        batch_start_indices = [0]
        for i in range(len(batch_sizes)):
            batch_start_indices.append(batch_start_indices[-1] + batch_sizes[i])
        
        # Fill block diagonal: extract subgraphs from each sample
        start_idx = 0
        for i, idx in enumerate(idx_list):
            adj_sub = extract_subgraph(idx, self.adj_full_list[i])
            
            # Extract non-zero elements
            if adj_sub.is_sparse:
                adj_sub_coo = adj_sub.coalesce()
                rows_sub = adj_sub_coo.indices()[0] + start_idx
                cols_sub = adj_sub_coo.indices()[1] + start_idx
                vals_sub = adj_sub_coo.values()
            else:
                rows_sub, cols_sub = torch.nonzero(adj_sub, as_tuple=True)
                rows_sub = rows_sub + start_idx
                cols_sub = cols_sub + start_idx
                vals_sub = adj_sub[rows_sub - start_idx, cols_sub - start_idx]
            
            row_indices.append(rows_sub)
            col_indices.append(cols_sub)
            values.append(vals_sub)
            
            start_idx += batch_sizes[i]
        
        # Add MNN pairs connections that exist in current batch
        for i in range(self.n_samples):
            for j in range(i + 1, self.n_samples):
                pair_key = (i, j)
                if pair_key not in self.mnn_pairs:
                    continue
                
                mnn_idx_i, mnn_idx_j = self.mnn_pairs[pair_key]
                if len(mnn_idx_i) == 0:
                    continue
                
                # Find which MNN pairs are in current batch
                idx_i_batch = idx_list[i].cpu().numpy()
                idx_j_batch = idx_list[j].cpu().numpy()
                
                # Convert global MNN indices to batch indices
                mnn_idx_i_tensor = torch.tensor(mnn_idx_i, dtype=torch.long)
                mnn_idx_j_tensor = torch.tensor(mnn_idx_j, dtype=torch.long)
                
                # Find matching pairs
                idx_i_set = set(idx_i_batch.tolist())
                idx_j_set = set(idx_j_batch.tolist())
                
                batch_mnn_i = []
                batch_mnn_j = []
                batch_pos_i = []
                batch_pos_j = []
                
                for k, (mnn_i, mnn_j) in enumerate(zip(mnn_idx_i, mnn_idx_j)):
                    if mnn_i in idx_i_set and mnn_j in idx_j_set:
                        batch_mnn_i.append(mnn_i)
                        batch_mnn_j.append(mnn_j)
                        # Find local positions in batch
                        batch_pos_i.append(np.where(idx_i_batch == mnn_i)[0][0])
                        batch_pos_j.append(np.where(idx_j_batch == mnn_j)[0][0])
                
                if len(batch_pos_i) > 0:
                    batch_pos_i = torch.tensor(batch_pos_i, dtype=torch.long, device=self.device)
                    batch_pos_j = torch.tensor(batch_pos_j, dtype=torch.long, device=self.device)
                    
                    # Convert to unified graph positions
                    unified_i = batch_pos_i + batch_start_indices[i]
                    unified_j = batch_pos_j + batch_start_indices[j]
                    
                    # Add bidirectional edges
                    row_indices.append(unified_i)
                    col_indices.append(unified_j)
                    values.append(torch.ones(len(unified_i), device=self.device))
                    
                    row_indices.append(unified_j)
                    col_indices.append(unified_i)
                    values.append(torch.ones(len(unified_j), device=self.device))
        
        # Build sparse COO tensor
        if len(row_indices) > 0:
            rows = torch.cat(row_indices)
            cols = torch.cat(col_indices)
            vals = torch.cat(values)
            
            adj_unified_batch = torch.sparse_coo_tensor(
                torch.stack([rows, cols]),
                vals,
                size=(total_batch_size, total_batch_size),
                device=self.device
            ).coalesce()
        else:
            adj_unified_batch = torch.sparse_coo_tensor(
                torch.empty((2, 0), dtype=torch.long, device=self.device),
                torch.empty((0,), device=self.device),
                size=(total_batch_size, total_batch_size),
                device=self.device
            )
        
        return adj_unified_batch, batch_sizes
    
    def _build_dataset(self, adata):
        """Build NNDataset from AnnData."""
        count = adata.obsm['feat'].toarray() if hasattr(adata.obsm['feat'], 'toarray') else adata.obsm['feat']
        spatial = adata.obsm['spatial']
        data_dict = {'count': count, 'spatial': spatial}
        dataset = NNDataset(data_dict=data_dict, data_device=self.device,
                            search_device='cpu', k_neighbors=self.k_neighbors)
        return dataset
    
    @torch.no_grad()
    def update_mnn_pairs(self, k=5, n_top_genes=200, use_c_features=False, c_list=None,
                         evaluate=True, cell_type_key='layer_guess', verbose=True,
                         max_pairs_per_sample_pair=None, max_total_pairs=None):
        """
        Update global MNN pairs for all sample pairs and optionally evaluate matching accuracy.
        
        Args:
            k: Number of nearest neighbors
            n_top_genes: Number of top HVG genes (if use_c_features=False)
            use_c_features: Whether to use c features instead of gene expression
            c_list: List of c features for all samples [required if use_c_features=True]
            evaluate: Whether to evaluate matching accuracy
            cell_type_key: Column name for cell type in adata.obs
            verbose: Whether to print evaluation results
        
        Returns:
            evaluation_dicts (dict): Dictionary of evaluation results for each sample pair
        """
        if verbose:
            print("  Updating MNN pairs...")
        
        self.mnn_pairs = {}
        evaluation_dicts = {}
        n_sample_pairs = self.n_samples * (self.n_samples - 1) // 2
        pair_cap_from_total = None
        if max_total_pairs is not None and n_sample_pairs > 0:
            pair_cap_from_total = max(1, max_total_pairs // n_sample_pairs)
        
        # Update MNN pairs for all sample pairs (i, j) where i < j
        for i in range(self.n_samples):
            for j in range(i + 1, self.n_samples):
                pair_key = (i, j)
                
                if use_c_features:
                    if c_list is None or len(c_list) != self.n_samples:
                        if verbose:
                            print(f"  Warning: c_list is required when use_c_features=True")
                        continue
                    mnn_idx_i, mnn_idx_j = find_mnn_pairs_from_c(
                        c_list[i], c_list[j], k=k, device=self.device
                    )
                    if verbose:
                        print(f"  Sample {i+1}-{j+1}: Found {len(mnn_idx_i)} MNN pairs from c features")
                else:
                    mnn_idx_i, mnn_idx_j = find_mnn_pairs_global(
                        self.adata_list[i], self.adata_list[j], k=k, n_top_genes=n_top_genes
                    )
                    if verbose:
                        print(f"  Sample {i+1}-{j+1}: Found {len(mnn_idx_i)} MNN pairs")
                
                self.mnn_pairs[pair_key] = (mnn_idx_i, mnn_idx_j)

                # Optional cap on total MNN pairs to avoid graph over-connection
                effective_cap = None
                if max_pairs_per_sample_pair is not None and pair_cap_from_total is not None:
                    effective_cap = min(max_pairs_per_sample_pair, pair_cap_from_total)
                elif max_pairs_per_sample_pair is not None:
                    effective_cap = max_pairs_per_sample_pair
                elif pair_cap_from_total is not None:
                    effective_cap = pair_cap_from_total
                
                if effective_cap is not None and len(mnn_idx_i) > effective_cap:
                    # Keep evenly spaced pairs to preserve coverage while limiting edge count
                    # (supports both Tensor and ndarray-like index containers).
                    selected_pos = np.linspace(
                        0, len(mnn_idx_i) - 1, num=effective_cap, dtype=int
                    )
                    if torch.is_tensor(mnn_idx_i):
                        keep_idx = torch.from_numpy(selected_pos).long().to(mnn_idx_i.device)
                        mnn_idx_i = mnn_idx_i[keep_idx]
                        mnn_idx_j = mnn_idx_j[keep_idx]
                    else:
                        mnn_idx_i = mnn_idx_i[selected_pos]
                        mnn_idx_j = mnn_idx_j[selected_pos]
                    self.mnn_pairs[pair_key] = (mnn_idx_i, mnn_idx_j)
                    if verbose:
                        print(
                            f"  Sample {i+1}-{j+1}: Capped MNN pairs to {len(mnn_idx_i)} "
                            f"(cap={effective_cap})"
                        )
                
                # Evaluate matching accuracy if requested
                if evaluate and len(mnn_idx_i) > 0:
                    if (cell_type_key in self.adata_list[i].obs.columns and 
                        cell_type_key in self.adata_list[j].obs.columns):
                        eval_dict = evaluate_mnn_matches(
                            mnn_idx_i, mnn_idx_j, 
                            self.adata_list[i], self.adata_list[j], 
                            cell_type_key=cell_type_key
                        )
                        evaluation_dicts[pair_key] = eval_dict
                        
                        if verbose:
                            print(f"    Matching Accuracy: {eval_dict['accuracy']:.4f} "
                                  f"({eval_dict['correct_matches']}/{eval_dict['total_matches']})")
                    elif verbose:
                        print(f"    Note: '{cell_type_key}' not found in adata.obs, skipping evaluation")
        
        # Rebuild unified graph with updated MNN pairs
        self._build_unified_graph()
        
        return evaluation_dicts
    
    
    def train_joint(self, epochs, batch_size, beta=1.0, 
                    lambda_recon=1.0, lambda_hsic=1.0,
                    lambda_batch_cls=1.0, lambda_mnn_mi=1.0,
                    n_negative_samples=5,
                    update_mnn_every=10,
                    use_c_features_for_mnn: bool = True,
                    c_mnn_start_epoch: int = 40,
                    max_mnn_pairs_per_sample_pair: int = None,
                    max_mnn_pairs_total: int = None,
                    early_stop_patience=10, early_stop_delta=1e-4):
        """
        Training with disentangled models.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            beta: KL divergence weight
            lambda_recon: Reconstruction loss weight
            lambda_hsic: HSIC loss weight
            lambda_batch_cls: Batch classification loss weight (if None, use self.lambda_batch_cls).
                This weight is used for: u batch classification, aleatoric regularization, 
                and c batch regularization (to prevent c from distinguishing batches).
            lambda_mnn_mi: MNN mutual information loss weight
            n_negative_samples: Number of negative samples for enhanced mutual information loss
            update_mnn_every: Update MNN pairs every N epochs
            use_c_features_for_mnn: Whether to ever use c features for MNN updates
            c_mnn_start_epoch: If not None, use this epoch as switch point instead of self.warmup_epochs.
                If None, uses self.warmup_epochs as the switch point.
            max_mnn_pairs_per_sample_pair: Optional cap for MNN pair count per sample pair.
                If set, each sample pair keeps at most this number of MNN pairs at each update.
            max_mnn_pairs_total: Optional global cap for total MNN pairs across all sample pairs.
                If set, it is converted to an equal per-pair cap internally.
            early_stop_patience: Early stopping patience
            early_stop_delta: Early stopping delta
            print_mnn_accuracy: Whether to print MNN matching accuracy during training
            
        Note:
            The model uses a warmup strategy for MNN computation:
            - Epochs 1 to self.warmup_epochs: Use gene expression for MNN
            - Epochs self.warmup_epochs+1 onwards: Use c features for MNN (if use_c_features_for_mnn=True)
            This allows the model to first learn basic spatial structure, then refine with learned representations.
            
            MNN loss uses discriminator-based mutual information maximization (inspired by scNiche).
        """
        lambda_batch_cls = self.lambda_batch_cls if lambda_batch_cls is None else lambda_batch_cls
        
        # Create loaders for all samples
        loaders = [
            DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            for dataset in self.datasets
        ]
        
        self.optimizer = AdamW(self.parameters(), lr=1e-3, weight_decay=1e-5)
        
        # Determine the switch epoch: use c_mnn_start_epoch if provided, otherwise use self.warmup_epochs
        switch_epoch = c_mnn_start_epoch if c_mnn_start_epoch is not None else self.warmup_epochs

        best_loss = float('inf')
        epochs_no_improve = 0
        best_state_dict = None
        
        pbar_epoch = tqdm(range(epochs), desc="Training", position=0)
        
        for epoch in pbar_epoch:
            self.train()
            
            if (epoch + 1) % update_mnn_every == 0:
                # Warmup strategy: use gene expression for MNN in warmup, then switch to c features
                # Epochs 1 to switch_epoch: use gene expression
                # Epochs (switch_epoch+1) onwards: use c features (if use_c_features_for_mnn=True)
                use_c_now = False
                if use_c_features_for_mnn:
                    use_c_now = (epoch + 1) > switch_epoch

                mode_str = "c features" if use_c_now else "gene expression"
                is_warmup = (epoch + 1) <= switch_epoch
                warmup_info = f" (warmup epoch {epoch + 1}/{switch_epoch})" if is_warmup else ""
                print(f"\n[Epoch {epoch + 1}] Updating MNN pairs ({mode_str}){warmup_info} ...")
                with torch.no_grad():
                    if use_c_now:
                        # Extract current c features for all samples using unified graph encoding
                        # This ensures consistency with training-time c encoding
                        adj_unified, sample_sizes = self._build_unified_graph()
                        
                        c_list = []
                        h_projected_list = []
                        
                        # Project all samples
                        for i in range(self.n_samples):
                            y_full = self.datasets[i].Y  # [N_i, G_i]
                            h_i = self.encoder.projection_layers[i](y_full)
                            h_i = self._apply_activation(h_i)
                            h_projected_list.append(h_i)
                        
                        # Concatenate and encode c using unified graph (same as predict/train)
                        h_concat = torch.cat(h_projected_list, dim=0)  # [N_total, proj_dim]
                        
                        # Use unified graph to encode c (shared representation)
                        if self.encoder.separate_mean_std:
                            h_c_base = self.encoder._forward_gcn(h_concat, adj_unified, self.encoder.common_gcn_base)
                            c_mean_concat = self.encoder.common_mean_head(h_c_base, adj_unified)
                            c_std_logit_concat = self.encoder.common_std_head(h_c_base, adj_unified)
                            c_std_concat = F.softplus(c_std_logit_concat)
                            c_std_concat = torch.clamp(c_std_concat, min=self.MIN_STD_CLAMP)
                        else:
                            c_params_concat = self.encoder._forward_gcn(h_concat, adj_unified, self.encoder.common_gcn_layers)
                            c_mean_concat = c_params_concat[..., :self.dim_c]
                            c_std_concat = F.softplus(c_params_concat[..., self.dim_c:])
                            c_std_concat = torch.clamp(c_std_concat, min=self.MIN_STD_CLAMP)
                        
                        # Split c back to samples
                        start_idx = 0
                        for i, size in enumerate(sample_sizes):
                            end_idx = start_idx + size
                            c_i_full = c_mean_concat[start_idx:end_idx].cpu()
                            c_list.append(c_i_full)
                            start_idx = end_idx
                        
                        evaluation = self.update_mnn_pairs(
                            k=self.mnn_k,
                            use_c_features=True,
                            c_list=c_list,
                            evaluate=True,
                            cell_type_key=self.cell_type_key,
                            verbose=self.print_mnn_accuracy,
                            max_pairs_per_sample_pair=max_mnn_pairs_per_sample_pair,
                            max_total_pairs=max_mnn_pairs_total,
                        )
                    else:
                        # Use gene expression HVG-based MNN
                        evaluation = self.update_mnn_pairs(
                            k=self.mnn_k,
                            n_top_genes=self.n_top_genes,
                            use_c_features=False,
                            evaluate=True,
                            cell_type_key=self.cell_type_key,
                            verbose=self.print_mnn_accuracy,
                            max_pairs_per_sample_pair=max_mnn_pairs_per_sample_pair,
                            max_total_pairs=max_mnn_pairs_total,
                        )
            
            # Initialize epoch losses
            epoch_losses = {'total': 0, 'hsic': 0, 'batch_cls': 0,
                            'aleatoric_reg': 0, 'mnn_mi': 0}
            for i in range(self.n_samples):
                epoch_losses[f'recon_c{i}'] = 0
                epoch_losses[f'recon_u{i}'] = 0
                epoch_losses[f'recon{i}'] = 0
                epoch_losses[f'kl{i}'] = 0
            
            n_batches = 0
            
            # Use zip to iterate over all loaders simultaneously
            min_batches = min(len(loader) for loader in loaders)
            pbar_batch = tqdm(zip(*loaders), desc=f"Epoch {epoch+1}/{epochs}", 
                            position=1, leave=False, total=min_batches)
            
            for batch_data_list in pbar_batch:
                # Process all samples
                batch_outputs = []
                c_list = []
                u_list = []  # Store original u for HSIC loss
                u_cleaned_list = []
                aleatoric_list = []
                recon_losses = []
                kl_list = []
                
                # 1. Collect all batch data
                y_list = []
                x_list = []
                idx_list = []
                adj_sub_list = []
                batch_id_list = []
                
                for i, batch_data in enumerate(batch_data_list):
                    y, x, idx = batch_data
                    y, x = y.to(self.device), x.to(self.device)
                    idx = idx.to(self.device)
                    
                    adj_sub = extract_subgraph(idx, self.adj_full_list[i])
                    batch_id = torch.full((y.size(0),), i, dtype=torch.long, device=self.device)
                    
                    y_list.append(y)
                    x_list.append(x)
                    idx_list.append(idx)
                    adj_sub_list.append(adj_sub)
                    batch_id_list.append(batch_id)
                
                # 2. Unified encoding for c: concatenate all samples and use unified graph (STAligner style)
                adj_unified_batch, batch_sizes = self._build_unified_graph_batch(idx_list)
                
                # Project each sample separately (projection layers are sample-specific)
                h_projected_list = []
                for i, y in enumerate(y_list):
                    h_i = self.encoder.projection_layers[i](y)
                    h_i = self._apply_activation(h_i)
                    h_projected_list.append(h_i)
                
                # Concatenate projected features
                h_concat = torch.cat(h_projected_list, dim=0)  # [B_total, proj_dim]
                
                # Use unified graph to encode c (shared representation)
                if self.encoder.separate_mean_std:
                    h_c_base = self.encoder._forward_gcn(h_concat, adj_unified_batch, self.encoder.common_gcn_base)
                    c_mean_concat = self.encoder.common_mean_head(h_c_base, adj_unified_batch)
                    c_std_logit_concat = self.encoder.common_std_head(h_c_base, adj_unified_batch)
                    c_std_concat = F.softplus(c_std_logit_concat)
                    c_std_concat = torch.clamp(c_std_concat, min=self.MIN_STD_CLAMP)
                else:
                    c_params_concat = self.encoder._forward_gcn(h_concat, adj_unified_batch, self.encoder.common_gcn_layers)
                    c_mean_concat = c_params_concat[..., :self.dim_c]
                    c_std_concat = F.softplus(c_params_concat[..., self.dim_c:])
                    c_std_concat = torch.clamp(c_std_concat, min=self.MIN_STD_CLAMP)
                
                # Sample c
                c_concat = Normal(c_mean_concat, c_std_concat).rsample()
                
                # Split c back to each sample's batch
                c_mean_list = []
                c_std_list = []
                c_list_batch = []
                start_idx = 0
                for i, batch_size in enumerate(batch_sizes):
                    end_idx = start_idx + batch_size
                    c_mean_list.append(c_mean_concat[start_idx:end_idx])
                    c_std_list.append(c_std_concat[start_idx:end_idx])
                    c_list_batch.append(c_concat[start_idx:end_idx])
                    start_idx = end_idx
                
                # 3. Separate encoding for u (sample-specific, using each sample's adj_sub)
                for i in range(self.n_samples):
                    y = y_list[i]
                    x = x_list[i]
                    idx = idx_list[i]
                    adj_sub = adj_sub_list[i]
                    batch_id = batch_id_list[i]
                    c_mean = c_mean_list[i]
                    c_std = c_std_list[i]
                    c = c_list_batch[i]
                    
                    # Encode u (sample-specific)
                    _, _, u_mean, u_std = self.encoder(y, adj_sub, sample_id=i, adj_c=adj_sub)
                    
                    # Sample u
                    u = Normal(u_mean, u_std).rsample()
                    
                    # Apply LD module
                    if self.gpvae_list[i].use_ld and self.gpvae_list[i].ld_module is not None:
                        u_cleaned, aleatoric = self.gpvae_list[i].ld_module(u)
                    else:
                        u_cleaned = u
                        aleatoric = torch.zeros_like(u)
                    
                    # Decode: two reconstruction modes
                    x_recon_c, x_recon_u, _ = self.gpvae_list[i].decoder(
                        c, u_cleaned, adj_sub, batch_id, sample_id=i
                    )
                    
                    # Two reconstruction losses
                    lik_c, _ = self.gpvae_list[i].loss_sws(y, x, x_recon_c, batch_id, beta=0)
                    lik_u, _ = self.gpvae_list[i].loss_sws(y, x, x_recon_u, batch_id, beta=0)
                    _, kl = self.gpvae_list[i].loss_sws(y, x, x_recon_c, batch_id, beta=beta)
                    
                    # Combined reconstruction loss
                    recon_loss = -0.5*lik_c - 0.5*lik_u  # Only use c and u reconstruction losses
                    
                    batch_outputs.append({
                        'y': y, 'x': x, 'idx': idx, 'batch_id': batch_id,
                        'lik_c': lik_c, 'lik_u': lik_u,
                        'recon_loss': recon_loss, 'kl': kl
                    })
                    c_list.append(c)
                    u_list.append(u)  # Store original u for HSIC
                    u_cleaned_list.append(u_cleaned)
                    aleatoric_list.append(aleatoric)
                    recon_losses.append(recon_loss)
                    kl_list.append(kl)
                
                # HSIC loss: sum over all samples
                # Use original u (not u_cleaned) to ensure independence of raw latent variables
                # This is consistent with KL divergence which also uses original u distribution
                hsic_loss = sum(HSIC(c_list[i], u_list[i]) for i in range(self.n_samples))
                
                # MNN loss: mutual information maximization over all sample pairs
                total_mnn_mi = torch.tensor(0.0, device=self.device)
                n_pairs = self.n_samples * (self.n_samples - 1) // 2
                
                for i in range(self.n_samples):
                    for j in range(i + 1, self.n_samples):
                        pair_key = (i, j)
                        if pair_key not in self.mnn_pairs:
                            continue
                        
                        mnn_idx_i, mnn_idx_j = self.mnn_pairs[pair_key]
                        batch_mnn_idx_i, batch_mnn_idx_j = compute_batch_mnn_pairs(
                            batch_outputs[i]['idx'], batch_outputs[j]['idx'], 
                            mnn_idx_i, mnn_idx_j
                        )
                
                        # Mutual information loss
                        if n_negative_samples > 1:
                            total_mnn_mi += enhanced_mnn_mutual_information_loss(
                                c_list[i], c_list[j], batch_mnn_idx_i, batch_mnn_idx_j,
                                self.mnn_discriminator, n_negative_samples=n_negative_samples
                            )
                        else:
                            total_mnn_mi += mnn_mutual_information_loss(
                                c_list[i], c_list[j], batch_mnn_idx_i, batch_mnn_idx_j,
                                self.mnn_discriminator
                            )
                
                # Average MNN loss over number of pairs
                if n_pairs > 0:
                    total_mnn_mi = total_mnn_mi / n_pairs
                
                # Total MNN loss
                total_mnn_loss = lambda_mnn_mi * total_mnn_mi

                # Batch classification loss on u_cleaned to encourage sample-specific info
                batch_cls_loss = torch.tensor(0.0, device=self.device)
                for i, u_cleaned in enumerate(u_cleaned_list):
                    batch_logits = self.batch_classifier(u_cleaned)
                    batch_labels = torch.full((u_cleaned.size(0),), i, dtype=torch.long, device=self.device)
                    batch_cls_loss += F.cross_entropy(batch_logits, batch_labels)
                batch_cls_loss = batch_cls_loss / self.n_samples
                
                # Noise (aleatoric) should NOT be able to distinguish batches
                # Random guessing for n-class classification has cross-entropy = log(n)
                log_n_samples = torch.tensor(np.log(self.n_samples), device=self.device)
                aleatoric_batch_cls_loss = torch.tensor(0.0, device=self.device)
                for i, aleatoric in enumerate(aleatoric_list):
                    aleatoric_logits = self.batch_classifier(aleatoric)
                    batch_labels = torch.full((aleatoric.size(0),), i, dtype=torch.long, device=self.device)
                    aleatoric_batch_cls_loss += F.cross_entropy(aleatoric_logits, batch_labels)
                aleatoric_batch_cls_loss = aleatoric_batch_cls_loss / self.n_samples
                # Penalize if aleatoric can distinguish batches (loss < log(n_samples))
                aleatoric_regularization = F.relu(log_n_samples - aleatoric_batch_cls_loss) ** 2
                
                # Total loss: sum over all samples
                total_recon_loss = sum(recon_losses)
                total_kl = sum(kl_list)
                
                # Total loss: reconstruction losses + KL (already weighted by beta in loss_sws) + other losses
                total_loss = lambda_recon * total_recon_loss + \
                            total_kl + \
                            lambda_hsic * hsic_loss + \
                            total_mnn_loss + \
                            lambda_batch_cls * batch_cls_loss + \
                            lambda_batch_cls * aleatoric_regularization

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
                self.optimizer.step()
                
                # Record losses
                epoch_losses['total'] += total_loss.item()
                for i, output in enumerate(batch_outputs):
                    epoch_losses[f'recon_c{i}'] += -output['lik_c'].item()
                    epoch_losses[f'recon_u{i}'] += -output['lik_u'].item()
                    epoch_losses[f'recon{i}'] += output['recon_loss'].item()
                    epoch_losses[f'kl{i}'] += output['kl'].item()
                epoch_losses['mnn_mi'] += total_mnn_mi.item()
                epoch_losses['hsic'] += hsic_loss.item()
                epoch_losses['batch_cls'] += batch_cls_loss.item()
                epoch_losses['aleatoric_reg'] += aleatoric_regularization.item()
                
                n_batches += 1
            
            # Average losses
            for key in epoch_losses:
                epoch_losses[key] /= n_batches
            
            # Record losses to history
            self.loss_history['epoch'].append(epoch + 1)
            self.loss_history['total'].append(epoch_losses['total'])
            
            # Record per-sample losses
            for i in range(self.n_samples):
                self.loss_history[f'recon_c{i}'].append(epoch_losses[f'recon_c{i}'])
                self.loss_history[f'recon_u{i}'].append(epoch_losses[f'recon_u{i}'])
                self.loss_history[f'recon{i}'].append(epoch_losses[f'recon{i}'])
                self.loss_history[f'kl{i}'].append(epoch_losses[f'kl{i}'])
            
            # Average losses
            recon_avg = sum(epoch_losses[f'recon{i}'] for i in range(self.n_samples)) / self.n_samples
            recon_c_avg = sum(epoch_losses[f'recon_c{i}'] for i in range(self.n_samples)) / self.n_samples
            recon_u_avg = sum(epoch_losses[f'recon_u{i}'] for i in range(self.n_samples)) / self.n_samples
            kl_avg = sum(epoch_losses[f'kl{i}'] for i in range(self.n_samples)) / self.n_samples
            
            self.loss_history['recon_avg'].append(recon_avg)
            self.loss_history['kl_avg'].append(kl_avg)
            self.loss_history['mnn_mi'].append(epoch_losses['mnn_mi'])
            self.loss_history['hsic'].append(epoch_losses['hsic'])
            self.loss_history['batch_cls'].append(epoch_losses['batch_cls'])
            if 'aleatoric_reg' not in self.loss_history:
                self.loss_history['aleatoric_reg'] = []
            self.loss_history['aleatoric_reg'].append(epoch_losses['aleatoric_reg'])
            
            pbar_epoch.set_postfix({
                'Total': f"{epoch_losses['total']:.4f}",
                'Recon': f"{recon_avg:.4f}",
                'ReconC': f"{recon_c_avg:.4f}",
                'ReconU': f"{recon_u_avg:.4f}",
                'KL': f"{kl_avg:.4f}",
                'MNNMI': f"{epoch_losses['mnn_mi']:.4f}",
                'HSIC': f"{epoch_losses['hsic']:.4f}",
                'BatchCls': f"{epoch_losses['batch_cls']:.4f}",
                'AleatReg': f"{epoch_losses['aleatoric_reg']:.4f}",
            })
            
            # Early stopping
            if epoch_losses['total'] + early_stop_delta < best_loss:
                best_loss = epoch_losses['total']
                best_state_dict = self.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= early_stop_patience:
                pbar_epoch.close()
                print(f"\n✓ Early stopping at epoch {epoch + 1}. Best loss: {best_loss:.4f}")
                if best_state_dict is not None:
                    self.load_state_dict(best_state_dict)
                break
        
        pbar_epoch.close()
        if epochs_no_improve < early_stop_patience:
            print(f"\n✓ Training completed. Final loss: {epoch_losses['total']:.4f}")
    
    @torch.no_grad()
    def predict(self, sample=True, return_aleatoric=False):
        """
        Extract latent representations using unified graph encoding for c (STAligner style).
        
        Args:
            sample (bool): Whether to sample from distributions or use mean
            return_aleatoric (bool): Whether to return aleatoric uncertainty (noise) extracted by LD module
        
        Returns:
            c_list: List of c embeddings for each sample [numpy arrays]
            u_list: List of u_cleaned embeddings for each sample [numpy arrays]
            aleatoric_list: List of aleatoric uncertainty (noise) for each sample [numpy arrays] (if return_aleatoric=True)
        """
        self.eval()
        
        # Build unified graph for full data
        adj_unified, sample_sizes = self._build_unified_graph()
        
        # 1. Unified encoding for c: process all samples together using unified graph
        # Project each sample separately (projection layers are sample-specific)
        h_projected_list = []
        for i in range(self.n_samples):
            y = self.datasets[i].Y  # [N_i, G_i]
            h_i = self.encoder.projection_layers[i](y)
            h_i = self._apply_activation(h_i)
            h_projected_list.append(h_i)
        
        # Concatenate projected features
        h_concat = torch.cat(h_projected_list, dim=0)  # [N_total, proj_dim]
        
        # Use unified graph to encode c (shared representation)
        if self.encoder.separate_mean_std:
            h_c_base = self.encoder._forward_gcn(h_concat, adj_unified, self.encoder.common_gcn_base)
            c_mean_concat = self.encoder.common_mean_head(h_c_base, adj_unified)
            c_std_logit_concat = self.encoder.common_std_head(h_c_base, adj_unified)
            c_std_concat = F.softplus(c_std_logit_concat)
            c_std_concat = torch.clamp(c_std_concat, min=self.MIN_STD_CLAMP)
        else:
            c_params_concat = self.encoder._forward_gcn(h_concat, adj_unified, self.encoder.common_gcn_layers)
            c_mean_concat = c_params_concat[..., :self.dim_c]
            c_std_concat = F.softplus(c_params_concat[..., self.dim_c:])
            c_std_concat = torch.clamp(c_std_concat, min=self.MIN_STD_CLAMP)
        
        # Sample or use mean
        if sample:
            c_concat = Normal(c_mean_concat, c_std_concat).sample()
        else:
            c_concat = c_mean_concat
        
        # Split c results
        c_list = []
        start_idx = 0
        for i, size in enumerate(sample_sizes):
            end_idx = start_idx + size
            c_list.append(c_concat[start_idx:end_idx].cpu().numpy())
            start_idx = end_idx
        
        # 2. Separate encoding for u (each sample independently)
        u_list = []
        aleatoric_list = []
        
        for i in range(self.n_samples):
            y = self.datasets[i].Y  # [N_i, G_i]
            adj_full = self.adj_full_list[i]  # [N_i, N_i]
            batch_id = torch.full((y.size(0),), i, dtype=torch.long, device=self.device)
            
            # Get u_cleaned and aleatoric (LD applied in sample_latent)
            _, u_cleaned, aleatoric, _, _, u_mean, u_std = self.gpvae_list[i].sample_latent(y, adj_full, batch_id, sample=sample)
            
            u_list.append(u_cleaned.cpu().numpy())
            if return_aleatoric:
                aleatoric_list.append(aleatoric.cpu().numpy())
        
        self.train()
        if return_aleatoric:
            return c_list, u_list, aleatoric_list
        else:
            return c_list, u_list
    
    def get_loss_history(self):
        """
        Get training loss history for plotting.
        
        Returns:
            dict: Dictionary containing loss history with keys:
                - 'epoch': List of epoch numbers
                - 'total': Total loss per epoch
                - 'recon1', 'recon2', 'recon_avg': Reconstruction losses
                - 'kl1', 'kl2', 'kl_avg': KL divergence losses
                - 'mnn_mi': MNN mutual information loss
                - 'hsic': HSIC loss
                - 'batch_cls': Batch classification loss
                - 'aleatoric_reg': Aleatoric regularization loss
        """
        return self.loss_history.copy()

