"""
Mutual Nearest Neighbors (MNN) related functions.
"""
import torch
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors


def find_mnn_pairs_global(adata1, adata2, k: int = 5, n_top_genes: int = 200):
    """
    Find mutual nearest neighbors between two samples using top HVG genes.
    
    Args:
        adata1, adata2: Two AnnData objects
        k (int): Number of nearest neighbors to consider for MNN
        n_top_genes (int): Number of top HVG genes to use
    
    Returns:
        mnn_idx1 (Tensor): Indices in sample1 that have MNN [M]
        mnn_idx2 (Tensor): Corresponding indices in sample2 [M]
    """
    print(f"Finding MNN pairs using top {n_top_genes} HVG genes...")
    
    # Extract expression matrices
    X1 = adata1.obsm['feat']
    X2 = adata2.obsm['feat']
    
    # Convert sparse to dense if needed
    if hasattr(X1, 'toarray'):
        X1 = X1.toarray()
    if hasattr(X2, 'toarray'):
        X2 = X2.toarray()
    
    X1 = np.asarray(X1, dtype=np.float32)
    X2 = np.asarray(X2, dtype=np.float32)
    
    # Simple gene selection: use all genes or select by variance
    if X1.shape[1] == X2.shape[1]:
        # Same number of genes, assume same gene order
        if X1.shape[1] > n_top_genes:
            # Select top variable genes from combined data
            X_combined = np.vstack([X1, X2])
            gene_vars = np.var(X_combined, axis=0)
            top_indices = np.argsort(gene_vars)[-n_top_genes:]
            X1 = X1[:, top_indices]
            X2 = X2[:, top_indices]
            print(f"  Selected {len(top_indices)} top variable genes")
        else:
            print(f"  Using all {X1.shape[1]} genes")
    else:
        print(f"  Warning: Different gene numbers ({X1.shape[1]} vs {X2.shape[1]})")
        print(f"  Using first {min(X1.shape[1], X2.shape[1])} genes")
        min_genes = min(X1.shape[1], X2.shape[1])
        X1 = X1[:, :min_genes]
        X2 = X2[:, :min_genes]
    
    # Compute MNN using simple implementation
    idx1_np, idx2_np = _mutual_knn_simple(X1, X2, k=k)
    
    if idx1_np.size == 0:
        print("  Warning: No MNN pairs found")
        return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)
    
    mnn_idx1 = torch.from_numpy(idx1_np.astype(np.int64))
    mnn_idx2 = torch.from_numpy(idx2_np.astype(np.int64))
    
    print(f"✓ Found {len(mnn_idx1)} MNN pairs")
    return mnn_idx1, mnn_idx2


def find_mnn_pairs_from_c(c1, c2, k: int = 5, device="cpu"):
    """
    Find mutual nearest neighbors (MNN) pairs based on synergistic features c.
    
    Args:
        c1 (Tensor or np.ndarray): Synergistic features of sample 1 [n1, dim_c]
        c2 (Tensor or np.ndarray): Synergistic features of sample 2 [n2, dim_c]
        k (int): Number of nearest neighbors to consider
        device: Computing device (kept for API compatibility)
    
    Returns:
        mnn_idx1 (Tensor): Indices in sample1 that have MNN [M]
        mnn_idx2 (Tensor): Corresponding indices in sample2 [M]
    """
    # Ensure numpy arrays
    if isinstance(c1, torch.Tensor):
        c1_np = c1.detach().cpu().numpy()
    else:
        c1_np = np.asarray(c1)

    if isinstance(c2, torch.Tensor):
        c2_np = c2.detach().cpu().numpy()
    else:
        c2_np = np.asarray(c2)

    idx1_np, idx2_np = _mutual_knn_simple(c1_np, c2_np, k=k)

    if idx1_np.size == 0:
        print("  Warning: No MNN pairs found from c features")
        return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)

    mnn_idx1 = torch.from_numpy(idx1_np.astype(np.int64))
    mnn_idx2 = torch.from_numpy(idx2_np.astype(np.int64))

    print(f"✓ Found {len(mnn_idx1)} MNN pairs from c features")
    return mnn_idx1, mnn_idx2


def _mutual_knn_simple(X1: np.ndarray, X2: np.ndarray, k: int = 5):
    """
    Efficient mutual KNN implementation using sklearn.
    Similar to STAligner's MNN: match1 ∩ reversed(match2)
    
    Args:
        X1: Feature matrix of dataset 1, shape [n1, d]
        X2: Feature matrix of dataset 2, shape [n2, d]
        k: Number of nearest neighbors
    
    Returns:
        idx1: Indices in dataset 1 that have at least one mutual NN [M]
        idx2: Corresponding indices in dataset 2 [M]
    """
    n1, n2 = X1.shape[0], X2.shape[0]
    
    if n1 == 0 or n2 == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    
    k = min(k, n1, n2)
    
    # Use sklearn's NearestNeighbors for efficient KNN computation (parallelized)
    # 1) KNN: X1 -> X2
    nn_1_to_2 = NearestNeighbors(n_neighbors=k, metric='euclidean', algorithm='auto', n_jobs=-1)
    nn_1_to_2.fit(X2)
    knn_1_to_2 = nn_1_to_2.kneighbors(X1, return_distance=False)  # [n1, k]
    
    # 2) KNN: X2 -> X1
    nn_2_to_1 = NearestNeighbors(n_neighbors=k, metric='euclidean', algorithm='auto', n_jobs=-1)
    nn_2_to_1.fit(X1)
    knn_2_to_1 = nn_2_to_1.kneighbors(X2, return_distance=False)  # [n2, k]
    
    # 3) Mutual edges (like STAligner.mnn: match1 ∩ reversed(match2))
    # For each i in X1, check if its neighbors j in X2 also have i as neighbor
    mnn_pairs = []
    for i in range(n1):
        neighbors_in_2 = knn_1_to_2[i]  # [k] - neighbors of i in X2
        for j in neighbors_in_2:
            neighbors_in_1 = knn_2_to_1[j]  # [k] - neighbors of j in X1
            if i in neighbors_in_1:
                mnn_pairs.append((i, int(j)))
    
    if len(mnn_pairs) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    
    # Convert to array (no need to remove duplicates as each pair should be unique)
    pairs_arr = np.array(mnn_pairs, dtype=np.int64)
    idx1 = pairs_arr[:, 0]
    idx2 = pairs_arr[:, 1]
    return idx1, idx2


def _simple_knn(X_query: np.ndarray, X_data: np.ndarray, k: int) -> np.ndarray:
    """
    Efficient KNN implementation using sklearn (kept for backward compatibility).
    
    Args:
        X_query: Query points [n_query, d]
        X_data: Data points to search in [n_data, d]
        k: Number of nearest neighbors
    
    Returns:
        indices: KNN indices [n_query, k]
    """
    k = min(k, X_data.shape[0])
    if k == 0:
        return np.array([], dtype=np.int64).reshape(0, 0)
    
    # Use sklearn's NearestNeighbors for efficient computation
    nn = NearestNeighbors(n_neighbors=k, metric='euclidean', algorithm='auto', n_jobs=-1)
    nn.fit(X_data)
    knn_indices, _ = nn.kneighbors(X_query)  # [n_query, k]
    
    return knn_indices


def evaluate_mnn_matches(mnn_idx1, mnn_idx2, adata1, adata2, cell_type_key='cell_type'):
    """
    Evaluate MNN matches: check how many matched spots have the same cell type.
    
    Args:
        mnn_idx1 (Tensor): Indices in sample1 that have MNN [M]
        mnn_idx2 (Tensor): Corresponding indices in sample2 [M]
        adata1, adata2: Two AnnData objects
        cell_type_key (str): Column name for cell type
    
    Returns:
        evaluation_dict: Dictionary with evaluation results
    """
    print("Evaluating MNN matches...")
    
    if cell_type_key not in adata1.obs.columns:
        raise ValueError(f"'{cell_type_key}' column not found in adata1.obs")
    if cell_type_key not in adata2.obs.columns:
        raise ValueError(f"'{cell_type_key}' column not found in adata2.obs")
    
    # Convert tensor indices to numpy for indexing
    mnn_idx1_np = mnn_idx1.cpu().numpy() if isinstance(mnn_idx1, torch.Tensor) else mnn_idx1
    mnn_idx2_np = mnn_idx2.cpu().numpy() if isinstance(mnn_idx2, torch.Tensor) else mnn_idx2
    
    cell_types1 = adata1.obs[cell_type_key].values
    cell_types2 = adata2.obs[cell_type_key].values
    
    # Count matches with same cell type
    same_type_count = 0
    total_count = len(mnn_idx1_np)
    
    # Statistics by cell type
    type_match_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    
    for idx1, idx2 in zip(mnn_idx1_np, mnn_idx2_np):
        ct1 = cell_types1[idx1]
        ct2 = cell_types2[idx2]
        
        type_match_stats[ct1]['total'] += 1
        if ct1 == ct2:
            same_type_count += 1
            type_match_stats[ct1]['correct'] += 1
    
    accuracy = same_type_count / total_count if total_count > 0 else 0.0
    
    # Calculate accuracy per cell type
    type_accuracies = {}
    for ct, stats in type_match_stats.items():
        if stats['total'] > 0:
            type_accuracies[ct] = stats['correct'] / stats['total']
    
    evaluation_dict = {
        'total_matches': total_count,
        'correct_matches': same_type_count,
        'accuracy': accuracy,
        'type_accuracies': type_accuracies,
        'type_match_stats': dict(type_match_stats)
    }
    
    print(f"✓ Evaluation complete! Overall accuracy: {accuracy:.4f}")
    return evaluation_dict