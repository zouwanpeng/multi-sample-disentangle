"""
Loss functions for disentangled training.
"""
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn

def pairwise_distances(x):
    """
    Compute pairwise squared Euclidean distances.
    x should be two-dimensional.
    """
    instances_norm = torch.sum(x ** 2, -1).reshape((-1, 1))
    return -2 * torch.mm(x, x.t()) + instances_norm + instances_norm.t()


def GaussianKernelMatrix(x, sigma=1):
    """
    Compute Gaussian kernel matrix.
    
    Uses standard RBF kernel: K(x, y) = exp(-||x-y||^2 / (2*sigma^2))
    Note: sigma here is the bandwidth parameter (standard deviation).
    """
    pairwise_distances_ = pairwise_distances(x)  # Returns squared distances
    # Standard RBF kernel: exp(-d^2 / (2*sigma^2))
    return torch.exp(-pairwise_distances_ / (2.0 * sigma ** 2))


def HSIC(x, y, s_x=None, s_y=None):
    """
    Hilbert-Schmidt Independence Criterion (HSIC)
    
    Measures independence between two variables. Lower HSIC means more independent.
    Used for disentanglement: minimize HSIC(c, u) to make c and u independent.
    
    Args:
        x (Tensor): First variable [m, dim_x]
        y (Tensor): Second variable [m, dim_y]
        s_x (float, optional): Kernel bandwidth for x. If None, uses median heuristic.
        s_y (float, optional): Kernel bandwidth for y. If None, uses median heuristic.
    
    Returns:
        HSIC value (Tensor): Scalar (lower is better for independence)
    """
    m, _ = x.shape  # batch size
    
    # Adaptive bandwidth selection using median heuristic (if not provided)
    if s_x is None:
        # Use median of pairwise distances as bandwidth
        with torch.no_grad():
            dists = torch.sqrt(torch.clamp(pairwise_distances(x), min=1e-8))
            median_dist = torch.median(dists[dists > 0])
            s_x = median_dist.item() if median_dist > 0 else 1.0
    
    if s_y is None:
        with torch.no_grad():
            dists = torch.sqrt(torch.clamp(pairwise_distances(y), min=1e-8))
            median_dist = torch.median(dists[dists > 0])
            s_y = median_dist.item() if median_dist > 0 else 1.0
    
    # Compute kernel matrices
    K = GaussianKernelMatrix(x, s_x)
    L = GaussianKernelMatrix(y, s_y)
    
    # Centering matrix: H = I - 1/m * 11^T
    H = torch.eye(m, device=x.device, dtype=x.dtype) - 1.0 / m * torch.ones((m, m), device=x.device, dtype=x.dtype)
    
    # HSIC = 1/(m-1)^2 * tr(KHLH) = 1/(m-1)^2 * tr(LHKH)
    # Using tr(LHKH) for computational efficiency
    HKH = torch.mm(H, torch.mm(K, H))
    HSIC_value = torch.trace(torch.mm(L, HKH)) / ((m - 1) ** 2)
    
    # Add small epsilon for numerical stability
    return HSIC_value + 1e-8


def compute_mmd(z1: Tensor, z2: Tensor, sigma: float = 1.0) -> Tensor:
    """
    Compute Maximum Mean Discrepancy (MMD) between two distributions
    
    Args:
        z1 (Tensor): First distribution [N1, dim]
        z2 (Tensor): Second distribution [N2, dim]
        sigma (float): Gaussian kernel bandwidth
    
    Returns:
        mmd_squared (Tensor): MMD^2 value
    """
    def gaussian_kernel(x, y, sigma):
        dist = torch.cdist(x, y, p=2)
        return torch.exp(-dist ** 2 / (2 * sigma ** 2))
    
    K_11 = gaussian_kernel(z1, z1, sigma)
    K_22 = gaussian_kernel(z2, z2, sigma)
    K_12 = gaussian_kernel(z1, z2, sigma)
    
    mmd_squared = K_11.mean() + K_22.mean() - 2 * K_12.mean()
    return mmd_squared


def mnn_contrastive_loss_single_direction(c_query: Tensor, c_key: Tensor, 
                                        mnn_query_idx: Tensor, mnn_key_idx: Tensor,
                                        temperature: float = 0.1) -> Tensor:
    """
    Single direction MNN contrastive loss using InfoNCE.
    
    Args:
        c_query: Query features [B_query, dim_c]
        c_key: Key features [B_key, dim_c] 
        mnn_query_idx: MNN indices in query batch [N_mnn] (local batch indices)
        mnn_key_idx: MNN indices in key batch [N_mnn] (local batch indices)
        temperature: Temperature parameter for softmax
    
    Returns:
        Contrastive loss (Tensor): Scalar
    """
    if len(mnn_query_idx) == 0:
        return torch.tensor(0.0, device=c_query.device)
    
    device = c_query.device
    
    # Ensure indices are valid and on correct device
    mnn_query_idx = mnn_query_idx.to(device)
    mnn_key_idx = mnn_key_idx.to(device)
    
    # Filter out invalid indices
    valid_mask = (mnn_query_idx < c_query.size(0)) & (mnn_key_idx < c_key.size(0))
    if not valid_mask.any():
        return torch.tensor(0.0, device=device)
    
    mnn_query_idx = mnn_query_idx[valid_mask]
    mnn_key_idx = mnn_key_idx[valid_mask]
    
    # Extract MNN features
    c_query_mnn = c_query[mnn_query_idx]  # [N_mnn, dim_c]
    
    # Normalize features for cosine similarity
    c_query_mnn = F.normalize(c_query_mnn, dim=1)
    c_key_norm = F.normalize(c_key, dim=1)
    
    # Compute similarity matrix: [N_mnn, B_key]
    sim_matrix = torch.mm(c_query_mnn, c_key_norm.t()) / temperature
    
    # Labels are the corresponding key indices for each query
    labels = mnn_key_idx.long()
    
    # InfoNCE loss
    loss = F.cross_entropy(sim_matrix, labels)
    return loss


def bidirectional_mnn_contrastive_loss(c1: Tensor, c2: Tensor,
                                     mnn_idx1: Tensor, mnn_idx2: Tensor,
                                     temperature: float = 0.1) -> Tensor:
    """
    Bidirectional MNN contrastive loss for c features.
    
    This loss encourages MNN pairs to have similar c representations while
    pushing apart non-corresponding cells.
    
    Args:
        c1: Sample 1 c features [B1, dim_c]
        c2: Sample 2 c features [B2, dim_c]
        mnn_idx1: MNN indices in sample 1 [N_mnn]
        mnn_idx2: MNN indices in sample 2 [N_mnn]
        temperature: Temperature parameter for contrastive learning
    
    Returns:
        Bidirectional contrastive loss (Tensor): Scalar
    """
    if len(mnn_idx1) == 0 or len(mnn_idx2) == 0:
        return torch.tensor(0.0, device=c1.device)
    
    # Ensure we have the same number of MNN pairs
    min_pairs = min(len(mnn_idx1), len(mnn_idx2))
    mnn_idx1 = mnn_idx1[:min_pairs]
    mnn_idx2 = mnn_idx2[:min_pairs]
    
    # Direction 1: c1 -> c2 (c1 as query, c2 as key)
    loss_1to2 = mnn_contrastive_loss_single_direction(
        c1, c2, mnn_idx1, mnn_idx2, temperature
    )
    
    # Direction 2: c2 -> c1 (c2 as query, c1 as key)
    loss_2to1 = mnn_contrastive_loss_single_direction(
        c2, c1, mnn_idx2, mnn_idx1, temperature
    )
    
    # Average bidirectional loss
    return (loss_1to2 + loss_2to1) / 2.0


def compute_batch_mnn_pairs(idx1: Tensor, idx2: Tensor, 
                          global_mnn_idx1: Tensor, global_mnn_idx2: Tensor) -> tuple:
    """
    Find MNN pairs that exist in the current batch (optimized with vectorized operations).
    
    Args:
        idx1: Global indices of sample 1 batch [B1]
        idx2: Global indices of sample 2 batch [B2] 
        global_mnn_idx1: Global MNN indices for sample 1 [N_mnn_global]
        global_mnn_idx2: Global MNN indices for sample 2 [N_mnn_global]
    
    Returns:
        batch_mnn_idx1: Local MNN indices in current batch for sample 1 [N_mnn_batch]
        batch_mnn_idx2: Local MNN indices in current batch for sample 2 [N_mnn_batch]
    """
    if global_mnn_idx1 is None or global_mnn_idx2 is None or len(global_mnn_idx1) == 0:
        return torch.tensor([], dtype=torch.long, device=idx1.device), \
               torch.tensor([], dtype=torch.long, device=idx2.device)
    
    device = idx1.device
    
    # Ensure tensors are on the same device and type
    global_mnn_idx1 = global_mnn_idx1.to(device).long() if torch.is_tensor(global_mnn_idx1) else torch.tensor(global_mnn_idx1, device=device, dtype=torch.long)
    global_mnn_idx2 = global_mnn_idx2.to(device).long() if torch.is_tensor(global_mnn_idx2) else torch.tensor(global_mnn_idx2, device=device, dtype=torch.long)
    idx1 = idx1.long()
    idx2 = idx2.long()
    
    # Create sets for fast lookup (using torch operations)
    # Find which global MNN indices are in the current batches
    # For sample 1: find local positions of global_mnn_idx1 in idx1
    # Use broadcasting: [B1, 1] == [1, N_mnn] -> [B1, N_mnn]
    matches1 = (idx1.unsqueeze(1) == global_mnn_idx1.unsqueeze(0))  # [B1, N_mnn]
    matches2 = (idx2.unsqueeze(1) == global_mnn_idx2.unsqueeze(0))  # [B2, N_mnn]
    
    # Find MNN pairs where both samples have the corresponding indices in the batch
    # For each MNN pair position i, check if matches1 has True at some batch pos and matches2 has True at some batch pos
    has_match1 = matches1.any(dim=0)  # [N_mnn] - which MNN pairs have matches in sample 1 batch
    has_match2 = matches2.any(dim=0)  # [N_mnn] - which MNN pairs have matches in sample 2 batch
    valid_pairs = has_match1 & has_match2  # [N_mnn] - pairs that exist in both batches
    
    if not valid_pairs.any():
        return torch.tensor([], dtype=torch.long, device=device), \
               torch.tensor([], dtype=torch.long, device=device)
    
    # Get valid MNN pair indices
    valid_mnn_indices = valid_pairs.nonzero(as_tuple=False).squeeze(1)  # [N_valid]
    
    # For each valid pair, find local batch positions
    batch_mnn_idx1_list = []
    batch_mnn_idx2_list = []
    
    for mnn_idx in valid_mnn_indices:
        # Find local position in sample 1
        local_pos1 = matches1[:, mnn_idx].nonzero(as_tuple=False).squeeze(1)
        # Find local position in sample 2
        local_pos2 = matches2[:, mnn_idx].nonzero(as_tuple=False).squeeze(1)
        
        # Take the first match (or could use all matches for more pairs)
        if len(local_pos1) > 0 and len(local_pos2) > 0:
            batch_mnn_idx1_list.append(local_pos1[0])
            batch_mnn_idx2_list.append(local_pos2[0])
    
    if len(batch_mnn_idx1_list) == 0:
        return torch.tensor([], dtype=torch.long, device=device), \
               torch.tensor([], dtype=torch.long, device=device)
    
    batch_mnn_idx1 = torch.stack(batch_mnn_idx1_list)
    batch_mnn_idx2 = torch.stack(batch_mnn_idx2_list)
    
    return batch_mnn_idx1, batch_mnn_idx2


class MNNDiscriminator(nn.Module):
    """
    Discriminator for MNN mutual information maximization.
    
    Distinguishes between real MNN pairs and random pairs to maximize
    mutual information between corresponding cells across samples.
    """
    
    def __init__(self, latent_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.discriminator = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, c_pairs: Tensor) -> Tensor:
        """
        Args:
            c_pairs: Concatenated c feature pairs [B, 2*dim_c]
        
        Returns:
            scores: Probability scores [B, 1]
        """
        return self.discriminator(c_pairs)


def mnn_mutual_information_loss(c1: Tensor, c2: Tensor,
                               mnn_idx1: Tensor, mnn_idx2: Tensor,
                               discriminator: nn.Module) -> Tensor:
    """
    MNN-based mutual information maximization loss.
    
    Adapts scNiche's mutual information approach to cross-sample MNN pairs.
    
    Args:
        c1: Sample 1 c features [B1, dim_c]
        c2: Sample 2 c features [B2, dim_c]
        mnn_idx1: MNN indices in sample 1 [N_mnn]
        mnn_idx2: MNN indices in sample 2 [N_mnn]
        discriminator: MNN discriminator network
    
    Returns:
        Mutual information loss (Tensor): Scalar
    """
    if len(mnn_idx1) == 0 or len(mnn_idx2) == 0:
        return torch.tensor(0.0, device=c1.device)
    
    device = c1.device
    
    # 1. Create positive pairs (real MNN correspondences)
    c1_mnn = c1[mnn_idx1]  # [N_mnn, dim_c]
    c2_mnn = c2[mnn_idx2]  # [N_mnn, dim_c]
    positive_pairs = torch.cat([c1_mnn, c2_mnn], dim=1)  # [N_mnn, 2*dim_c]
    
    # 2. Create negative pairs (random cross-sample pairs)
    n_mnn = len(mnn_idx1)
    
    # Random sampling from c2 for each c1_mnn
    # Ensure we don't sample more than available in c2
    n_negative = min(n_mnn, c2.size(0))
    if n_negative > 0:
        random_idx2 = torch.randperm(c2.size(0), device=device)[:n_negative]
        c2_random = c2[random_idx2]  # [n_negative, dim_c]
        c1_mnn_neg = c1_mnn[:n_negative]  # [n_negative, dim_c]
        negative_pairs = torch.cat([c1_mnn_neg, c2_random], dim=1)  # [n_negative, 2*dim_c]
    else:
        negative_pairs = torch.zeros((0, c1.shape[1] * 2), device=device)
    
    # 3. Discriminator scores
    positive_scores = discriminator(positive_pairs)  # [N_mnn, 1]
    
    # 4. Mutual information loss (maximize positive, minimize negative)
    # Similar to scNiche's formulation
    n_negative = negative_pairs.size(0)
    if n_negative > 0:
        negative_scores = discriminator(negative_pairs)  # [n_negative, 1]
        # Use only the first n_negative positive scores to match negative scores
        positive_scores_matched = positive_scores[:n_negative]  # [n_negative, 1]
        mi_loss = -torch.mean(
            torch.log(positive_scores_matched + 1e-6) +           # Maximize real MNN pairs
            torch.log(1 - negative_scores + 1e-6)         # Minimize random pairs
        )
    else:
        # If no negative samples, only maximize positive pairs
        mi_loss = -torch.mean(torch.log(positive_scores + 1e-6))
    
    return mi_loss


def enhanced_mnn_mutual_information_loss(c1: Tensor, c2: Tensor,
                                       mnn_idx1: Tensor, mnn_idx2: Tensor,
                                       discriminator: nn.Module,
                                       n_negative_samples: int = 3) -> Tensor:
    """
    Enhanced MNN mutual information loss with multiple negative samples.
    
    Args:
        c1: Sample 1 c features [B1, dim_c]
        c2: Sample 2 c features [B2, dim_c]
        mnn_idx1: MNN indices in sample 1 [N_mnn]
        mnn_idx2: MNN indices in sample 2 [N_mnn]
        discriminator: MNN discriminator network
        n_negative_samples: Number of negative samples per positive pair
    
    Returns:
        Enhanced mutual information loss (Tensor): Scalar
    """
    if len(mnn_idx1) == 0 or len(mnn_idx2) == 0:
        return torch.tensor(0.0, device=c1.device)
    
    device = c1.device
    n_mnn = len(mnn_idx1)
    
    # Limit n_mnn to available batch sizes to avoid dimension mismatch
    n_effective = min(n_mnn, c1.size(0), c2.size(0))
    
    if n_effective == 0:
        return torch.tensor(0.0, device=device)
    
    # 1. Positive pairs (real MNN correspondences)
    c1_mnn = c1[mnn_idx1[:n_effective]]  # [N_effective, dim_c]
    c2_mnn = c2[mnn_idx2[:n_effective]]  # [N_effective, dim_c]
    positive_pairs = torch.cat([c1_mnn, c2_mnn], dim=1)  # [N_effective, 2*dim_c]
    
    # 2. Multiple negative pairs for each positive pair
    negative_pairs_list = []
    
    for _ in range(n_negative_samples):
        # Random sampling strategy 1: Random c2 for each c1_mnn
        random_idx2 = torch.randperm(c2.size(0), device=device)[:n_effective]
        c2_random = c2[random_idx2]
        neg_pairs_1 = torch.cat([c1_mnn, c2_random], dim=1)
        negative_pairs_list.append(neg_pairs_1)
        
        # Random sampling strategy 2: Random c1 for each c2_mnn  
        random_idx1 = torch.randperm(c1.size(0), device=device)[:n_effective]
        c1_random = c1[random_idx1]
        neg_pairs_2 = torch.cat([c1_random, c2_mnn], dim=1)
        negative_pairs_list.append(neg_pairs_2)
    
    # Concatenate all negative pairs
    negative_pairs = torch.cat(negative_pairs_list, dim=0)  # [N_mnn * 2 * n_negative_samples, 2*dim_c]
    
    # 3. Discriminator scores
    positive_scores = discriminator(positive_pairs)  # [N_mnn, 1]
    negative_scores = discriminator(negative_pairs)  # [N_mnn * 2 * n_negative_samples, 1]
    
    # 4. Enhanced mutual information loss
    positive_loss = -torch.mean(torch.log(positive_scores + 1e-6))
    negative_loss = -torch.mean(torch.log(1 - negative_scores + 1e-6))
    
    mi_loss = positive_loss + negative_loss
    
    return mi_loss


