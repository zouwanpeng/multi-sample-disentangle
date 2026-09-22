"""
Utility functions for the disentangled model.
"""
import os
import pickle
import random

from anndata import AnnData
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
import torch
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri, rpy2.robjects.pandas2ri

def seed_everything(seed: int):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pca(adata, use_reps=None, n_comps=10):
    """
    Dimension reduction with PCA algorithm.
    
    Args:
        adata: AnnData object
        use_reps: Key in adata.obsm to use for PCA. If None, uses adata.X
        n_comps: Number of principal components
    
    Returns:
        feat_pca: PCA transformed features [n_cells, n_comps]
    """
    pca_model = PCA(n_components=n_comps)
    if use_reps is not None:
        feat_pca = pca_model.fit_transform(adata.obsm[use_reps])
    else:
        if isinstance(adata.X, (csc_matrix, csr_matrix)):
            feat_pca = pca_model.fit_transform(adata.X.toarray())
        else:
            feat_pca = pca_model.fit_transform(adata.X)
    
    return feat_pca


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2020):
    """
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    
    Args:
        adata: AnnData object
        num_cluster: Number of clusters
        modelNames: Model name for mclust (default: 'EEE')
        used_obsm: Key in adata.obsm to use for clustering
        random_seed: Random seed
    
    Returns:
        adata: AnnData object with clustering results in adata.obs['mclust']
    """
    np.random.seed(random_seed)
    try:
        import rpy2.robjects as robjects
        robjects.r.library("mclust")
        
        import rpy2.robjects.numpy2ri
        rpy2.robjects.numpy2ri.activate()
        r_random_seed = robjects.r['set.seed']
        r_random_seed(random_seed)
        rmclust = robjects.r['Mclust']
        
        res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), 
                     num_cluster, modelNames)
        mclust_res = np.array(res[-2])
        
        adata.obs['mclust'] = mclust_res
        adata.obs['mclust'] = adata.obs['mclust'].astype('int')
        adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    except ImportError:
        raise ImportError("rpy2 is required for mclust clustering. Install with: pip install rpy2")
    
    return adata


def search_res(adata, n_clusters, method='leiden', use_rep='emb', 
               start=0.1, end=3.0, increment=0.01):
    """
    Search for the resolution parameter that yields the target number of clusters.
    
    Args:
        adata: AnnData object
        n_clusters: Target number of clusters
        method: Clustering method ('leiden' or 'louvain')
        use_rep: Key in adata.obsm to use for clustering
        start: Start value for resolution search
        end: End value for resolution search
        increment: Step size for resolution search
    
    Returns:
        res: Resolution value that yields n_clusters
    
    Raises:
        AssertionError: If resolution is not found in the given range
    """
    print('Searching resolution...')
    label = 0
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
            sc.tl.leiden(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
            print(f'resolution={res:.4f}, cluster number={count_unique}')
        elif method == 'louvain':
            sc.tl.louvain(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique())
            print(f'resolution={res:.4f}, cluster number={count_unique}')
        else:
            raise ValueError(f"Unsupported method: {method}. Use 'leiden' or 'louvain'.")
        
        if count_unique == n_clusters:
            label = 1
            break
    
    assert label == 1, "Resolution is not found. Please try bigger range or smaller step!"
    
    return res


def clustering(adata: AnnData,
               n_clusters: int = 7,
               key: str = 'emb',
               add_key: str = 'clustering',
               method: str = 'kmeans',
               use_pca: bool = False,
               n_comps: int = 20,
               resolution: float = None,
               n_neighbors: int = 20,
               start: float = 0.1,
               end: float = 3.0,
               increment: float = 0.01,
               mclust_model: str = 'EEE',
               mclust_seed: int = 2020,
               kmeans_seed: int = 123,
               use_niche_labels: bool = False):
    """
    Perform clustering on spatial data based on latent representations.
    
    Supports multiple clustering methods: kmeans, mclust, leiden, louvain.
    
    Args:
        adata: AnnData object
        n_clusters: Target number of clusters
        key: Key in adata.obsm containing the representation to cluster
        add_key: Key to store clustering results in adata.obs
        method: Clustering method. Options: 'kmeans', 'mclust', 'leiden', 'louvain'
        use_pca: Whether to apply PCA before clustering
        n_comps: Number of PCA components (if use_pca=True)
        resolution: Resolution parameter for leiden/louvain (if None, will search)
        n_neighbors: Number of neighbors for leiden/louvain
        start: Start value for resolution search (if resolution=None)
        end: End value for resolution search (if resolution=None)
        increment: Step size for resolution search (if resolution=None)
        mclust_model: Model name for mclust
        mclust_seed: Random seed for mclust
        kmeans_seed: Random seed for kmeans
        use_niche_labels: If True, use 'Niche0', 'Niche1', ... labels. 
                         If False, use numeric labels.
    
    Returns:
        adata: AnnData object with clustering results in adata.obs[add_key]
    
    Raises:
        ValueError: If method is not supported
        KeyError: If key is not found in adata.obsm
    """
    # Validate method
    method = method.lower()
    supported_methods = ['kmeans', 'mclust', 'leiden', 'louvain']
    if method not in supported_methods:
        raise ValueError(f"Unsupported method: {method}. Choose from {supported_methods}")
    
    # Check if key exists
    if key not in adata.obsm:
        raise KeyError(f"Key '{key}' not found in adata.obsm. Available keys: {list(adata.obsm.keys())}")
    
    # Apply PCA if requested
    if use_pca:
        adata.obsm[key + '_pca'] = pca(adata, use_reps=key, n_comps=n_comps)
        use_rep = key + '_pca'
    else:
        use_rep = key
    
    # Get representation
    X = adata.obsm[use_rep]
    labels = None
    
    # Perform clustering
    if method == 'kmeans':
        print(f"Applying K-Means clustering with {n_clusters} clusters...")
        labels = KMeans(n_clusters=n_clusters, random_state=kmeans_seed, n_init=10).fit_predict(X)
        
    elif method == 'mclust':
        print(f"Applying mclust clustering with {n_clusters} clusters...")
        np.random.seed(mclust_seed)
        robjects.r.library("mclust")

        rpy2.robjects.numpy2ri.activate()
        r_random_seed = robjects.r['set.seed']
        r_random_seed(mclust_seed)
        rmclust = robjects.r['Mclust']

        mclust_res = np.array(rmclust(np.array(adata.obsm[key]), n_clusters, mclust_model)[-2])

        adata.obs['mclust'] = mclust_res
        adata.obs['mclust'] = adata.obs['mclust'].astype('int')
        adata.obs['mclust'] = adata.obs['mclust'].astype('category')
        labels = adata.obs['mclust'].values
        
    elif method in ['leiden', 'louvain']:
        # Build neighbor graph
        print(f"Building neighbor graph (n_neighbors={n_neighbors})...")
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=use_rep)
        
        # Determine resolution
        if resolution is None:
            print(f"Searching for resolution to get {n_clusters} clusters...")
            resolution = search_res(adata, n_clusters, method=method, use_rep=use_rep,
                                   start=start, end=end, increment=increment)
        else:
            print(f"Using fixed resolution={resolution}...")
        
        # Apply clustering
        if method == 'leiden':
            sc.tl.leiden(adata, random_state=0, resolution=resolution)
            labels = adata.obs['leiden'].values
        else:  # louvain
            sc.tl.louvain(adata, random_state=0, resolution=resolution)
            labels = adata.obs['louvain'].values
    
    # Format labels
    if use_niche_labels:
        # Convert to 'Niche0', 'Niche1', ... format
        unique_labels = sorted(np.unique(labels))
        label_map = {label: f'Niche{i}' for i, label in enumerate(unique_labels)}
        formatted_labels = [label_map[label] for label in labels]
        categories = [f'Niche{i}' for i in range(len(unique_labels))]
        adata.obs[add_key] = pd.Categorical(formatted_labels, categories=categories, ordered=True)
    else:
        # Use numeric labels
        if isinstance(labels[0], (int, np.integer)):
            adata.obs[add_key] = labels
        else:
            # Convert string labels to numeric if needed
            unique_labels = sorted(np.unique(labels))
            label_map = {label: i for i, label in enumerate(unique_labels)}
            numeric_labels = [label_map[label] for label in labels]
            adata.obs[add_key] = np.array(numeric_labels, dtype=int)
    
    return adata


# ============================================================================
# Compute Evaluation Metrics
# ============================================================================

def compute_metrics(adata, ground_truth_key='layer_guess'):
    """
    Compute clustering evaluation metrics.
    
    If ground truth is available, computes ARI and NMI.
    Otherwise, computes Silhouette score.
    
    Args:
        adata: AnnData object
        ground_truth_key: Key in adata.obs containing ground truth labels
    
    Returns:
        dict: Dictionary of metrics for each representation
    """
    results = {}
    
    if ground_truth_key in adata.obs and ground_truth_key is not None:
        gt = adata.obs[ground_truth_key]
        valid_idx = gt != 'NA'  # Filter out NA labels if any
        
        for key, name in [('cluster_c', 'Shared (c)'), 
                         ('cluster_u', 'Specific (u)')]:
            if key not in adata.obs:
                continue
                
            pred = adata.obs[key]
            
            # ARI and NMI
            try:
                ari = adjusted_rand_score(gt[valid_idx], pred[valid_idx])
                nmi = normalized_mutual_info_score(gt[valid_idx], pred[valid_idx])
                results[name] = {'ARI': ari, 'NMI': nmi}
            except Exception as e:
                print(f"Warning: Could not compute metrics for {name}: {e}")
                results[name] = {'ARI': None, 'NMI': None}
    else:
        # No ground truth, use silhouette score
        for key, name, emb_key in [('cluster_c', 'Shared (c)', 'c_shared'), 
                                   ('cluster_u', 'Specific (u)', 'u_specific')]:
            if key not in adata.obs or emb_key not in adata.obsm:
                continue
                
            try:
                labels = adata.obs[key].astype(int)
                emb = adata.obsm[emb_key]
                
                # Silhouette score
                sil = silhouette_score(emb, labels)
                results[name] = {'Silhouette': sil}
            except Exception as e:
                print(f"Warning: Could not compute silhouette score for {name}: {e}")
                results[name] = {'Silhouette': None}
    
    return results


def plot_sample_comparison(adata, metrics, sample_name='Sample', spot_sz=6, figsize=(10, 5)):
    """
    Plot two representations side by side with ARI scores in titles
    
    Args:
        adata: AnnData object
        metrics: Dictionary with metrics from compute_metrics()
        sample_name: Name for the sample
        spot_sz: Spot size for spatial plot
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Extract ARI scores if available
    ari_c = metrics.get('Shared (c)', {}).get('ARI', None)
    ari_u = metrics.get('Specific (u)', {}).get('ARI', None)
    
    # Build titles with ARI
    title_c = f'{sample_name}: Shared (c)'
    if ari_c is not None:
        title_c += f' | ARI={ari_c:.3f}'
    
    title_u = f'{sample_name}: Specific (u)'
    if ari_u is not None:
        title_u += f' | ARI={ari_u:.3f}'
    
    # Plot shared (c)
    sc.pl.spatial(adata, img_key="hires", color="cluster_c", 
                 ax=axes[0], title=title_c, 
                 spot_size=spot_sz, show=False)
    
    # Plot specific (u)
    sc.pl.spatial(adata, img_key="hires", color="cluster_u", 
                 ax=axes[1], title=title_u, 
                 spot_size=spot_sz, show=False)
    
    plt.tight_layout()
    return fig