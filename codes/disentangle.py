"""
Disentanglement modules
"""
import torch
from torch import nn


class LD(nn.Module):
    """
    Latent Decomposition (LD) module for extracting aleatoric uncertainty from task-related features.
    """
    def __init__(self, dim, n_round=3, dropout=0.3):
        super(LD, self).__init__()
        self.n_round = n_round
        self.dim = dim
        self.decomposition = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.BatchNorm1d(dim // 4),
            nn.Dropout(p=dropout),
            nn.Mish(),
            nn.Linear(dim // 4, dim),
            nn.BatchNorm1d(dim),
            nn.Dropout(p=dropout),
            nn.Mish(),
        )
        self.weight_net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.BatchNorm1d(dim),
            nn.Dropout(p=dropout),
            nn.Mish(),
            nn.Linear(dim, dim // 4),
            nn.BatchNorm1d(dim // 4),
            nn.Dropout(p=dropout),
            nn.Mish(),
            nn.Linear(dim // 4, 1),
        )

    def forward(self, task_related):
        """
        Extract aleatoric uncertainty from task_related features.
        
        Args:
            task_related (Tensor): Task-related features [B, dim]
        
        Returns:
            task_related (Tensor): Cleaned task-related features [B, dim]
            aleatoric (Tensor): Extracted aleatoric uncertainty [B, dim]
        """
        aleatoric = torch.zeros_like(task_related)
        for _ in range(self.n_round):
            primary = self.decomposition(task_related)
            input_f = torch.cat([task_related, primary], dim=1)
            weight = self.weight_net(input_f)
            aleatoric = aleatoric + primary * weight
            task_related = task_related - primary * weight
        return task_related, aleatoric

