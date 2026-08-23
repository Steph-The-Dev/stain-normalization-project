"""
PatchNCE (Patch-based Noise-Contrastive Estimation) Loss Module for PyTorch.

Enforces content representation consistency across corresponding spatial patches
between input source image X and translated output G(X) in Contrastive Unpaired Translation (CUT).
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchNCELoss(nn.Module):
    """
    Patch-based Noise-Contrastive Estimation Loss (Park et al., CUT 2020).
    """

    def __init__(self, nce_temp: float = 0.07, num_patches: int = 64) -> None:
        """
        Args:
            nce_temp: Temperature parameter for scaling similarity logits.
            num_patches: Number of spatial patch locations sampled per feature map.
        """
        super().__init__()
        self.nce_temp = nce_temp
        self.num_patches = num_patches
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="mean")

    def forward(
        self,
        src_feats: List[torch.Tensor],
        tgt_feats: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Computes PatchNCE loss over extracted feature maps.

        Args:
            src_feats: List of feature map tensors from source image X.
            tgt_feats: List of corresponding feature map tensors from translated image G(X).

        Returns:
            Scalar Loss Tensor.
        """
        total_nce_loss = torch.tensor(0.0, device=src_feats[0].device)

        for feat_src, feat_tgt in zip(src_feats, tgt_feats):
            b, c, h, w = feat_src.shape
            spatial_dim = h * w

            if spatial_dim == 0:
                continue

            # Reshape feature maps to (B * H * W, C)
            feat_src_flat = feat_src.permute(0, 2, 3, 1).reshape(-1, c)
            feat_tgt_flat = feat_tgt.permute(0, 2, 3, 1).reshape(-1, c)

            # L2 Normalize feature vectors
            feat_src_norm = F.normalize(feat_src_flat, dim=1)
            feat_tgt_norm = F.normalize(feat_tgt_flat, dim=1)

            # Sample random patch indices if spatial_dim > num_patches
            n_samples = min(self.num_patches, spatial_dim)
            sample_ids = torch.randperm(spatial_dim, device=feat_src.device)[:n_samples]

            # Sample positive query and key vectors
            query = feat_tgt_norm[sample_ids]  # (n_samples, C)
            key = feat_src_norm[sample_ids]    # (n_samples, C)

            # Cosine similarity logits
            # Positive pairs: (n_samples, 1)
            pos_logits = torch.sum(query * key, dim=1, keepdim=True) / self.nce_temp

            # All-pairs similarity matrix: (n_samples, n_samples)
            all_logits = torch.mm(query, key.t()) / self.nce_temp

            # Target labels for CrossEntropy: diagonal indices 0..n_samples-1
            labels = torch.arange(n_samples, device=feat_src.device, dtype=torch.long)

            loss = self.cross_entropy_loss(all_logits, labels)
            total_nce_loss += loss

        return total_nce_loss / max(1, len(src_feats))
