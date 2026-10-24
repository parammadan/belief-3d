"""Multi-hypothesis 6D pose proposer (level-2 neural baseline).

PointNet encoder (Qi et al., CVPR 2017) over the centred segment points plus a
class embedding. The head outputs K hypotheses, each a rotation in the
continuous 6D representation (Zhou et al., CVPR 2019) and a translation
offset from the point centroid, plus K confidence logits. Trained with a
min-of-K (multiple-hypothesis) symmetry-invariant loss (models/losses).
"""
from __future__ import annotations

import torch
from torch import nn


def rot6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    """(..., 6) -> (..., 3, 3) by Gram-Schmidt on the two predicted columns."""
    a, b = x[..., :3], x[..., 3:]
    c1 = nn.functional.normalize(a, dim=-1)
    c2 = nn.functional.normalize(b - (c1 * b).sum(-1, keepdim=True) * c1, dim=-1)
    c3 = torch.cross(c1, c2, dim=-1)
    return torch.stack([c1, c2, c3], dim=-1)


class PointNetPose(nn.Module):
    def __init__(self, n_classes: int, k: int = 4, width: int = 256, emb: int = 32):
        super().__init__()
        self.k = k
        self.enc = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.GroupNorm(8, 64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.GroupNorm(8, 128), nn.ReLU(),
            nn.Conv1d(128, width, 1), nn.GroupNorm(8, width), nn.ReLU(),
            nn.Conv1d(width, 2 * width, 1),
        )
        self.cls_emb = nn.Embedding(n_classes, emb)
        self.head = nn.Sequential(
            nn.Linear(2 * width + emb, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, k * 10),
        )

    def forward(self, points: torch.Tensor, cls: torch.Tensor):
        """points (B, P, 3) centred; cls (B,). Returns R (B,K,3,3), t (B,K,3), logits (B,K)."""
        f = self.enc(points.transpose(1, 2)).max(dim=2).values
        out = self.head(torch.cat([f, self.cls_emb(cls)], dim=1)).view(-1, self.k, 10)
        R = rot6d_to_matrix(out[..., :6])
        t = 0.05 * torch.tanh(out[..., 6:9])        # offsets bounded to +-5 cm from the centroid
        return R, t, out[..., 9]
