"""Symmetry-invariant pose loss through the primitive's analytic SDF (torch).

For hypothesis (R_k, t_k) and ground truth (R*, t*), transform the model's
surface samples m_j by the hypothesis and measure their distance to the
*true* surface:

    L_k = mean_j | sdf*( R*^T (R_k m_j + t_k - t*) ) |

L_k is zero iff the hypothesised surface coincides with the true one, i.e.
for every pose in the object's symmetry class. So no symmetry group is
enumerated, and continuous symmetries (cylinder, sphere) are exact. Training
uses min_k L_k (multiple-hypothesis prediction, Rupprecht et al., ICCV 2017)
plus cross-entropy teaching the logits which hypothesis won.
"""
from __future__ import annotations

import torch

from data.schemas.objects import CATALOG
from perception.pose.completeness import surface_grid

SHAPE_ID = {"box": 0, "cylinder": 1, "sphere": 2, "mug": 1}   # mug ≈ its body for the (untrained-on-mug) proposer


class SurfaceBank:
    """Per-class padded surface samples and SDF parameters, on a device."""

    def __init__(self, spacing: float = 0.006, device: str | torch.device = "cpu"):
        grids = [surface_grid(o, spacing) for o in CATALOG]
        m = max(len(g) for g in grids)
        pts = torch.zeros(len(CATALOG), m, 3)
        mask = torch.zeros(len(CATALOG), m, dtype=torch.bool)
        for i, g in enumerate(grids):
            pts[i, : len(g)] = torch.as_tensor(g, dtype=torch.float32)
            mask[i, : len(g)] = True
        self.points, self.mask = pts.to(device), mask.to(device)
        self.shape = torch.tensor([SHAPE_ID[o.shape] for o in CATALOG], device=device)
        sizes = torch.zeros(len(CATALOG), 5)
        for i, o in enumerate(CATALOG):
            sizes[i, : len(o.size)] = torch.as_tensor(o.size)
        self.size = sizes.to(device)


def sdf_torch(shape: torch.Tensor, size: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """shape (B,), size (B,3), P (B, ..., 3) object-frame points -> (B, ...) signed distance."""
    view = (-1,) + (1,) * (P.dim() - 2)
    s = size[:, :3].reshape(*view, 3)
    q = P.abs() - s
    box = q.clamp(min=0).norm(dim=-1) + q.max(dim=-1).values.clamp(max=0)
    rxy = P[..., :2].norm(dim=-1)
    d = torch.stack([rxy - s[..., 0], P[..., 2].abs() - s[..., 1]], dim=-1)
    cyl = d.max(dim=-1).values.clamp(max=0) + d.clamp(min=0).norm(dim=-1)
    sph = P.norm(dim=-1) - s[..., 0]
    sh = shape.view(view)
    return torch.where(sh == 0, box, torch.where(sh == 1, cyl, sph))


def hypothesis_losses(bank: SurfaceBank, cls: torch.Tensor, R: torch.Tensor, t: torch.Tensor,
                      R_gt: torch.Tensor, t_gt: torch.Tensor) -> torch.Tensor:
    """(B,K) per-hypothesis mean surface distance in meters. R (B,K,3,3), t (B,K,3)."""
    M = bank.points[cls]                                             # (B, Mp, 3)
    mask = bank.mask[cls].float()                                    # (B, Mp)
    X = torch.einsum("bkij,bmj->bkmi", R, M) + t[:, :, None]        # hypothesis surface, world-relative
    Y = torch.einsum("bji,bkmj->bkmi", R_gt, X - t_gt[:, None, None])   # into GT object frame
    d = sdf_torch(bank.shape[cls], bank.size[cls], Y).abs()          # (B, K, Mp)
    return (d * mask[:, None]).sum(-1) / mask.sum(-1)[:, None]


def mhp_loss(bank: SurfaceBank, cls, R, t, logits, R_gt, t_gt, eps: float = 0.05):
    """Relaxed min-of-K: the winner gets weight 1 - eps, the others share eps (avoids dead heads)."""
    L = hypothesis_losses(bank, cls, R, t, R_gt, t_gt)
    k = L.shape[1]
    win = L.argmin(dim=1)
    w = torch.full_like(L, eps / max(k - 1, 1))
    w.scatter_(1, win[:, None], 1.0 - eps)
    pose = (w * L).sum(1).mean()
    ce = torch.nn.functional.cross_entropy(logits, win)
    return pose, ce, L.min(dim=1).values
