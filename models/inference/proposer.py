"""Neural pose proposals for one segment: K world-frame hypotheses, by confidence."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from data.schemas.objects import CATALOG
from models.architectures.pointnet_pose import PointNetPose

CLS = {o.name: i for i, o in enumerate(CATALOG)}


class NeuralProposer:
    def __init__(self, checkpoint: str | Path, device: str = "cpu", n_points: int = 512, seed: int = 0):
        ck = torch.load(checkpoint, map_location=device, weights_only=True)
        self.model = PointNetPose(ck["n_classes"], k=ck["k"]).to(device).eval()
        self.model.load_state_dict(ck["model"])
        self.device, self.n_points = device, n_points
        self.rng = np.random.default_rng(seed)

    @torch.no_grad()
    def propose(self, points_W: np.ndarray, spec_name: str) -> tuple[list[np.ndarray], np.ndarray]:
        """Returns (hypotheses T_WO sorted by confidence, softmax confidences)."""
        cen = points_W.mean(0)
        P = points_W[self.rng.choice(len(points_W), self.n_points, replace=len(points_W) < self.n_points)] - cen
        R, t, logits = self.model(torch.as_tensor(P, dtype=torch.float32, device=self.device)[None],
                                  torch.tensor([CLS[spec_name]], device=self.device))
        conf = torch.softmax(logits[0], 0).cpu().numpy()
        order = np.argsort(-conf)
        Ts = []
        for k in order:
            T = np.eye(4)
            T[:3, :3] = R[0, k].cpu().numpy()
            T[:3, 3] = t[0, k].cpu().numpy() + cen
            Ts.append(T)
        return Ts, conf[order]
