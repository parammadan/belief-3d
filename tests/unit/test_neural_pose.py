"""Neural pose components. Skipped where torch is unavailable (Mac dev box, CI)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from data.schemas.objects import CATALOG
from models.architectures.pointnet_pose import PointNetPose, rot6d_to_matrix
from models.losses.sdf_loss import SurfaceBank, hypothesis_losses, mhp_loss, sdf_torch
from perception.geometry import se3
from perception.pose.primitive_fit import sdf


def test_rot6d_is_orthonormal():
    R = rot6d_to_matrix(torch.randn(100, 6))
    eye = torch.eye(3).expand(100, 3, 3)
    assert torch.allclose(R.transpose(1, 2) @ R, eye, atol=1e-5)
    assert torch.allclose(torch.linalg.det(R), torch.ones(100), atol=1e-5)


@pytest.mark.parametrize("i", range(len(CATALOG)))
def test_torch_sdf_matches_numpy(i):
    bank = SurfaceBank()
    P = np.random.default_rng(i).normal(0, 0.04, (200, 3))
    ref = sdf(CATALOG[i], P)
    got = sdf_torch(bank.shape[i:i + 1], bank.size[i:i + 1], torch.as_tensor(P, dtype=torch.float32)[None])[0]
    assert np.allclose(got.numpy(), ref, atol=1e-6)


def test_loss_zero_at_truth_and_symmetries_positive_elsewhere():
    bank = SurfaceBank()
    cls = torch.tensor([0, 1, 2])                     # box_red, cyl_green, sphere_blue
    R_gt = torch.as_tensor(np.stack([se3.so3_exp([0.1, 0.2, 0.3])] * 3), dtype=torch.float32)
    t_gt = torch.zeros(3, 3)
    sym = torch.as_tensor(np.stack([se3.so3_exp([0, 0, np.pi]), se3.so3_exp([0, 0, 0.7]),
                                    se3.so3_exp([0.4, -1.0, 2.0])]), dtype=torch.float32)
    wrong = torch.as_tensor(se3.so3_exp([np.pi / 2, 0, 0]), dtype=torch.float32).expand(3, 3, 3)
    R = torch.stack([R_gt, R_gt @ sym, R_gt @ wrong], dim=1)          # (3, K=3, 3, 3)
    t = torch.zeros(3, 3, 3)
    t[:, 2] = 0.0
    L = hypothesis_losses(bank, cls, R, t, R_gt, t_gt)
    assert torch.all(L[:, 0] < 1e-6) and torch.all(L[:, 1] < 1e-6)  # truth and a symmetric copy
    assert L[0, 2] > 1e-3 and L[1, 2] > 1e-3                       # wrong box / cylinder pose
    assert L[2, 2] < 1e-6                                          # any rotation is right for a sphere
    t2 = t.clone()
    t2[:, 0, 0] = 0.003
    assert torch.all(hypothesis_losses(bank, cls, R, t2, R_gt, t_gt)[:, 0] > 1e-4)


def test_model_shapes_and_overfits_tiny_batch():
    torch.manual_seed(0)
    bank = SurfaceBank()
    model = PointNetPose(len(CATALOG), k=4)
    B = 16
    cls = torch.randint(0, len(CATALOG), (B,))
    R_gt = rot6d_to_matrix(torch.randn(B, 6))
    t_gt = 0.01 * torch.randn(B, 3)
    pts = torch.randn(B, 128, 3) * 0.02
    R, t, lg = model(pts, cls)
    assert R.shape == (B, 4, 3, 3) and t.shape == (B, 4, 3) and lg.shape == (B, 4)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    first = None
    for _ in range(150):
        R, t, lg = model(pts, cls)
        pose, ce, best = mhp_loss(bank, cls, R, t, lg, R_gt, t_gt)
        loss = pose + 0.01 * ce
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else float(best.mean())
    assert float(best.mean()) < 0.5 * first
