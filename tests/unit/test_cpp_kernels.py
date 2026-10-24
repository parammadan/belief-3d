"""The C++ free-space kernel must reproduce the NumPy reference (skipped when not built)."""
import dataclasses

import numpy as np
import pytest

from tests.unit.test_belief import T_WC_TOP, K, _two_face_hypotheses, grid, zbuffer
from tracking.belief import _cpp
from tracking.belief.particle import BeliefConfig, DepthView, PoseBelief

pytestmark = pytest.mark.skipif(not _cpp.available(), reason="belief3d_cpp not built")


def _scene():
    table = grid(-0.15, 0.15, -0.15, 0.15, 0.40, 0.0008)
    top = grid(-0.03, 0.03, -0.02, 0.02, 0.45, 0.0005)
    return DepthView(zbuffer(np.vstack([table, top])), K, T_WC_TOP)


@pytest.mark.parametrize("name", ["box_red", "cyl_green", "sphere_blue", "mug_cyan"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_cpp_free_space_matches_numpy(name, seed):
    from data.schemas.objects import CATALOG
    spec, T, T_y = _two_face_hypotheses()
    spec = {o.name: o for o in CATALOG}[name] if name != "box_red" else spec
    view = _scene()
    cfg = BeliefConfig(init_t_sigma=0.02, init_r_sigma=0.8)      # wide cloud: many violations and misses
    b = PoseBelief(spec, [T, T_y], 0.0, cfg, seed=seed)
    ref = b.free_space_log_lik(view, 0.002)
    b.cfg = dataclasses.replace(cfg, fs_backend="cpp")
    out = b.free_space_log_lik(view, 0.002)
    assert (ref < 0).sum() > 10                                    # the test exercises violations
    assert np.allclose(out, ref, rtol=1e-9, atol=1e-9)


def test_cpp_rejects_bad_shapes():
    m = _cpp.kernels()
    with pytest.raises(ValueError):
        m.free_space_log_lik(np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((4, 3)), np.zeros((4, 3)), np.eye(4),
                             1, 1, 0, 0, np.zeros((4, 4), np.float32), 1, 1, 0, 1, 1)
