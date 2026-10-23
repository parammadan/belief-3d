"""Inference-side packages must not be able to reach simulator truth.

Static check over the import graph: nothing under perception/, tracking/,
policies/, manipulation/ or models/ may import mujoco, sim.*, the GroundTruth
schema, or the ground-truth loader. (The robot interface reaches the arm via
an injected object at runtime, not via imports.)
"""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
INFERENCE_PACKAGES = ["perception", "tracking", "policies", "manipulation", "models"]
FORBIDDEN_MODULES = ("mujoco", "sim")
FORBIDDEN_NAMES = {"GroundTruth", "ObjectState", "load_ground_truth"}


def _files():
    for pkg in INFERENCE_PACKAGES:
        yield from (ROOT / pkg).rglob("*.py")


@pytest.mark.parametrize("path", list(_files()), ids=lambda p: str(p.relative_to(ROOT)))
def test_no_ground_truth_imports(path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
            names = set()
        elif isinstance(node, ast.ImportFrom):
            mods = [node.module or ""]
            names = {a.name for a in node.names}
        else:
            continue
        for m in mods:
            assert m.split(".")[0] not in FORBIDDEN_MODULES, f"{path}: imports {m}"
        assert not names & FORBIDDEN_NAMES, f"{path}: imports {names & FORBIDDEN_NAMES}"
