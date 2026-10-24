"""Loader for the optional C++ kernels (cpp/, built with CMake; see README "C++ kernels")."""
from __future__ import annotations

import sys
from functools import cache
from pathlib import Path

_BUILD = Path(__file__).resolve().parents[2] / "cpp" / "build"


@cache
def kernels():
    try:
        import belief3d_cpp
    except ImportError:
        if str(_BUILD) not in sys.path:
            sys.path.append(str(_BUILD))
        try:
            import belief3d_cpp
        except ImportError as e:
            raise ImportError("belief3d_cpp not built: cmake -S . -B cpp/build && cmake --build cpp/build") from e
    return belief3d_cpp


def available() -> bool:
    try:
        kernels()
    except ImportError:
        return False
    return True
