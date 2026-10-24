"""PERF-M7-005: NumPy vs C++ free-space likelihood, same inputs, same numbers.

    python benchmarks/performance/free_space_bench.py [--out reports/m7/free_space_bench.json]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import time

import numpy as np

from data.schemas.objects import CATALOG
from tests.unit.test_belief import T_WC_TOP, K, _two_face_hypotheses, grid, zbuffer
from tracking.belief.particle import BeliefConfig, DepthView, PoseBelief


def timeit(f, reps):
    f()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 1e3), float(np.percentile(ts, 95) * 1e3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    _, T, T_y = _two_face_hypotheses()
    view = DepthView(zbuffer(np.vstack([grid(-0.15, 0.15, -0.15, 0.15, 0.40, 0.0008),
                                        grid(-0.03, 0.03, -0.02, 0.02, 0.45, 0.0005)])), K, T_WC_TOP)
    rows = []
    for spec in CATALOG:
        for n in (256, 1024):
            cfg = BeliefConfig(n_particles=n)
            b = PoseBelief(spec, [T, T_y], 0.0, cfg, seed=0)
            ref = b.free_space_log_lik(view, 0.002)
            np_med, np_p95 = timeit(lambda b=b: b.free_space_log_lik(view, 0.002), a.reps)
            b.cfg = dataclasses.replace(cfg, fs_backend="cpp")
            out = b.free_space_log_lik(view, 0.002)
            cpp_med, cpp_p95 = timeit(lambda b=b: b.free_space_log_lik(view, 0.002), a.reps)
            r = {"object": spec.name, "particles": n, "samples": len(b._S_O),
                 "numpy_ms": np_med, "numpy_p95_ms": np_p95, "cpp_ms": cpp_med, "cpp_p95_ms": cpp_p95,
                 "speedup": np_med / cpp_med, "max_abs_diff": float(np.abs(out - ref).max())}
            rows.append(r)
            print(f"{spec.name:12s} N={n:5d} M={r['samples']:5d}  numpy {np_med:7.2f} ms  cpp {cpp_med:6.2f} ms  "
                  f"x{r['speedup']:5.1f}  maxdiff {r['max_abs_diff']:.1e}")
    # End to end: 5 full updates (progressive correction + resample-move), same seed both backends.
    spec, T, T_y = _two_face_hypotheses()
    P = grid(-0.03, 0.03, -0.02, 0.02, 0.45, 0.002)
    e2e = {}
    for backend in ("numpy", "cpp"):
        b = PoseBelief(spec, [T, T_y], 0.0, BeliefConfig(fs_backend=backend), seed=0)
        t0 = time.perf_counter()
        for k in range(5):
            b.predict(0.1 * (k + 1))
            b.update(P, 0.001, views=[view, view])
        e2e[backend] = {"ms_per_update": (time.perf_counter() - t0) / 5 * 1e3, "x": b.x.copy(), "R": b.R.copy()}
    same = bool(np.allclose(e2e["numpy"]["x"], e2e["cpp"]["x"], atol=1e-9)
                and np.allclose(e2e["numpy"]["R"], e2e["cpp"]["R"], atol=1e-9))
    upd = {k: v["ms_per_update"] for k, v in e2e.items()}
    print(f"end-to-end update: numpy {upd['numpy']:.1f} ms  cpp {upd['cpp']:.1f} ms  "
          f"x{upd['numpy'] / upd['cpp']:.2f}  identical particles after 5 updates: {same}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"machine": platform.platform(), "processor": platform.processor(), "single_thread": True,
                       "reps": a.reps, "rows": rows, "update_ms": upd, "update_particles_identical": same},
                      f, indent=1)


if __name__ == "__main__":
    main()
