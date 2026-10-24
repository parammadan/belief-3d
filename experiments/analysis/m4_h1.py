"""EXP-M4-001 pre-registered analysis (H1 primary, H2 non-inferiority).

H1: on tracked object-frames with visible fraction < 0.5 (max over the cameras the method used),
B2's rate of symmetry-aware rotation error < 5 deg exceeds B1's. Unit = episode (episodes with
>= 1 such frame for EITHER method are included; a method with no matched track on a
partial-visibility frame scores a failure there). Pooled over occlusion/nominal and
occlusion/harsh. Statistic: per-episode mean(B2 - B1), paired bootstrap 95 % CI, exact two-sided
sign test.

Matching follows benchmarks/perception/eval_m2.py: Hungarian on centre distance, <= 5 cm.

Usage:  PYTHONPATH=. python experiments/analysis/m4_h1.py
"""
from __future__ import annotations

import json
from math import comb
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from benchmarks.metrics import rotation_error
from data.schemas.objects import CATALOG

SPEC = {o.name: o for o in CATALOG}
CONDS = ["occlusion_nominal", "occlusion_harsh"]
ALL = ["occlusion_nominal", "occlusion_harsh", "dynamic_nominal", "dynamic_harsh"]


def per_episode_partial_success(ds: Path, preds: Path, match_dist: float = 0.05) -> dict[str, list[bool]]:
    m = json.loads((ds / "manifest.json").read_text())
    cams = json.loads((preds / "meta.json").read_text())["cameras"]
    out = {}
    for ep in m["episodes"]:
        eid = ep["episode_id"]
        P = json.loads((preds / f"{eid}.json").read_text())
        res = []
        for k in range(ep["n_frames"]):
            gts = {c: json.loads((ds / eid / f"gt_{k:04d}_{c}.json").read_text()) for c in cams}
            ref = gts[cams[0]]["objects"]
            vis = [max(gts[c]["objects"][i]["visible_fraction"] for c in cams) for i in range(len(ref))]
            tracks = P["frames"][k]["tracks"]
            G = np.array([np.array(o["T_WO"])[:3, 3] for o in ref])
            pairs = {}
            if tracks:
                T = np.array([np.array(t["T_WO"])[:3, 3] for t in tracks])
                D = np.linalg.norm(G[:, None] - T[None], axis=2)
                r, c = linear_sum_assignment(np.where(D <= match_dist, D, 1e6))
                pairs = {i: j for i, j in zip(r, c) if D[i, j] <= match_dist}
            for i, o in enumerate(ref):
                if vis[i] >= 0.5 or vis[i] <= 0.0:
                    continue                       # only partially visible (not fully hidden) frames
                if i not in pairs:
                    res.append(False)
                    continue
                spec = SPEC[o["class_name"]]
                e = rotation_error(spec, np.array(tracks[pairs[i]]["T_WO"])[:3, :3], np.array(o["T_WO"])[:3, :3])
                res.append(bool(np.isnan(e) or e < np.radians(5)))
        out[eid] = res
    return out


def paired(d: np.ndarray, n_boot: int = 10000) -> dict:
    rng = np.random.default_rng(0)
    boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    pos, neg = int((d > 1e-12).sum()), int((d < -1e-12).sum())
    n, k = pos + neg, min(pos, neg)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0
    return {"n_episodes": len(d), "mean_diff": float(d.mean()),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "episodes_b2_better": pos, "episodes_b2_worse": neg, "sign_test_p": p}


def main() -> None:
    R = Path("reports/m4")
    diffs, rates = [], {"b1": [], "b2": []}
    for c in CONDS:
        ds, pr = Path(f"datasets/test_m4_{c}"), Path(f"preds/test_m4_{c}")
        a = per_episode_partial_success(ds, pr / "b1")
        b = per_episode_partial_success(ds, pr / "b2")
        for eid in a:
            if a[eid] or b[eid]:
                ra = np.mean(a[eid]) if a[eid] else 0.0
                rb = np.mean(b[eid]) if b[eid] else 0.0
                diffs.append(rb - ra)
                rates["b1"].extend(a[eid])
                rates["b2"].extend(b[eid])
    h1 = paired(np.asarray(diffs)) | {"frames_b1": len(rates["b1"]), "frames_b2": len(rates["b2"]),
                                       "pooled_rate_b1": float(np.mean(rates["b1"])),
                                       "pooled_rate_b2": float(np.mean(rates["b2"]))}
    h2 = {}
    for c in ALL:
        m1 = json.loads((R / f"test_m4_{c}_b1.json").read_text())["tracking"]["MOTA"]
        m2 = json.loads((R / f"test_m4_{c}_b2.json").read_text())["tracking"]["MOTA"]
        h2[c] = {"MOTA_b1": m1, "MOTA_b2": m2, "non_inferior": m2 >= m1 - 0.01}
    out = {"H1": h1, "H2": h2, "H2_all_pass": all(v["non_inferior"] for v in h2.values())}
    (R / "exp_m4_001_analysis.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
