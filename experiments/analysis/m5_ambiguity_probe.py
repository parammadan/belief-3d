"""Cheap regime probe for M5 design (val only): how often is the grasp target ambiguous at t0?

Runs only the policy-independent part of an M5 episode (t0 + warm-up observation) and records
the target's grasp-now risk and mode count, per regime. Usage:
  MUJOCO_GL=cgl PYTHONPATH=. python experiments/analysis/m5_ambiguity_probe.py --episodes 12
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from data.schemas.objects import CATALOG
from data.schemas.splits import seeds_for
from policies.predict import SignatureRenderer, modes_of
from policies.viewpoint import PolicyContext, TaskRiskPolicy
from scripts.run_m5 import Episode, step_tracker
from tracking.belief.particle import BeliefConfig
from tracking.belief_tracker import BeliefTracker
from tracking.tracker import TrackerConfig

SPEC = {o.name: o for o in CATALOG}


def probe(seed, scenario, noise, cams):
    ep = Episode(seed, scenario, noise, cams)
    tr = BeliefTracker(TrackerConfig(calib_rot_sigma=np.radians(1.0), calib_trans_sigma=0.005), BeliefConfig(), seed=seed)
    tracks, K = step_tracker(tr, ep.observe(), 0)
    tracks, _ = step_tracker(tr, ep.observe(), 100)
    out = None
    if tracks:
        plane = next((t.belief.plane for t in tracks if t.belief.plane is not None), None)
        r = SignatureRenderer(K)
        risk = TaskRiskPolicy()
        best = None
        for t in tracks:
            occl = [(SPEC[o.spec_name], o.T_WO) for o in tr.tracks if o is not t]
            ctx = PolicyContext(SPEC[t.spec_name], t.belief, occl, plane, [v.T_WC for v in ep.views.views],
                                ep.view, ep.views.travel_time, r)
            l = risk.grasp_now_loss(ctx)
            if best is None or l > best[0]:
                best = (l, len(modes_of(t.belief).weights))
        out = {"risk": float(best[0]), "modes": int(best[1]), "n_tracks": len(tracks), "n_objects": len(ep.scenario.objects)}
    ep.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--regimes", default="occlusion:nominal:mobile,occlusion:harsh:mobile,clutter:nominal:mobile,"
                                         "clutter:harsh:mobile,clutter:nominal:overhead+mobile")
    a = ap.parse_args()
    regimes = [(r.split(":")[0], r.split(":")[1], tuple(r.split(":")[2].split("+"))) for r in a.regimes.split(",")]
    for sc, nz, cams in regimes:
        rows = [probe(s, sc, nz, cams) for s in seeds_for("val", a.episodes, 800)]
        rows = [r for r in rows if r]
        amb = [r for r in rows if r["modes"] >= 2 or r["risk"] > 0.1]
        print(json.dumps({"regime": f"{sc}/{nz}/{'+'.join(cams)}", "n": len(rows), "ambiguous": len(amb),
                          "risk_mean": round(float(np.mean([r['risk'] for r in rows])), 3),
                          "risk>0.3": int(sum(r["risk"] > 0.3 for r in rows)),
                          "tracks/objects": round(float(np.mean([r['n_tracks'] / r['n_objects'] for r in rows])), 2)}))


if __name__ == "__main__":
    main()
