import numpy as np

from data.generators.episodes import EpisodeConfig, run_episode
from sim.environments.scenario import PRESET_SCENARIOS, Scenario


def test_scenario_sampling_is_seeded():
    cfg = PRESET_SCENARIOS["dynamic"]
    a = Scenario.sample(cfg, np.random.default_rng(4), 12)
    b = Scenario.sample(cfg, np.random.default_rng(4), 12)
    c = Scenario.sample(cfg, np.random.default_rng(5), 12)
    assert a == b and a != c


def test_similar_pair_and_object_count():
    for s in range(20):
        sc = Scenario.sample(PRESET_SCENARIOS["dynamic"], np.random.default_rng(s), 10)
        names = [o.name for o in sc.objects]
        assert "box_red" in names and "box_red_b" in names
        assert 3 <= len(names) <= 6 and len(set(names)) == len(names)


def test_occlusion_interval_is_complete_from_overhead():
    frames = list(run_episode(3, EpisodeConfig(n_frames=12, scenario="occlusion", noise="clean")))
    sc = frames[0].scenario
    t, (a, b) = sc.occl_target, sc.occl_frames
    for fr in frames:
        v = fr.gt["overhead"].visible_fraction[t]
        if a <= fr.index < b:
            assert v < 0.05, (fr.index, v)
        else:
            assert v > 0.9, (fr.index, v)


def test_kick_moves_only_after_kick_frame():
    frames = list(run_episode(5, EpisodeConfig(n_frames=12, scenario="dynamic", noise="clean")))
    sc = frames[0].scenario
    pos = np.array([fr.gt["overhead"].objects[sc.kick_target].T_WO[:3, 3] for fr in frames])
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    assert np.all(step[: sc.kick_frame - 1] < 1e-4)
    assert step[sc.kick_frame - 1:].max() > 0.005
