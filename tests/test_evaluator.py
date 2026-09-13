import sys
import types

import numpy as np
import pytest

from post_train_vla import evaluator


@pytest.mark.parametrize("save_video", [False, True])
def test_rollout_uses_fresh_observations_and_preserves_video_frames(tmp_path, monkeypatch, save_video):
    """Skipping preprocessing must not reuse stale inputs or drop video frames."""
    task = types.SimpleNamespace(language="test task")
    suite = types.SimpleNamespace(
        n_tasks=1, get_task=lambda _: task, get_task_init_states=lambda _: [None]
    )
    libero = types.ModuleType("libero.libero")
    libero.benchmark = types.SimpleNamespace(get_benchmark_dict=lambda: {"libero_spatial": lambda: suite})
    monkeypatch.setitem(sys.modules, "libero", types.ModuleType("libero"))
    monkeypatch.setitem(sys.modules, "libero.libero", libero)
    prepared = []
    inferred = []
    frames = []

    class Environment:
        step_index = 0
        closed = False
        actions = []

        def reset(self):
            pass

        def set_init_state(self, state):
            return 0

        def step(self, action):
            self.actions.append(action)
            self.step_index += 1
            return self.step_index, 0, self.step_index == 12, {}

        def close(self):
            self.closed = True

    class Policy:
        def reset(self):
            pass

        def infer(self, observation):
            index = observation["observation/image"]
            inferred.append(index)
            return {"actions": np.full((5, 7), index, dtype=np.float32)}

    def prepare(observation, prompt, size):
        prepared.append(observation)
        return {"observation/image": observation}

    environment = Environment()
    monkeypatch.setattr(evaluator, "_create_environment", lambda *args: environment)
    monkeypatch.setattr(evaluator, "make_policy_observation", prepare)
    monkeypatch.setattr(evaluator.imageio, "mimwrite", lambda path, values, fps: frames.extend(values))
    summary = evaluator.evaluate(
        Policy(),
        evaluator.EvalConfig(episodes_per_task=1, wait_steps=0, output_dir=tmp_path, save_video=save_video),
    )

    assert inferred == [0, 5, 10]
    assert prepared == (list(range(12)) if save_video else [0, 5, 10])
    assert frames == (list(range(12)) if save_video else [])
    np.testing.assert_array_equal(environment.actions, [[0] * 7] * 5 + [[5] * 7] * 5 + [[10] * 7] * 2)
    assert summary["successes"] == 1
    assert environment.closed
