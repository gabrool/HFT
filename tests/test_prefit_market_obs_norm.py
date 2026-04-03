import numpy as np

import RL_exec


class _FakeTrainEnv:
    def __init__(self) -> None:
        self.allow_taker = True
        self._done_after = 3
        self._step_calls = 0
        self._last_action_ids = []

    def set_obs_norm_state(self, state, freeze=False):
        self._set_state = state
        self._freeze = freeze

    def reset(self, start_idx=0):
        self._step_calls = 0
        self._last_action_ids.clear()
        return np.zeros(2, dtype=np.float32)

    def step(self, action, emit_info=False):
        raise AssertionError("prefit loop must not call step(); it should call step_canonical_action_array()")

    def step_canonical_action_array(self, action_arr, *, emit_info=False):
        self._step_calls += 1
        self._last_action_ids.append(id(action_arr))
        done = self._step_calls >= self._done_after
        return np.zeros(2, dtype=np.float32), 0.0, done, None

    def get_obs_norm_state(self):
        return {
            "count": 3,
            "mean": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "m2": np.array([4.0, 5.0, 6.0], dtype=np.float32),
            "continuous_mask": np.array([True, False, True], dtype=bool),
        }


def test_prefit_market_obs_norm_uses_canonical_step_and_preserves_outputs(monkeypatch):
    env = _FakeTrainEnv()
    canonical = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    canonical_calls = {"count": 0}

    def _fake_zero_action(*, allow_taker):
        canonical_calls["count"] += 1
        assert allow_taker is True
        return canonical

    monkeypatch.setattr(RL_exec, "_canonical_zero_market_action", _fake_zero_action)

    state = RL_exec.prefit_market_obs_norm(env)

    assert canonical_calls["count"] == 1
    assert env._step_calls == 3
    assert len(set(env._last_action_ids)) == 1, "prefit loop should reuse canonical action array"

    assert state["count"] == 3
    assert state["mean"] == [1.0, 2.0, 3.0]
    assert state["m2"] == [4.0, 5.0, 6.0]
    assert state["continuous_mask"] == [True, False, True]
