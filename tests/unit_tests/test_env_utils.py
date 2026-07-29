"""Regression tests for environment wrapper compatibility helpers."""

from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym

from rlinf.envs.utils import get_env_attr
from rlinf.envs.wrappers.record_video import RecordVideo


class _WorldModelEnv:
    """Minimal non-Gym environment matching the Wan lifecycle surface."""

    seed = 0

    def offload(self) -> str:
        return "world-offloaded"

    def onload(self) -> str:
        return "world-onloaded"

    def close(self) -> None:
        return None


def test_get_env_attr_crosses_gym_wrapper_to_non_gym_environment() -> None:
    base = _WorldModelEnv()
    wrapped = RecordVideo(base, SimpleNamespace(fps=30))
    try:
        offload = get_env_attr(wrapped, "offload")
        onload = get_env_attr(wrapped, "onload")

        assert callable(offload)
        assert callable(onload)
        assert offload() == "world-offloaded"
        assert onload() == "world-onloaded"
    finally:
        wrapped.close()


def test_get_env_attr_crosses_nested_gym_wrappers() -> None:
    base = _WorldModelEnv()
    inner = gym.Wrapper(base)
    outer = gym.Wrapper(inner)

    assert get_env_attr(outer, "offload")() == "world-offloaded"


def test_get_env_attr_preserves_outer_wrapper_override() -> None:
    class OverrideWrapper(gym.Wrapper):
        def offload(self) -> str:
            return "wrapper-offloaded"

    wrapped = OverrideWrapper(_WorldModelEnv())

    assert get_env_attr(wrapped, "offload")() == "wrapper-offloaded"


def test_get_env_attr_supports_dynamic_terminal_environment() -> None:
    class DynamicEnv:
        def __getattr__(self, name: str):
            if name == "offload":
                return lambda: "dynamic-offloaded"
            raise AttributeError(name)

    assert get_env_attr(DynamicEnv(), "offload")() == "dynamic-offloaded"


def test_get_env_attr_returns_default_for_missing_or_cyclic_wrapper_chain() -> None:
    missing = object()
    left = SimpleNamespace()
    right = SimpleNamespace(env=left)
    left.env = right

    assert get_env_attr(_WorldModelEnv(), "missing", missing) is missing
    assert get_env_attr(left, "missing", missing) is missing
