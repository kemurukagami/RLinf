# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common utilities for world model based environments."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Mapping, Optional, Union

import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.envs.utils import recursive_to_device
from rlinf.scheduler import Worker, WorkerInfo
from rlinf.utils.nested_dict_process import clone_nested_to_cpu

WORLD_ENV_RESUME_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class WorldEnvSnapshotContext:
    worker_rank: int
    worker_world_size: int
    stage_id: int
    lifecycle_generation: int
    chunk_index: int
    next_transition_id: int
    episode_generations: torch.Tensor
    reset_state_ids: torch.Tensor


@dataclass(frozen=True, slots=True)
class WorldEnvResumeState:
    schema_version: int
    environment_type: str
    config_fingerprint: str
    worker_rank: int
    worker_world_size: int
    stage_id: int
    lifecycle_generation: int
    chunk_index: int
    next_transition_id: int
    episode_generations: torch.Tensor
    reset_state_ids: torch.Tensor
    reset_generator_state: torch.Tensor
    diffusion_generator_state: torch.Tensor | None
    diffusion_seed: int | None
    current_obs: Any
    image_queue: tuple[tuple[Any, ...], ...]
    condition_action: torch.Tensor | None
    task_descriptions: tuple[str, ...]
    init_ee_poses: tuple[Any, ...]
    elapsed_steps: int
    prev_step_reward: torch.Tensor
    success_once: torch.Tensor | None
    returns: torch.Tensor | None
    is_start: bool


@dataclass(slots=True)
class _PreparedWorldEnvState:
    current_obs: Any
    image_queue: tuple[tuple[Any, ...], ...]
    condition_action: torch.Tensor | None
    task_descriptions: list[str]
    init_ee_poses: list[Any]
    elapsed_steps: int
    prev_step_reward: torch.Tensor
    success_once: torch.Tensor | None
    returns: torch.Tensor | None
    is_start: bool
    reset_state_ids: torch.Tensor
    episode_generations: torch.Tensor
    reset_generator_state: torch.Tensor
    diffusion_generator_state: torch.Tensor | None
    diffusion_seed: int | None
    model_state: Any


class BaseWorldEnv(ABC):
    """Base class that provides shared utilities for world model environments.

    Subclasses are expected to implement dataset creation as well as any
    environment-specific logic such as model loading, stepping, and rendering.
    """

    def __init__(
        self,
        cfg,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info: WorkerInfo,
        record_metrics: bool = True,
    ):
        self.cfg = cfg
        self.device = torch.device(Worker.torch_device_type or "cpu")

        self.seed = cfg.seed + seed_offset
        self.total_num_processes = total_num_processes
        self.num_envs = num_envs
        self.worker_info = worker_info
        self.record_metrics = record_metrics

        self.auto_reset = getattr(cfg, "auto_reset", True)
        self.ignore_terminations = getattr(cfg, "ignore_terminations", False)
        self.use_rel_reward = getattr(cfg, "use_rel_reward", False)

        self._is_start = True
        self.elapsed_steps = 0

        self.video_cfg = cfg.video_cfg

        self.prev_step_reward = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.episode_generations = torch.zeros(
            self.num_envs, dtype=torch.int64, device=self.device
        )

        # Whether to use KIR Trick
        self.enable_kir = cfg.get("enable_kir", True)

        self.dataset = self._build_dataset(cfg)

        if self.record_metrics:
            self._init_metrics()

    @property
    def info_logging_keys(self):
        return []

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    @property
    def elapsed_steps(self):
        if not hasattr(self, "_elapsed_steps"):
            self._elapsed_steps = 0
        return self._elapsed_steps

    @elapsed_steps.setter
    def elapsed_steps(self, value):
        self._elapsed_steps = value

    @abstractmethod
    def _build_dataset(self, cfg):
        """Return the dataset wrapper used for resets."""

    @abstractmethod
    def chunk_step(self, actions):
        """Advance the environment by one chunk and return (obs, reward, done, info)."""

    @abstractmethod
    def reset(self):
        """Reset the environment and return initial observations."""

    @abstractmethod
    def step(self, actions):
        """Perform a single action step and return (obs, reward, done, info)."""

    def _get_runtime_device_str(self) -> str:
        if Worker.torch_device_type is not None:
            device_index = 0 if self.device.index is None else self.device.index
            return f"{Worker.torch_device_type}:{device_index}"
        return self.device.type

    @staticmethod
    def _clear_accelerator_cache() -> None:
        Worker.torch_platform.empty_cache()

    def _elastic_residency_modules(self) -> tuple[Any, ...]:
        """Return model modules whose device residency this environment owns."""

        return ()

    def verify_elastic_residency(self, *, resident: bool) -> None:
        """Verify model residency and CPU safety for elastic pause/resume."""

        if not hasattr(self, "_is_offloaded"):
            raise NotImplementedError(
                "world environment does not expose an offload residency state"
            )
        if bool(self._is_offloaded) == resident:
            raise RuntimeError("world environment offload state does not match receipt")
        modules = self._elastic_residency_modules()
        if not modules:
            raise NotImplementedError(
                "world environment does not declare elastic residency modules"
            )
        tensor_count = 0
        for module in modules:
            for tensors in (module.parameters(), module.buffers()):
                for tensor in tensors:
                    tensor_count += 1
                    if resident and tensor.device.type == "cpu":
                        raise RuntimeError(
                            "world-environment model remained on CPU after onload"
                        )
                    if not resident and tensor.device.type != "cpu":
                        raise RuntimeError(
                            "world-environment model remained on device after offload"
                        )
        if tensor_count == 0:
            raise RuntimeError("world-environment residency has no tensors to verify")
        if not resident:
            self.assert_cpu_only(
                (
                    getattr(self, "current_obs", None),
                    getattr(self, "image_queue", None),
                    getattr(self, "condition_action", None),
                    self.prev_step_reward,
                    self.reset_state_ids,
                    getattr(self, "success_once", None),
                    getattr(self, "returns", None),
                ),
                "world_environment_continuation",
            )

    def _environment_type(self) -> str:
        return type(self).__name__

    def _continuation_config(self) -> dict[str, Any]:
        """Return continuation-relevant resolved config values.

        Subclasses should override this to keep the fingerprint precise. The
        base default is intentionally conservative for fake/minimal env tests.
        """

        return {
            "environment_type": self._environment_type(),
            "num_envs": self.num_envs,
            "auto_reset": self.auto_reset,
            "ignore_terminations": self.ignore_terminations,
            "use_rel_reward": self.use_rel_reward,
            "record_metrics": self.record_metrics,
        }

    def _config_fingerprint(self) -> str:
        payload = self._to_jsonable(self._continuation_config())
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def _to_jsonable(cls, value: Any) -> Any:
        if OmegaConf.is_config(value):
            return cls._to_jsonable(OmegaConf.to_container(value, resolve=True))
        if isinstance(value, Mapping):
            return {str(key): cls._to_jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._to_jsonable(item) for item in value]
        if isinstance(value, torch.dtype):
            return str(value)
        if isinstance(value, torch.device):
            return str(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _commit_episode_reset(
        self, env_indices: torch.Tensor | np.ndarray | list[int] | int | None = None
    ) -> None:
        if env_indices is None:
            self.episode_generations += 1
            return
        if not isinstance(env_indices, torch.Tensor):
            env_indices = torch.as_tensor(env_indices, dtype=torch.long)
        self.episode_generations[env_indices.to(self.episode_generations.device)] += 1

    def _snapshot_image_queue(self) -> tuple[tuple[Any, ...], ...]:
        queues = getattr(self, "image_queue", [])
        return tuple(
            tuple(clone_nested_to_cpu(frame) for frame in queue) for queue in queues
        )

    def _snapshot_condition_action(self) -> torch.Tensor | None:
        return None

    def _snapshot_diffusion_state(self) -> tuple[torch.Tensor | None, int | None]:
        return None, None

    def _validate_model_resume_state(self, state: WorldEnvResumeState) -> None:
        return

    def _prepare_model_resume_state(self, state: WorldEnvResumeState) -> Any:
        return None

    def _commit_model_resume_state(self, prepared: _PreparedWorldEnvState) -> None:
        return

    def snapshot_resume_state(
        self, context: WorldEnvSnapshotContext
    ) -> WorldEnvResumeState:
        self._validate_context(context)
        diffusion_generator_state, diffusion_seed = self._snapshot_diffusion_state()
        state = WorldEnvResumeState(
            schema_version=WORLD_ENV_RESUME_SCHEMA_VERSION,
            environment_type=self._environment_type(),
            config_fingerprint=self._config_fingerprint(),
            worker_rank=context.worker_rank,
            worker_world_size=context.worker_world_size,
            stage_id=context.stage_id,
            lifecycle_generation=context.lifecycle_generation,
            chunk_index=context.chunk_index,
            next_transition_id=context.next_transition_id,
            episode_generations=clone_nested_to_cpu(self.episode_generations),
            reset_state_ids=clone_nested_to_cpu(self.reset_state_ids),
            reset_generator_state=clone_nested_to_cpu(self._generator.get_state()),
            diffusion_generator_state=clone_nested_to_cpu(diffusion_generator_state),
            diffusion_seed=diffusion_seed,
            current_obs=clone_nested_to_cpu(getattr(self, "current_obs", None)),
            image_queue=self._snapshot_image_queue(),
            condition_action=clone_nested_to_cpu(self._snapshot_condition_action()),
            task_descriptions=tuple(str(item) for item in self.task_descriptions),
            init_ee_poses=tuple(
                clone_nested_to_cpu(item) for item in self.init_ee_poses
            ),
            elapsed_steps=int(self.elapsed_steps),
            prev_step_reward=clone_nested_to_cpu(self.prev_step_reward),
            success_once=(
                clone_nested_to_cpu(self.success_once) if self.record_metrics else None
            ),
            returns=clone_nested_to_cpu(self.returns) if self.record_metrics else None,
            is_start=bool(self.is_start),
        )
        self.assert_cpu_only(state, "resume_state")
        return state

    def validate_resume_state(
        self,
        state: WorldEnvResumeState,
        expected: WorldEnvSnapshotContext,
    ) -> None:
        if not isinstance(state, WorldEnvResumeState):
            raise TypeError("resume state must be a WorldEnvResumeState")
        self._validate_context(expected)
        self.assert_cpu_only(state, "resume_state")
        self._expect_equal(
            "schema_version", state.schema_version, WORLD_ENV_RESUME_SCHEMA_VERSION
        )
        self._expect_equal(
            "environment_type", state.environment_type, self._environment_type()
        )
        self._expect_equal(
            "config_fingerprint", state.config_fingerprint, self._config_fingerprint()
        )
        self._expect_equal("worker_rank", state.worker_rank, expected.worker_rank)
        self._expect_equal(
            "worker_world_size", state.worker_world_size, expected.worker_world_size
        )
        self._expect_equal("stage_id", state.stage_id, expected.stage_id)
        self._expect_equal(
            "lifecycle_generation",
            state.lifecycle_generation,
            expected.lifecycle_generation,
        )
        if state.chunk_index < 0:
            raise ValueError("chunk_index must be non-negative")
        if state.next_transition_id < 0:
            raise ValueError("next_transition_id must be non-negative")
        self._expect_equal("chunk_index", state.chunk_index, expected.chunk_index)
        self._expect_equal(
            "next_transition_id",
            state.next_transition_id,
            expected.next_transition_id,
        )
        self._validate_int64_vector("episode_generations", state.episode_generations)
        self._validate_int64_vector("reset_state_ids", state.reset_state_ids)
        torch.testing.assert_close(
            state.episode_generations,
            expected.episode_generations.cpu(),
            msg="episode_generations do not match expected context",
        )
        torch.testing.assert_close(
            state.reset_state_ids,
            expected.reset_state_ids.cpu(),
            msg="reset_state_ids do not match expected context",
        )
        self._validate_uint8_tensor(
            "reset_generator_state", state.reset_generator_state
        )
        if state.diffusion_generator_state is not None:
            self._validate_uint8_tensor(
                "diffusion_generator_state", state.diffusion_generator_state
            )
        if len(state.task_descriptions) != self.num_envs:
            raise ValueError("task_descriptions length must match num_envs")
        if len(state.init_ee_poses) != self.num_envs:
            raise ValueError("init_ee_poses length must match num_envs")
        self._validate_metric_tensor(
            "prev_step_reward", state.prev_step_reward, torch.float32
        )
        if self.record_metrics:
            if state.success_once is None or state.returns is None:
                raise ValueError(
                    "record_metrics state requires success_once and returns"
                )
            self._validate_metric_tensor("success_once", state.success_once, torch.bool)
            self._validate_metric_tensor("returns", state.returns, torch.float32)
        elif state.success_once is not None or state.returns is not None:
            raise ValueError("metrics are present but record_metrics is disabled")
        self._validate_model_resume_state(state)

    def prepare_resume_state(
        self,
        state: WorldEnvResumeState,
        expected: WorldEnvSnapshotContext,
    ) -> _PreparedWorldEnvState:
        self.validate_resume_state(state, expected)
        return _PreparedWorldEnvState(
            current_obs=recursive_to_device(
                clone_nested_to_cpu(state.current_obs), self.device
            ),
            image_queue=tuple(
                tuple(
                    recursive_to_device(clone_nested_to_cpu(frame), self.device)
                    for frame in queue
                )
                for queue in state.image_queue
            ),
            condition_action=(
                state.condition_action.to(self.device).contiguous()
                if state.condition_action is not None
                else None
            ),
            task_descriptions=list(state.task_descriptions),
            init_ee_poses=list(clone_nested_to_cpu(state.init_ee_poses)),
            elapsed_steps=int(state.elapsed_steps),
            prev_step_reward=state.prev_step_reward.to(self.device).contiguous(),
            success_once=(
                state.success_once.to(self.device).contiguous()
                if state.success_once is not None
                else None
            ),
            returns=(
                state.returns.to(self.device).contiguous()
                if state.returns is not None
                else None
            ),
            is_start=bool(state.is_start),
            reset_state_ids=state.reset_state_ids.to(self.device).contiguous(),
            episode_generations=state.episode_generations.to(self.device).contiguous(),
            reset_generator_state=state.reset_generator_state.clone(),
            diffusion_generator_state=(
                state.diffusion_generator_state.clone()
                if state.diffusion_generator_state is not None
                else None
            ),
            diffusion_seed=state.diffusion_seed,
            model_state=self._prepare_model_resume_state(state),
        )

    def commit_resume_state(self, prepared: _PreparedWorldEnvState) -> None:
        self.current_obs = prepared.current_obs
        self.image_queue = [
            deque(queue, maxlen=getattr(self, "z_condition_frame_length", len(queue)))
            for queue in prepared.image_queue
        ]
        self.condition_action = prepared.condition_action
        self.task_descriptions = prepared.task_descriptions
        self.init_ee_poses = prepared.init_ee_poses
        self.elapsed_steps = prepared.elapsed_steps
        self.prev_step_reward = prepared.prev_step_reward
        self._is_start = prepared.is_start
        self.reset_state_ids = prepared.reset_state_ids
        self.episode_generations = prepared.episode_generations
        self._generator.set_state(prepared.reset_generator_state)
        if self.record_metrics:
            self.success_once = prepared.success_once
            self.returns = prepared.returns
        self._commit_model_resume_state(prepared)

    def _validate_context(self, context: WorldEnvSnapshotContext) -> None:
        self._expect_equal(
            "context.worker_rank", context.worker_rank, self.worker_info.rank
        )
        self._expect_equal(
            "context.worker_world_size",
            context.worker_world_size,
            self.worker_info.group_world_size,
        )
        if context.stage_id < 0:
            raise ValueError("context.stage_id must be non-negative")
        if context.lifecycle_generation < 0:
            raise ValueError("context.lifecycle_generation must be non-negative")
        self._validate_int64_vector(
            "context.episode_generations", context.episode_generations
        )
        self._validate_int64_vector("context.reset_state_ids", context.reset_state_ids)
        torch.testing.assert_close(
            context.episode_generations.cpu(),
            self.episode_generations.detach().cpu(),
            msg="context episode_generations do not match live environment",
        )
        torch.testing.assert_close(
            context.reset_state_ids.cpu(),
            self.reset_state_ids.detach().cpu(),
            msg="context reset_state_ids do not match live environment",
        )

    def _validate_int64_vector(self, name: str, value: torch.Tensor) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.device.type != "cpu":
            raise ValueError(f"{name} must be on CPU")
        if value.dtype != torch.int64:
            raise ValueError(f"{name} must have dtype torch.int64")
        if tuple(value.shape) != (self.num_envs,):
            raise ValueError(f"{name} must have shape ({self.num_envs},)")

    @staticmethod
    def _validate_uint8_tensor(name: str, value: torch.Tensor) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.device.type != "cpu":
            raise ValueError(f"{name} must be on CPU")
        if value.dtype != torch.uint8:
            raise ValueError(f"{name} must have dtype torch.uint8")

    def _validate_metric_tensor(
        self, name: str, value: torch.Tensor, dtype: torch.dtype
    ) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.device.type != "cpu":
            raise ValueError(f"{name} must be on CPU")
        if value.dtype != dtype:
            raise ValueError(f"{name} must have dtype {dtype}")
        if tuple(value.shape) != (self.num_envs,):
            raise ValueError(f"{name} must have shape ({self.num_envs},)")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite values")

    @staticmethod
    def _expect_equal(name: str, actual: Any, expected: Any) -> None:
        if actual != expected:
            raise ValueError(f"{name} mismatch: expected {expected!r}, got {actual!r}")

    @classmethod
    def assert_cpu_only(cls, value: Any, path: str = "value") -> None:
        if isinstance(value, torch.Tensor):
            if value.device.type != "cpu":
                raise ValueError(f"{path} must be on CPU, got {value.device}")
            return
        if isinstance(value, np.ndarray):
            return
        if is_dataclass(value) and not isinstance(value, type):
            for field in fields(value):
                cls.assert_cpu_only(getattr(value, field.name), f"{path}.{field.name}")
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                cls.assert_cpu_only(item, f"{path}[{key!r}]")
            return
        if isinstance(value, (list, tuple, deque)):
            for idx, item in enumerate(value):
                cls.assert_cpu_only(item, f"{path}[{idx}]")
            return
        if value is None or isinstance(value, (str, int, float, bool, bytes)):
            return
        raise TypeError(f"{path} has unsupported snapshot type {type(value).__name__}")

    def _init_metrics(self):
        """Initialize episode metrics tensors."""
        self.success_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.returns = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )

    def _reset_metrics(self, env_idx: Optional[Union[int, torch.Tensor]] = None):
        """Reset metrics either globally or for targeted environments."""
        if env_idx is not None:
            mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            if self.record_metrics:
                self.success_once[mask] = False
                self.returns[mask] = 0
        else:
            self.prev_step_reward[:] = 0
            if self.record_metrics:
                self.success_once[:] = False
                self.returns[:] = 0.0
        self.elapsed_steps = 0

    def _record_metrics(self, step_reward, terminations, infos):
        """Store episode metrics inside the info dict."""
        if not self.record_metrics:
            return infos

        # Update success_once based on terminations
        if isinstance(terminations, torch.Tensor):
            self.success_once = self.success_once | terminations
        else:
            terminations_tensor = torch.tensor(
                terminations, device=self.device, dtype=torch.bool
            )
            self.success_once = self.success_once | terminations_tensor

        episode_info = {}
        self.returns += step_reward
        episode_info["return"] = self.returns.clone()
        infos["episode"] = episode_info
        return infos

    def update_reset_state_ids(self):
        """Optional hook to manage reset states for subclasses."""
        return
