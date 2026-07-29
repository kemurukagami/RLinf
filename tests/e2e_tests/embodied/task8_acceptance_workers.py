"""Acceptance-only worker subclasses that preserve production computation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from task8_acceptance_support import logical_tensor_bytes, normalize_manifest

from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


@dataclass(frozen=True, slots=True)
class AcceptanceWorkerEvent:
    """One worker-boundary observation awaiting driver identity enrichment."""

    event: str
    details: Mapping[str, Any]


AcceptanceWorkerObserver = Callable[[AcceptanceWorkerEvent], None]


class AcceptanceWorkerRecorderMixin:
    """Add an explicitly configured fail-closed acceptance observer."""

    _task8_acceptance_observer: AcceptanceWorkerObserver | None = None

    def configure_acceptance_observer(self, observer: AcceptanceWorkerObserver) -> None:
        """Install the observer before running an instrumented acceptance case."""
        if not callable(observer):
            raise TypeError("acceptance observer must be callable")
        self._task8_acceptance_observer = observer

    def _record_acceptance(self, event: str, **details: Any) -> None:
        observer = self._task8_acceptance_observer
        if observer is None:
            raise RuntimeError("acceptance observer is not configured")
        self._add_elastic_acceptance_context(details)
        observer(
            AcceptanceWorkerEvent(
                event=event,
                details=normalize_manifest(details),
            )
        )

    def _add_elastic_acceptance_context(self, details: dict[str, Any]) -> None:
        """Add rank-local elastic context used by acceptance event validation."""
        cursor = getattr(self, "_elastic_cursor", None)
        if cursor is None:
            cursor = getattr(self, "_rollout_cursor", None)
        if cursor is None:
            return
        lifecycle_generation = getattr(cursor, "lifecycle_generation", None)
        if lifecycle_generation is not None:
            details.setdefault("lifecycle_generation", lifecycle_generation)
        policy_version = getattr(cursor, "policy_version", None)
        if (
            isinstance(policy_version, int)
            and not isinstance(policy_version, bool)
            and policy_version >= 0
        ):
            details.setdefault("policy_version", policy_version)
        transition_id = getattr(cursor, "expected_transition_id", None)
        if transition_id is None:
            expected_transition = getattr(self, "_elastic_expected_transition_id", None)
            if callable(expected_transition):
                transition_id = expected_transition()
        if transition_id is not None:
            details.setdefault("transition_id", transition_id)


class RecordingEnvWorkerMixin(AcceptanceWorkerRecorderMixin):
    """Bracket environment computation and elastic residency transitions."""

    _task8_phase_diagnostics_enabled = False

    def configure_task8_phase_diagnostics(self, enabled: bool) -> None:
        """Enable or disable acceptance-only fine-grained phase diagnostics."""
        if not isinstance(enabled, bool):
            raise TypeError("Task 8 phase diagnostics flag must be a bool")
        self._task8_phase_diagnostics_enabled = enabled

    def _run_world_model_diagnostic_phase(
        self,
        phase: str,
        operation: Callable[..., Any],
        *args: Any,
        stage_id: int,
        **kwargs: Any,
    ) -> Any:
        """Record one acceptance-only world-model phase without changing output."""
        started_ns = time.monotonic_ns()
        self._record_acceptance(f"world_model_{phase}_started", stage_id=stage_id)
        try:
            result = operation(*args, **kwargs)
        except BaseException as exc:
            self._record_acceptance(
                f"world_model_{phase}_failed",
                stage_id=stage_id,
                elapsed_seconds=(time.monotonic_ns() - started_ns) / 1_000_000_000,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        self._record_acceptance(
            f"world_model_{phase}_completed",
            stage_id=stage_id,
            elapsed_seconds=(time.monotonic_ns() - started_ns) / 1_000_000_000,
        )
        return result

    def _install_world_model_diagnostic_wrappers(
        self, env: Any, *, stage_id: int
    ) -> Callable[[], None]:
        """Temporarily bracket Wan execution phases on one concrete environment."""
        if not self._task8_phase_diagnostics_enabled:
            return lambda: None

        phase_methods = {
            "onload": "onload",
            "_infer_next_chunk_frames": "diffusion",
            "_infer_next_chunk_rewards": "reward",
        }
        sentinel = object()
        previous_local_values: dict[str, Any] = {}
        for method_name, phase in phase_methods.items():
            operation = getattr(env, method_name, None)
            if not callable(operation):
                continue
            env_dict = getattr(env, "__dict__", {})
            previous_local_values[method_name] = env_dict.get(method_name, sentinel)

            def diagnostic_wrapper(
                *args: Any,
                _operation: Callable[..., Any] = operation,
                _phase: str = phase,
                **kwargs: Any,
            ) -> Any:
                return self._run_world_model_diagnostic_phase(
                    _phase,
                    _operation,
                    *args,
                    stage_id=stage_id,
                    **kwargs,
                )

            setattr(env, method_name, diagnostic_wrapper)

        set_chunk_observer = getattr(env, "set_chunk_step_diagnostic_observer", None)
        previous_chunk_observer = getattr(env, "_chunk_step_diagnostic_observer", None)
        if callable(set_chunk_observer):

            def chunk_observer(event: str, details: Mapping[str, Any]) -> None:
                self._record_acceptance(
                    f"world_model_{event}", stage_id=stage_id, **details
                )

            set_chunk_observer(chunk_observer)

        def restore() -> None:
            for method_name, previous in previous_local_values.items():
                if previous is sentinel:
                    delattr(env, method_name)
                else:
                    setattr(env, method_name, previous)
            if callable(set_chunk_observer):
                set_chunk_observer(previous_chunk_observer)

        return restore

    def env_interact_step(self, chunk_actions: Any, stage_id: int) -> Any:
        self._record_acceptance(
            "chunk_started", stage_id=stage_id, chunk_actions=chunk_actions
        )

        def restore() -> None:
            return None

        if self._task8_phase_diagnostics_enabled:
            env = self.env_list[stage_id]
            restore = self._install_world_model_diagnostic_wrappers(
                env, stage_id=stage_id
            )
        try:
            result = super().env_interact_step(chunk_actions, stage_id)
        finally:
            restore()
        if self._task8_phase_diagnostics_enabled:
            self._record_acceptance("env_output_constructed", stage_id=stage_id)
        # Keep the commit marker bounded and synchronization-free. The live
        # result contains accelerator tensors; detailed numerical evidence is
        # captured later from deliberately CPU-sealed artifacts.
        self._record_acceptance("chunk_committed", stage_id=stage_id)
        return result

    def offload_elastic_environment(self, token: Any) -> Any:
        self._record_acceptance("environment_offload_started", token=token)
        receipt = super().offload_elastic_environment(token)
        self._record_acceptance("environment_offload_verified", receipt=receipt)
        return receipt

    def snapshot_rollout_stage(self) -> Any:
        self._record_acceptance("snapshot_started")
        state = super().snapshot_rollout_stage()
        self._record_acceptance(
            "snapshot_completed",
            state=state,
            encoded_tensor_bytes=logical_tensor_bytes(state),
        )
        return state

    def restore_rollout_stage(self, state: Any, **kwargs: Any) -> Any:
        result = super().restore_rollout_stage(state, **kwargs)
        self._record_acceptance("restore_validated", state=state, arguments=kwargs)
        self._record_acceptance("restore_committed", state=state)
        return result

    def prepare_elastic_resume(self, token: Any) -> Any:
        self._record_acceptance("environment_onload_started", token=token)
        receipt = super().prepare_elastic_resume(token)
        self._task8_resume_dispatch_pending = True
        self._record_acceptance("environment_onload_verified", receipt=receipt)
        return receipt

    async def request_elastic_drain(self, request: Any) -> Any:
        status = await super().request_elastic_drain(request)
        self._record_acceptance("drain_requested", request=request, status=status)
        return status

    async def _send_elastic_barrier(self, rollout_channel: Any, token: Any) -> None:
        diagnostics_enabled = self._task8_phase_diagnostics_enabled
        if diagnostics_enabled:
            self._record_acceptance("barrier_send_started", token=token)
        try:
            await super()._send_elastic_barrier(rollout_channel, token)
        except BaseException as exc:
            if diagnostics_enabled:
                self._record_acceptance(
                    "barrier_send_failed",
                    token=token,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
            raise
        if diagnostics_enabled:
            self._record_acceptance("barrier_send_completed", token=token)
        self._record_acceptance("drain_observed", token=token)

    async def _send_elastic_observation(
        self, rollout_channel: Any, env_output: Any
    ) -> None:
        resumed = getattr(self, "_task8_resume_dispatch_pending", False)
        await super()._send_elastic_observation(rollout_channel, env_output)
        # The awaited production send is the earliest reliable indication that
        # this rank has produced a valid observation and handed it to rollout.
        # Recording before the await would allow the harness to introduce
        # competing demand while bootstrap was still blocked or had failed.
        self._record_acceptance("bootstrap_dispatched", env_output=env_output)
        if resumed:
            self._record_acceptance(
                "resumed_bootstrap_dispatched", env_output=env_output
            )
            self._task8_resume_dispatch_pending = False


class RecordingMultiStepRolloutWorkerMixin(AcceptanceWorkerRecorderMixin):
    """Bracket real policy calls and rollout model residency transitions."""

    def _build_train_rollout_result(self, env_output: Any, **kwargs: Any) -> Any:
        self._record_acceptance(
            "policy_request_started", env_output=env_output, arguments=kwargs
        )
        result = super()._build_train_rollout_result(env_output, **kwargs)
        self._record_acceptance("policy_request_completed", result=result)
        return result

    def offload_elastic_rollout(self, token: Any) -> Any:
        self._record_acceptance("rollout_offload_started", token=token)
        receipt = super().offload_elastic_rollout(token)
        self._record_acceptance("rollout_offload_verified", receipt=receipt)
        return receipt

    def prepare_elastic_resume(self, token: Any) -> Any:
        self._record_acceptance("rollout_onload_started", token=token)
        receipt = super().prepare_elastic_resume(token)
        self._record_acceptance("rollout_onload_verified", receipt=receipt)
        return receipt

    async def request_elastic_drain(self, request: Any) -> Any:
        status = await super().request_elastic_drain(request)
        self._record_acceptance("drain_requested", request=request, status=status)
        return status

    async def generate_until_pause_or_complete(self, *args: Any, **kwargs: Any) -> Any:
        result = await super().generate_until_pause_or_complete(*args, **kwargs)
        outcome = getattr(getattr(result, "outcome", None), "value", None)
        if outcome == "pause_ready":
            self._record_acceptance("barrier_consumed", result=result)
        elif outcome == "completed":
            self._record_acceptance("rank_completed", result=result)
        return result


class RecordingEmbodiedFSDPActorMixin(AcceptanceWorkerRecorderMixin):
    """Capture the sealed CPU batch and exact GRPO computation boundaries."""

    @staticmethod
    def _receipt_policy_version(receipt: Any) -> int | None:
        if isinstance(receipt, Mapping):
            return receipt.get("policy_version")
        return getattr(receipt, "policy_version", None)

    def seal_rlix_batch(self, **kwargs: Any) -> Any:
        receipt = super().seal_rlix_batch(**kwargs)
        batch = self.rollout_batch
        self._record_acceptance(
            "batch_sealed",
            receipt=receipt,
            transition_ids=getattr(self, "_rlix_received_transition_ids", ()),
            batch=batch,
            batch_tensor_bytes=logical_tensor_bytes(batch),
        )
        return receipt

    def compute_advantages_and_returns(self) -> Any:
        result = super().compute_advantages_and_returns()
        receipt = getattr(self, "_rlix_batch_receipt", None)
        self._record_acceptance(
            "advantages_computed",
            advantage_type=self.cfg.algorithm.adv_type,
            policy_version=self._receipt_policy_version(receipt),
            metrics=result,
        )
        return result

    def run_rlix_training(self, batch_receipt: Any) -> Any:
        policy_version = self._receipt_policy_version(batch_receipt)
        self._record_acceptance(
            "training_started",
            advantage_type=self.cfg.algorithm.adv_type,
            policy_version=policy_version,
            batch_receipt=batch_receipt,
        )
        result = super().run_rlix_training(batch_receipt)
        self._record_acceptance(
            "training_completed",
            advantage_type=self.cfg.algorithm.adv_type,
            policy_version=policy_version,
            metrics=result,
        )
        return result


class RecordingEmbodiedRunnerMixin(AcceptanceWorkerRecorderMixin):
    """Record policy sync, sealed collection, and actor update boundaries."""

    def update_rollout_weights(self) -> Any:
        policy_version = self.global_step
        self._record_acceptance(
            "stage_acquired", stage="policy_sync", policy_version=policy_version
        )
        result = super().update_rollout_weights()
        self._record_acceptance("policy_synchronized", policy_version=policy_version)
        self._record_acceptance(
            "stage_released", stage="policy_sync", policy_version=policy_version
        )
        return result

    def _collect_rlix_rollouts(self) -> Any:
        policy_version = self.global_step
        self._record_acceptance("reward_started", policy_version=policy_version)
        self._record_acceptance("generation_requested", policy_version=policy_version)
        if self.rlix_runtime is None:
            receipt = super()._collect_rlix_rollouts()
            self._record_acceptance("reward_completed", receipt=receipt)
            self._record_acceptance("batch_sealed", receipt=receipt)
            return receipt

        canonical_ranks = tuple(
            rank for rank, _ in self.rlix_runtime.placement_plan.actor_infer_bundles
        )
        total_num_envs = int(self.cfg.env.train.total_num_envs)
        if total_num_envs % len(canonical_ranks) != 0:
            raise ValueError("training environments do not divide across RLix ranks")
        assigned_per_rank = (
            total_num_envs
            // len(canonical_ranks)
            * int(self.cfg.env.train.rollout_epoch)
        )
        assignments = dict.fromkeys(canonical_ranks, assigned_per_rank)
        reward_handle = None
        if self.reward is not None:
            reward_handle = self.reward.compute_rewards(
                input_channel=self.reward_channel,
                output_channel=self.env_channel,
            )
        session = self.rlix_runtime.begin_collection(
            policy_version=self.global_step,
            assigned_trajectories_by_rank=assignments,
            env_input_channel=self.env_channel,
            rollout_request_channel=self.rollout_channel,
            reward_channel=self.reward_channel,
            actor_channel=self.actor_channel,
            actor_receiver_start=lambda: self.actor.recv_rollout_trajectories(
                input_channel=self.actor_channel
            ),
        )
        self._record_acceptance(
            "generation_granted",
            policy_version=policy_version,
            active_dp_ranks=tuple(sorted(session.active_dp_ranks)),
        )
        poll_interval_s = float(self.cfg.rlix.get("monitor_poll_interval_s", 0.01))
        self.rlix_runtime.wait_for_collection(
            session,
            poll_interval_s=poll_interval_s,
        )
        if reward_handle is not None:
            reward_handle.wait()
        self._last_rlix_reward_handle = reward_handle
        receipt = self.rlix_runtime.seal_collection(
            session,
            actor_seal_start=lambda expected: self.actor.seal_rlix_batch(
                lifecycle_generation=session.context.lifecycle_generation,
                policy_version=session.context.policy_version,
                contributing_dp_ranks=session.context.dp_ranks,
                expected_trajectories=expected,
            ),
        )
        self._last_rlix_collection_session = session
        self._record_acceptance("reward_completed", receipt=receipt)
        self._record_acceptance("batch_sealed", receipt=receipt)
        return receipt


class RecordingEnvWorker(RecordingEnvWorkerMixin, EnvWorker):
    """Production EnvWorker with acceptance-only transparent recording."""


class RecordingMultiStepRolloutWorker(
    RecordingMultiStepRolloutWorkerMixin, MultiStepRolloutWorker
):
    """Production rollout worker with acceptance-only transparent recording."""


class RecordingEmbodiedFSDPActor(RecordingEmbodiedFSDPActorMixin, EmbodiedFSDPActor):
    """Production embodied actor with acceptance-only batch recording."""


class RecordingEmbodiedRunner(RecordingEmbodiedRunnerMixin, EmbodiedRunner):
    """Production EmbodiedRunner with acceptance-only stage recording."""
