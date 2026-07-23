"""Acceptance-only worker subclasses that preserve production computation."""

from __future__ import annotations

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
        observer(
            AcceptanceWorkerEvent(
                event=event,
                details=normalize_manifest(details),
            )
        )


class RecordingEnvWorkerMixin(AcceptanceWorkerRecorderMixin):
    """Bracket environment computation and elastic residency transitions."""

    def env_interact_step(self, chunk_actions: Any, stage_id: int) -> Any:
        self._record_acceptance(
            "chunk_started", stage_id=stage_id, chunk_actions=chunk_actions
        )
        result = super().env_interact_step(chunk_actions, stage_id)
        self._record_acceptance("chunk_committed", stage_id=stage_id, result=result)
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
        await super()._send_elastic_barrier(rollout_channel, token)
        self._record_acceptance("drain_observed", token=token)

    async def _send_elastic_observation(
        self, rollout_channel: Any, env_output: Any
    ) -> None:
        resumed = getattr(self, "_task8_resume_dispatch_pending", False)
        await super()._send_elastic_observation(rollout_channel, env_output)
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
        self._record_acceptance(
            "advantages_computed",
            advantage_type=self.cfg.algorithm.adv_type,
            metrics=result,
        )
        return result

    def run_rlix_training(self, batch_receipt: Any) -> Any:
        self._record_acceptance(
            "training_started",
            advantage_type=self.cfg.algorithm.adv_type,
            batch_receipt=batch_receipt,
        )
        result = super().run_rlix_training(batch_receipt)
        self._record_acceptance(
            "training_completed",
            advantage_type=self.cfg.algorithm.adv_type,
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
        receipt = super()._collect_rlix_rollouts()
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
