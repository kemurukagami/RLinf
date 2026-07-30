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

import logging
import os
import queue
import threading
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Mapping, Union

from omegaconf.dictconfig import DictConfig

from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.logging import get_logger
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.metric_utils import compute_evaluate_metrics, print_metrics_table
from rlinf.utils.runner_utils import check_progress
from rlinf.utils.timers import Timer

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from rlinf.workers.actor.async_fsdp_sac_policy_worker import (
        AsyncEmbodiedSACFSDPPolicy,
    )
    from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
    from rlinf.workers.actor.fsdp_nft_policy_worker import EmbodiedNFTFSDPPolicy
    from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy
    from rlinf.workers.env.async_env_worker import AsyncEnvWorker
    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.reward.reward_worker import EmbodiedRewardWorker
    from rlinf.workers.rollout.hf.async_huggingface_worker import (
        AsyncMultiStepRolloutWorker,
    )
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


def _resolve_channel_names(
    channel_names: Mapping[str, str] | None, *, use_reward: bool
) -> dict[str, str]:
    """Validate optional process-owned channel names while preserving defaults."""
    defaults = {"env": "Env", "rollout": "Rollout", "actor": "Actor"}
    if use_reward:
        defaults["reward"] = "Reward"
    if channel_names is None:
        return defaults
    resolved = dict(channel_names)
    if set(resolved) != set(defaults):
        raise ValueError(
            f"channel_names must contain exactly {sorted(defaults)!r}, "
            f"got {sorted(resolved)!r}"
        )
    if any(
        not isinstance(value, str) or not value or any(char.isspace() for char in value)
        for value in resolved.values()
    ):
        raise ValueError("channel names must be non-empty strings without whitespace")
    if len(set(resolved.values())) != len(resolved):
        raise ValueError("channel names must be unique")
    return resolved


class EmbodiedRunner:
    def __init__(
        self,
        cfg: DictConfig,
        actor: Union[
            "EmbodiedFSDPActor",
            "EmbodiedNFTFSDPPolicy",
            "EmbodiedSACFSDPPolicy",
            "AsyncEmbodiedSACFSDPPolicy",
        ],
        rollout: Union["MultiStepRolloutWorker", "AsyncMultiStepRolloutWorker"],
        env: Union["EnvWorker", "AsyncEnvWorker"],
        reward: Union["EmbodiedRewardWorker"] = None,
        critic=None,
        rlix_runtime=None,
        channel_names: Mapping[str, str] | None = None,
    ):
        self.cfg = cfg
        self.actor = actor
        self.rollout = rollout
        self.env = env
        self.critic = critic
        self.reward = reward
        self.rlix_runtime = rlix_runtime
        self.weight_sync_interval = self.cfg.runner.weight_sync_interval
        self.overlap_env_bootstrap = bool(
            self.cfg.runner.get("overlap_env_bootstrap", False)
        )

        # Step-gated profiling: ``cluster.profiling.steps`` lists the global step
        profiling_raw = self.cfg.cluster.get("profiling", None)
        profiling_enabled = profiling_raw is not None and bool(
            profiling_raw.get("enabled", True)
        )
        profile_steps_raw = (
            profiling_raw.get("steps", None) if profiling_enabled else None
        )
        self._profile_all_steps = profiling_enabled and profile_steps_raw is None
        self._profile_steps: set[int] | None = (
            {int(s) for s in profile_steps_raw}
            if profile_steps_raw is not None
            else None
        )

        # Data channels
        resolved_channel_names = _resolve_channel_names(
            channel_names, use_reward=self.reward is not None
        )
        self.env_channel = Channel.create(resolved_channel_names["env"])
        self.rollout_channel = Channel.create(resolved_channel_names["rollout"])
        self.actor_channel = Channel.create(resolved_channel_names["actor"])
        if self.reward is not None:
            self.reward_channel = Channel.create(resolved_channel_names["reward"])
        else:
            self.reward_channel = None

        # this timer checks if we should stop training
        self.run_timer = Timer(None)  # Timer that checks if we should stop training

        self.consumed_samples = 0
        # the step here is GRPO step
        self.global_step = 0

        # compute `max_steps`
        self.set_max_steps()

        self.timer = ScopedTimer(reduction="max", sync_cuda=False)

        self.logger = get_logger()
        self.metric_logger = MetricLogger(cfg)
        self.enable_per_worker_metric_log = bool(
            self.cfg.runner.get("per_worker_log", False)
        )

        # Async logging setup
        self.stop_logging = False
        self.log_queue = queue.Queue()
        self.log_thread = threading.Thread(target=self._log_worker, daemon=True)
        self.log_thread.start()

    def _log_worker(self):
        """Background thread for processing log messages."""
        while not self.stop_logging:
            try:
                # Wait for log message with timeout
                log_func, args = self.log_queue.get(timeout=0.1)
                log_func(*args)
                self.log_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Logging error: {e}")
                continue

    def print_metrics_table_async(
        self,
        step: int,
        total_steps: int,
        start_time: float,
        metrics: dict,
        start_step: int = 0,
    ):
        """Async version that puts table printing in queue."""
        self.log_queue.put(
            (
                print_metrics_table,
                (
                    step,
                    total_steps,
                    start_time,
                    metrics,
                    start_step,
                    self.metric_logger.log_path,
                ),
            )
        )

    def init_workers(self):
        if self.rlix_runtime is None:
            self._initialize_workers_and_resume()
            return

        from rlix_core.protocol.types import (
            INITIALIZATION_CLUSTER_NAME,
            Priority,
        )

        with self.rlix_runtime.fixed_stage(
            cluster_name=INITIALIZATION_CLUSTER_NAME,
            priority=Priority.INITIALIZATION,
            global_step=self.global_step,
        ) as stage:
            self._initialize_workers_and_resume()
            residencies = self._get_rlix_fixed_residencies(
                self.actor, self.rollout, self.env
            )
            stage.complete(
                self.rlix_runtime.fixed_residency_receipt(
                    cluster_name=INITIALIZATION_CLUSTER_NAME,
                    worker_residencies=residencies,
                    policy_version=self.global_step,
                )
            )

    def _initialize_workers_and_resume(self) -> None:
        """Initialize in legacy peak-memory order and restore actor state."""
        # create worker in order to decrease the maximum memory usage
        rollout_handle = self.rollout.init_worker()
        env_handle = self.env.init_worker()

        if self.reward is not None:
            self.reward.init_worker().wait()

        rollout_handle.wait()
        env_handle.wait()
        self.actor.init_worker().wait()

        resume_dir = self.cfg.runner.get("resume_dir", None)
        if resume_dir is None:
            return

        self.logger.info(f"Resuming training from checkpoint directory {resume_dir}.")
        actor_checkpoint_path = os.path.join(resume_dir, "actor")
        assert os.path.exists(actor_checkpoint_path), (
            f"resume_dir {actor_checkpoint_path} does not exist."
        )
        if self.rlix_runtime is None:
            self.actor.load_checkpoint(actor_checkpoint_path).wait()
        else:
            self.actor.load_rlix_checkpoint(actor_checkpoint_path).wait()
        self.global_step = int(resume_dir.split("global_step_")[-1])

    @staticmethod
    def _get_rlix_fixed_residencies(*worker_groups) -> list:
        """Collect public per-rank residency observations from worker groups."""
        residencies = []
        for worker_group in worker_groups:
            results = worker_group.get_rlix_fixed_residency().wait()
            if not isinstance(results, list):
                raise TypeError("RLix worker residency query must return a list")
            residencies.extend(results)
        return residencies

    def update_rollout_weights(self):
        if self.rlix_runtime is None:
            self._sync_rollout_weights()
            return

        from rlix_core.protocol.types import POLICY_SYNC_CLUSTER_NAME

        with self.rlix_runtime.policy_sync_stage(
            expected_policy_version=self.global_step
        ) as stage:
            self._on_rlix_policy_sync_stage_acquired()
            self._sync_rollout_weights()
            residencies = self._get_rlix_fixed_residencies(self.actor, self.rollout)
            stage.complete(
                self.rlix_runtime.fixed_residency_receipt(
                    cluster_name=POLICY_SYNC_CLUSTER_NAME,
                    worker_residencies=residencies,
                    policy_version=self.global_step,
                )
            )

    def _on_rlix_policy_sync_stage_acquired(self) -> None:
        """Hook invoked only after fixed policy-sync ownership is acquired."""

    def _sync_rollout_weights(self) -> None:
        """Run the existing all-rank actor/rollout collective pair."""
        rollout_handle: Handle = self.rollout.sync_model_from_actor()
        actor_handle: Handle = self.actor.sync_model_to_rollout()
        actor_handle.wait()
        rollout_handle.wait()

    def _collect_rlix_rollouts(self):
        """Collect and seal one elastic, policy-version-consistent actor batch."""
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
        return receipt

    def _train_rlix_batch(self, batch_receipt):
        """Train one sealed batch under fixed all-rank actor ownership."""
        from rlix_core.protocol.types import ACTOR_TRAIN_CLUSTER_NAME, Priority

        consumed_policy_version = self.global_step
        produced_policy_version = consumed_policy_version + 1
        with self.rlix_runtime.fixed_stage(
            cluster_name=ACTOR_TRAIN_CLUSTER_NAME,
            priority=Priority.ACTOR_TRAINING,
            global_step=consumed_policy_version,
        ) as stage:
            actor_rollout_metrics = self.actor.compute_advantages_and_returns().wait()
            training_handle = self.actor.run_rlix_training(batch_receipt)
            actor_training_metrics = training_handle.wait()
            # Publishing the version is part of the successful training
            # transaction.  Policy sync must never depend on a caller remembering
            # to update the actor's authoritative source-version stamp later.
            self.actor.set_global_step(produced_policy_version).wait()
            residencies = self._get_rlix_fixed_residencies(self.actor)
            stage.complete(
                self.rlix_runtime.fixed_residency_receipt(
                    cluster_name=ACTOR_TRAIN_CLUSTER_NAME,
                    worker_residencies=residencies,
                    policy_version=produced_policy_version,
                )
            )
        self.global_step = produced_policy_version
        return actor_rollout_metrics, actor_training_metrics, training_handle

    def evaluate(self):
        if self.rlix_runtime is None:
            return self._evaluate_workers()

        from rlix_core.protocol.types import EVALUATION_CLUSTER_NAME, Priority

        with self.rlix_runtime.fixed_stage(
            cluster_name=EVALUATION_CLUSTER_NAME,
            priority=Priority.INITIALIZATION,
            global_step=self.global_step,
        ) as stage:
            eval_metrics = self._evaluate_workers()
            residencies = self._get_rlix_fixed_residencies(self.rollout, self.env)
            stage.complete(
                self.rlix_runtime.fixed_residency_receipt(
                    cluster_name=EVALUATION_CLUSTER_NAME,
                    worker_residencies=residencies,
                    policy_version=self.global_step,
                )
            )
        return eval_metrics

    def _evaluate_workers(self):
        """Run the existing paired rollout/environment evaluation."""
        env_handle: Handle = self.env.evaluate(
            input_channel=self.env_channel,
            rollout_channel=self.rollout_channel,
        )
        rollout_handle: Handle = self.rollout.evaluate(
            input_channel=self.rollout_channel,
            output_channel=self.env_channel,
        )
        env_results = env_handle.wait()
        rollout_handle.wait()
        eval_metrics_list = [results for results in env_results if results is not None]
        eval_metrics = compute_evaluate_metrics(eval_metrics_list)
        return eval_metrics

    def _log_ranked_metrics(
        self,
        metrics_list: list[dict] | None,
        step: int,
        prefix: str,
        worker_group_name: str,
        add_prefix: bool = True,
    ):
        if not self.enable_per_worker_metric_log or not metrics_list:
            return
        for rank, metrics in enumerate(metrics_list):
            if not metrics:
                continue
            metrics_to_log = (
                {f"{prefix}/{k}": v for k, v in metrics.items()}
                if add_prefix
                else metrics
            )
            self.metric_logger.log(
                data=metrics_to_log,
                step=step,
                worker_group_name=worker_group_name,
                rank=rank,
            )

    def _aggregate_numeric_metrics(self, metrics_list: list[dict] | None) -> dict:
        if not metrics_list:
            return {}
        merged_metrics = defaultdict(list)
        for metrics in metrics_list:
            if not metrics:
                continue
            for key, value in metrics.items():
                merged_metrics[key].append(value)
        return {
            key: (sum(values) / len(values))
            for key, values in merged_metrics.items()
            if values
        }

    def _process_ranked_numeric_results(
        self, results: list[dict], metric_field: str
    ) -> tuple[dict, list[dict]]:
        metric_list: list[dict] = []
        per_rank_metrics: dict[int, list[dict]] = defaultdict(list)
        for result in results:
            metrics = result.get(metric_field, None)
            if not metrics:
                continue
            metric_list.append(metrics)
            rank = result.get("rank", None)
            if rank is not None:
                per_rank_metrics[int(rank)].append(metrics)

        aggregated_metrics = self._aggregate_numeric_metrics(metric_list)
        ranked_metrics_list: list[dict] = []
        if per_rank_metrics:
            max_rank = max(per_rank_metrics.keys())
            ranked_metrics_list = [{} for _ in range(max_rank + 1)]
            for rank, metrics_list in per_rank_metrics.items():
                ranked_metrics_list[rank] = self._aggregate_numeric_metrics(
                    metrics_list
                )
        return aggregated_metrics, ranked_metrics_list

    def _process_ranked_eval_results(
        self, results: list[dict], metric_field: str
    ) -> tuple[dict, list[dict]]:
        metric_list: list[dict] = []
        per_rank_metrics: dict[int, list[dict]] = defaultdict(list)
        for result in results:
            metrics = result.get(metric_field, None)
            if not metrics:
                continue
            metric_list.append(metrics)
            rank = result.get("rank", None)
            if rank is not None:
                per_rank_metrics[int(rank)].append(metrics)

        aggregated_metrics = (
            compute_evaluate_metrics(metric_list) if metric_list else {}
        )
        ranked_metrics_list: list[dict] = []
        if per_rank_metrics:
            max_rank = max(per_rank_metrics.keys())
            ranked_metrics_list = [{} for _ in range(max_rank + 1)]
            for rank, metrics_list in per_rank_metrics.items():
                ranked_metrics_list[rank] = compute_evaluate_metrics(metrics_list)
        return aggregated_metrics, ranked_metrics_list

    @staticmethod
    def _split_pipeline_actor_results(
        results: list[dict] | None,
    ) -> tuple[list[dict], list[dict]]:
        if not results:
            return [], []
        rollout_metrics = [result.get("rollout_metrics", {}) for result in results]
        training_metrics = [result.get("training_metrics", {}) for result in results]
        return rollout_metrics, training_metrics

    def _maybe_eval_and_checkpoint(self, step: int) -> dict:
        run_val, save_model, _ = check_progress(
            self.global_step,
            self.max_steps,
            self.cfg.runner.val_check_interval,
            self.cfg.runner.save_interval,
            1.0,
            run_time_exceeded=False,
        )

        eval_metrics = {}
        if run_val:
            with self.timer("eval"):
                self.update_rollout_weights()
                eval_metrics = self.evaluate()
                eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
                self.metric_logger.log(data=eval_metrics, step=step)

        if save_model:
            self._save_checkpoint()

        return eval_metrics

    def _log_step_metrics(
        self,
        step: int,
        start_time: float,
        start_step: int,
        env_handle: Handle,
        rollout_handle: Handle,
        actor_training_handle: Handle,
        reward_handle: Handle | None,
        actor_rollout_metrics: list[dict],
        actor_training_metrics: list[dict],
        eval_metrics: dict,
    ) -> None:
        time_metrics = self.timer.consume_durations()
        time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}
        env_time_metrics, env_time_metrics_per_rank = env_handle.consume_durations(
            return_per_rank=True
        )
        rollout_time_metrics, rollout_time_metrics_per_rank = (
            rollout_handle.consume_durations(return_per_rank=True)
        )
        actor_time_metrics, actor_time_metrics_per_rank = (
            actor_training_handle.consume_durations(return_per_rank=True)
        )
        time_metrics.update({f"time/env/{k}": v for k, v in env_time_metrics.items()})
        time_metrics.update(
            {f"time/rollout/{k}": v for k, v in rollout_time_metrics.items()}
        )
        time_metrics.update(
            {f"time/actor/{k}": v for k, v in actor_time_metrics.items()}
        )
        if self.reward is not None:
            assert reward_handle is not None
            reward_time_metrics, reward_time_metrics_per_rank = (
                reward_handle.consume_durations(return_per_rank=True)
            )
            time_metrics.update(
                {f"time/reward/{k}": v for k, v in reward_time_metrics.items()}
            )

        env_results = env_handle.wait()
        env_results_list = [results for results in env_results if results is not None]
        env_metrics = compute_evaluate_metrics(env_results_list)
        env_metrics = {f"env/{k}": v for k, v in env_metrics.items()}
        ranked_env_results = [
            {"rank": rank, "env": rank_metrics}
            for rank, rank_metrics in enumerate(env_results)
            if rank_metrics is not None
        ]
        _, env_metrics_per_rank = self._process_ranked_eval_results(
            ranked_env_results, metric_field="env"
        )

        rollout_metrics = {
            f"rollout/{k}": v
            for k, v in self._aggregate_numeric_metrics(actor_rollout_metrics).items()
        }
        training_metrics = {
            f"train/{k}": v
            for k, v in self._aggregate_numeric_metrics(actor_training_metrics).items()
        }

        self.metric_logger.log(env_metrics, step)
        self.metric_logger.log(rollout_metrics, step)
        self.metric_logger.log(time_metrics, step)
        self.metric_logger.log(training_metrics, step)
        self._log_ranked_metrics(
            metrics_list=actor_rollout_metrics,
            step=step,
            prefix="rollout",
            worker_group_name=self.actor.worker_group_name,
        )
        self._log_ranked_metrics(
            metrics_list=actor_training_metrics,
            step=step,
            prefix="train",
            worker_group_name=self.actor.worker_group_name,
        )
        self._log_ranked_metrics(
            metrics_list=actor_time_metrics_per_rank,
            step=step,
            prefix="time/actor",
            worker_group_name=self.actor.worker_group_name,
        )
        self._log_ranked_metrics(
            metrics_list=rollout_time_metrics_per_rank,
            step=step,
            prefix="time/rollout",
            worker_group_name=self.rollout.worker_group_name,
        )
        self._log_ranked_metrics(
            metrics_list=env_time_metrics_per_rank,
            step=step,
            prefix="time/env",
            worker_group_name=self.env.worker_group_name,
        )
        self._log_ranked_metrics(
            metrics_list=env_metrics_per_rank,
            step=step,
            prefix="env",
            worker_group_name=self.env.worker_group_name,
        )
        if self.reward is not None:
            self._log_ranked_metrics(
                metrics_list=reward_time_metrics_per_rank,
                step=step,
                prefix="time/reward",
                worker_group_name=self.reward.worker_group_name,
            )

        logging_metrics = time_metrics
        logging_metrics.update(eval_metrics)
        logging_metrics.update(env_metrics)
        logging_metrics.update(rollout_metrics)
        logging_metrics.update(training_metrics)

        self.print_metrics_table_async(
            step, self.max_steps, start_time, logging_metrics, start_step
        )

    def _finish_run(self) -> None:
        self.metric_logger.finish()

        # Stop logging thread
        self.stop_logging = True
        self.log_queue.join()  # Wait for all queued logs to be processed
        self.log_thread.join(timeout=1.0)

    def _should_profile_step(self, step_idx: int) -> bool:
        return self._profile_all_steps or (
            self._profile_steps is not None and step_idx in self._profile_steps
        )

    def _open_profiling_window(self, step_idx: int) -> None:
        """Dispatch ``start_profile`` to all compute worker groups for this step."""
        self.logger.info(f"Opening profiling window at step {step_idx}")
        self.actor.start_profile(step_idx).wait()
        self.rollout.start_profile(step_idx).wait()
        self.env.start_profile(step_idx).wait()

    def _close_profiling_window(self, step_idx: int) -> None:
        """Dispatch ``stop_profile`` to all compute worker groups."""
        self.actor.stop_profile().wait()
        self.rollout.stop_profile().wait()
        self.env.stop_profile().wait()
        self.logger.info(f"Closed profiling window at step {step_idx}")

    def run(self):
        if self.rlix_runtime is not None:
            return self._run_rlix()
        if self.cfg.runner.get("use_training_pipeline", False):
            return self.run_pipeline()

        start_step = self.global_step
        start_time = time.time()
        for _step in range(start_step, self.max_steps):
            # set global step
            self.actor.set_global_step(self.global_step)
            self.rollout.set_global_step(self.global_step)

            profiled_step = (
                self.global_step
                if self._should_profile_step(self.global_step)
                else None
            )
            if profiled_step is not None:
                self._open_profiling_window(profiled_step)

            with self.timer("step"):
                with self.timer("sync_weights"):
                    if _step % self.weight_sync_interval == 0:
                        self.update_rollout_weights()
                with self.timer("generate_rollouts"):
                    env_handle: Handle = self.env.interact(
                        input_channel=self.env_channel,
                        rollout_channel=self.rollout_channel,
                        reward_channel=self.reward_channel,
                        actor_channel=self.actor_channel,
                    )
                    rollout_handle: Handle = self.rollout.generate(
                        input_channel=self.rollout_channel,
                        output_channel=self.env_channel,
                    )
                    reward_handle = None
                    if self.reward is not None:
                        reward_handle: Handle = self.reward.compute_rewards(
                            input_channel=self.reward_channel,
                            output_channel=self.env_channel,
                        )
                    self.actor.recv_rollout_trajectories(
                        input_channel=self.actor_channel
                    ).wait()
                    rollout_handle.wait()
                    if self.reward is not None:
                        reward_handle.wait()

                # compute advantages and returns.
                with self.timer("cal_adv_and_returns"):
                    actor_rollout_metrics = (
                        self.actor.compute_advantages_and_returns().wait()
                    )

                # actor training.
                actor_training_handle: Handle = self.actor.run_training()
                env_bootstrap_handle: Handle | None = None
                if self.overlap_env_bootstrap and _step + 1 < self.max_steps:
                    env_bootstrap_handle = self.env.prefetch_train_bootstrap(
                        rollout_channel=self.rollout_channel
                    )

                actor_training_metrics = actor_training_handle.wait()
                if env_bootstrap_handle is not None:
                    env_bootstrap_handle.wait()

                self.global_step += 1
                eval_metrics = self._maybe_eval_and_checkpoint(_step)

            if profiled_step is not None:
                self._close_profiling_window(profiled_step)

            self._log_step_metrics(
                step=_step,
                start_time=start_time,
                start_step=start_step,
                env_handle=env_handle,
                rollout_handle=rollout_handle,
                actor_training_handle=actor_training_handle,
                reward_handle=reward_handle,
                actor_rollout_metrics=actor_rollout_metrics,
                actor_training_metrics=actor_training_metrics,
                eval_metrics=eval_metrics,
            )

        self._finish_run()

    def _run_rlix(self) -> None:
        """Run the supported synchronous loop through registered stage ownership."""
        if self.cfg.runner.get("use_training_pipeline", False):
            raise RuntimeError("RLix does not support the training pipeline runner")
        start_step = self.global_step
        start_time = time.time()
        for _step in range(start_step, self.max_steps):
            if self._should_profile_step(self.global_step):
                raise RuntimeError(
                    "RLix profiling requires component-scoped stage integration"
                )
            self.actor.set_global_step(self.global_step)
            self.rollout.set_global_step(self.global_step)
            with self.timer("step"):
                with self.timer("sync_weights"):
                    if _step % self.weight_sync_interval == 0:
                        self.update_rollout_weights()
                with self.timer("generate_rollouts"):
                    batch_receipt = self._collect_rlix_rollouts()
                with self.timer("cal_adv_and_returns"):
                    (
                        actor_rollout_metrics,
                        actor_training_metrics,
                        _,
                    ) = self._train_rlix_batch(batch_receipt)
                eval_metrics = self._maybe_eval_and_checkpoint(_step)
            self._log_rlix_step_metrics(
                step=_step,
                start_time=start_time,
                start_step=start_step,
                actor_rollout_metrics=actor_rollout_metrics,
                actor_training_metrics=actor_training_metrics,
                eval_metrics=eval_metrics,
            )
        self._finish_run()

    def _log_rlix_step_metrics(
        self,
        *,
        step: int,
        start_time: float,
        start_step: int,
        actor_rollout_metrics: list[dict],
        actor_training_metrics: list[dict],
        eval_metrics: dict,
    ) -> None:
        """Log elastic results without legacy environment group handles."""
        time_metrics = {
            f"time/{key}": value
            for key, value in self.timer.consume_durations().items()
        }
        session = self._last_rlix_collection_session
        env_metrics_list = [
            session.final_env_metrics[rank]
            for rank in sorted(session.final_env_metrics)
        ]
        env_metrics = (
            compute_evaluate_metrics(env_metrics_list) if env_metrics_list else {}
        )
        env_metrics = {f"env/{key}": value for key, value in env_metrics.items()}
        rollout_metrics = {
            f"rollout/{key}": value
            for key, value in self._aggregate_numeric_metrics(
                actor_rollout_metrics
            ).items()
        }
        training_metrics = {
            f"train/{key}": value
            for key, value in self._aggregate_numeric_metrics(
                actor_training_metrics
            ).items()
        }
        for metrics in (
            time_metrics,
            env_metrics,
            rollout_metrics,
            training_metrics,
            eval_metrics,
        ):
            self.metric_logger.log(metrics, step)
        logging_metrics = {
            **time_metrics,
            **env_metrics,
            **rollout_metrics,
            **training_metrics,
            **eval_metrics,
        }
        self.print_metrics_table_async(
            step,
            self.max_steps,
            start_time,
            logging_metrics,
            start_step,
        )

    def run_pipeline(self):
        start_step = self.global_step
        start_time = time.time()
        for _step in range(start_step, self.max_steps):
            # set global step
            self.actor.set_global_step(self.global_step)
            self.rollout.set_global_step(self.global_step)

            profiled_step = (
                self.global_step
                if self._should_profile_step(self.global_step)
                else None
            )
            if profiled_step is not None:
                self._open_profiling_window(profiled_step)

            with self.timer("step"):
                with self.timer("sync_weights"):
                    if _step % self.weight_sync_interval == 0:
                        self.update_rollout_weights()
                env_handle: Handle = self.env.interact(
                    input_channel=self.env_channel,
                    rollout_channel=self.rollout_channel,
                    reward_channel=self.reward_channel,
                    actor_channel=self.actor_channel,
                )
                rollout_handle: Handle = self.rollout.generate(
                    input_channel=self.rollout_channel,
                    output_channel=self.env_channel,
                )
                reward_handle = None
                if self.reward is not None:
                    reward_handle: Handle = self.reward.compute_rewards(
                        input_channel=self.reward_channel,
                        output_channel=self.env_channel,
                    )
                # actor training.
                actor_training_handle: Handle = self.actor.run_training(
                    input_channel=self.actor_channel
                )
                with self.timer("generate_rollouts"):
                    rollout_handle.wait()
                    if self.reward is not None:
                        reward_handle.wait()

                env_bootstrap_handle: Handle | None = None
                if self.overlap_env_bootstrap and _step + 1 < self.max_steps:
                    env_bootstrap_handle = self.env.prefetch_train_bootstrap(
                        rollout_channel=self.rollout_channel
                    )

                actor_results = actor_training_handle.wait()
                actor_rollout_metrics, actor_training_metrics = (
                    self._split_pipeline_actor_results(actor_results)
                )
                if env_bootstrap_handle is not None:
                    env_bootstrap_handle.wait()

                self.global_step += 1
                eval_metrics = self._maybe_eval_and_checkpoint(_step)

            if profiled_step is not None:
                self._close_profiling_window(profiled_step)

            self._log_step_metrics(
                step=_step,
                start_time=start_time,
                start_step=start_step,
                env_handle=env_handle,
                rollout_handle=rollout_handle,
                actor_training_handle=actor_training_handle,
                reward_handle=reward_handle,
                actor_rollout_metrics=actor_rollout_metrics,
                actor_training_metrics=actor_training_metrics,
                eval_metrics=eval_metrics,
            )

        self._finish_run()

    def _save_checkpoint(self):
        if self.rlix_runtime is not None:
            from rlix_core.protocol.types import ACTOR_TRAIN_CLUSTER_NAME, Priority

            with self.rlix_runtime.fixed_stage(
                cluster_name=ACTOR_TRAIN_CLUSTER_NAME,
                priority=Priority.ACTOR_TRAINING,
                global_step=self.global_step,
            ) as stage:
                self._save_checkpoint_workers()
                residencies = self._get_rlix_fixed_residencies(self.actor)
                stage.complete(
                    self.rlix_runtime.fixed_residency_receipt(
                        cluster_name=ACTOR_TRAIN_CLUSTER_NAME,
                        worker_residencies=residencies,
                        policy_version=self.global_step,
                    )
                )
            return
        self._save_checkpoint_workers()

    def _save_checkpoint_workers(self) -> None:
        """Save actor state using the existing checkpoint layout."""
        self.logger.info(f"Saving checkpoint at step {self.global_step}.")
        base_output_dir = os.path.join(
            self.cfg.runner.logger.log_path,
            self.cfg.runner.logger.experiment_name,
            f"checkpoints/global_step_{self.global_step}",
        )
        actor_save_path = os.path.join(base_output_dir, "actor")
        os.makedirs(actor_save_path, exist_ok=True)
        self.actor.save_checkpoint(actor_save_path, self.global_step).wait()

    def set_max_steps(self):
        self.num_steps_per_epoch = 1
        self.max_steps = self.num_steps_per_epoch * self.cfg.runner.max_epochs

        if (max_steps := self.cfg.runner.get("max_steps", -1)) >= 0:
            self.max_steps = min(self.max_steps, max_steps)

    @property
    def epoch(self):
        return self.global_step // self.num_steps_per_epoch
