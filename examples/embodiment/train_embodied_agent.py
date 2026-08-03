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

import json

import hydra
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.reward.reward_worker import EmbodiedRewardWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

mp.set_start_method("spawn", force=True)


@hydra.main(
    version_base="1.1", config_path="config", config_name="maniskill_ppo_openvlaoft"
)
def main(cfg) -> None:
    from rlinf.scheduler.rlix.validation import validate_rlix_entrypoint

    validate_rlix_entrypoint(cfg, entrypoint="train_embodied_agent")
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(
        cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path
    )
    component_placement = HybridComponentPlacement(cfg, cluster)
    rlix_enabled = bool(cfg.rlix.enabled)
    resolved_rlix = None
    if rlix_enabled:
        from rlinf.scheduler.rlix.entrypoint import preflight_rlix_placements

        resolved_rlix = preflight_rlix_placements(component_placement, cluster)

    # Create actor worker group
    actor_placement = (
        resolved_rlix.actor_strategy
        if resolved_rlix is not None
        else component_placement.get_strategy("actor")
    )
    use_training_pipeline = bool(cfg.runner.get("use_training_pipeline", False))

    if cfg.algorithm.loss_type == "embodied_sac":
        if use_training_pipeline:
            raise ValueError(
                "runner.use_training_pipeline=True is not supported for embodied_sac."
            )
        from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy

        actor_worker_cls = EmbodiedSACFSDPPolicy
    elif cfg.algorithm.loss_type == "rlt_ac":
        if use_training_pipeline:
            raise ValueError(
                "runner.use_training_pipeline=True is not supported for rlt_ac."
            )
        from rlinf.workers.actor.rlt_ac_policy_worker import RLTACFSDPPolicy

        actor_worker_cls = RLTACFSDPPolicy
    elif cfg.algorithm.loss_type == "embodied_dagger":
        if use_training_pipeline:
            raise ValueError(
                "runner.use_training_pipeline=True is not supported for embodied_dagger."
            )
        from rlinf.workers.actor.fsdp_dagger_policy_worker import (
            EmbodiedDAGGERFSDPPolicy,
        )

        actor_worker_cls = EmbodiedDAGGERFSDPPolicy
    elif cfg.algorithm.loss_type == "embodied_nft":
        if use_training_pipeline:
            raise ValueError(
                "runner.use_training_pipeline=True is not supported for embodied_nft."
            )
        from rlinf.workers.actor.fsdp_nft_policy_worker import EmbodiedNFTFSDPPolicy

        actor_worker_cls = EmbodiedNFTFSDPPolicy
    else:
        if use_training_pipeline:
            from rlinf.workers.actor.fsdp_actor_worker_pipeline import (
                PipelineEmbodiedFSDPActor,
            )

            actor_worker_cls = PipelineEmbodiedFSDPActor
        else:
            from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor

            actor_worker_cls = EmbodiedFSDPActor

    # Create rollout worker group
    rollout_placement = (
        resolved_rlix.rollout_strategy
        if resolved_rlix is not None
        else component_placement.get_strategy("rollout")
    )
    # Create env worker group
    env_placement = (
        resolved_rlix.env_strategy
        if resolved_rlix is not None
        else component_placement.get_strategy("env")
    )
    rlix_runtime = None
    if rlix_enabled:
        from rlinf.scheduler.rlix.entrypoint import launch_registered_rlix_workers
        from rlinf.scheduler.rlix.runtime import (
            bootstrap_registered_rlix_pipeline,
        )

        launched = launch_registered_rlix_workers(
            cluster=cluster,
            actor_group=actor_worker_cls.create_group(cfg),
            rollout_group=MultiStepRolloutWorker.create_group(cfg),
            env_group=EnvWorker.create_group(cfg),
            actor_name=cfg.actor.group_name,
            rollout_name=cfg.rollout.group_name,
            env_name=cfg.env.group_name,
            resolved=resolved_rlix,
            worker_max_concurrency=cfg.rlix.worker_max_concurrency,
            operation_timeout_s=cfg.rlix.operation_timeout_s,
            enable_gpu_tracing=cfg.rlix.enable_gpu_tracing,
            completed_bundle_handoff=cfg.rlix.completed_bundle_handoff,
            policy_sync_mode=str(
                OmegaConf.select(cfg, "rlix.policy_sync.mode", default="fixed_all_rank")
            ),
            policy_sync_max_retries=int(
                OmegaConf.select(cfg, "rlix.policy_sync.max_retries", default=1)
            ),
            bootstrapper=bootstrap_registered_rlix_pipeline,
        )
        actor_group = launched.actor
        rollout_group = launched.rollout
        env_group = launched.env
        rlix_runtime = launched.runtime
    else:
        from rlinf.scheduler.rlix.entrypoint import launch_standalone_worker_groups

        actor_group, rollout_group, env_group = launch_standalone_worker_groups(
            cluster=cluster,
            actor_group_factory=lambda: actor_worker_cls.create_group(cfg),
            rollout_group_factory=lambda: MultiStepRolloutWorker.create_group(cfg),
            env_group_factory=lambda: EnvWorker.create_group(cfg),
            actor_name=cfg.actor.group_name,
            rollout_name=cfg.rollout.group_name,
            env_name=cfg.env.group_name,
            actor_placement=actor_placement,
            rollout_placement=rollout_placement,
            env_placement=env_placement,
        )

    reward_group = None
    if cfg.get("reward", {}).get("use_reward_model", False) and not cfg.get(
        "reward", {}
    ).get("standalone_realworld", False):
        # Create reward worker group
        reward_placement = component_placement.get_strategy("reward")
        reward_group = EmbodiedRewardWorker.create_group(cfg).launch(
            cluster, name=cfg.reward.group_name, placement_strategy=reward_placement
        )

    runner_kwargs = {}
    if rlix_runtime is not None:
        runner_kwargs["rlix_runtime"] = rlix_runtime
    runner = EmbodiedRunner(
        cfg=cfg,
        actor=actor_group,
        rollout=rollout_group,
        env=env_group,
        reward=reward_group,
        **runner_kwargs,
    )

    if rlix_runtime is None:
        runner.init_workers()
        runner.run()
    else:
        from rlinf.scheduler.rlix.entrypoint import run_registered_rlix_runner

        run_registered_rlix_runner(runner=runner, runtime=rlix_runtime)


if __name__ == "__main__":
    main()
