# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL의 최근 실행 이력과 정책 버전을 수집하고 진단 파일로 저장함."""

from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from ._tensor_utils import _copy_tensors

if TYPE_CHECKING:
    from rsl_rl.runners import OnPolicyRunner


class RslRlHistoryLogger:
    """최근 실행의 이력 버퍼와 정책 사본을 소유하며 학습 카운터는 호출자로부터 전달받음.

    action과 물리 상태, reset 경계, PPO 입력을 같은 프레임·정책 버전 기준으로 저장함.
    저장한 이력은 관측에 추가하지 않으며 물리 solver의 정확한 재생을 보장하지 않음.
    """

    def __init__(self, runner: OnPolicyRunner, log_dir: Path, *, steps: int, threshold: float):
        if steps < 1 or not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("action history 길이는 1 이상이며 임계값은 유한한 양수여야 함.")
        self.runner = runner
        self.alg = runner.alg
        self.log_dir = log_dir
        self.threshold = threshold
        self.saved = False
        self._frames = deque(maxlen=steps)
        self._policy_versions = deque()
        self._policy_version = 0
        self._absolute_step = 0
        self._latest_update = None
        self._episode_ids = None

    @property
    def has_frames(self) -> bool:
        """정책 호출로 생성된 실행 프레임이 있는지 반환함."""
        return bool(self._frames)

    def start(self):
        """첫 action 이전의 에피소드 번호와 초기 정책 사본을 보관함."""
        self._episode_ids = torch.zeros_like(self.runner.env.episode_length_buf)
        self._remember_policy(None)

    def record_external_reset(self):
        """Mark the new training episodes after a shared-environment evaluation reset."""
        self._episode_ids += 1

    def record_action(self, actions):
        """샘플링한 action과 동일 시점의 정책 평균·표준편차·가치를 복사함."""
        self._frames[-1]["data"].update(
            _copy_tensors(
                {
                    "raw_action": actions,
                    "action_mean": self.alg.actor.output_mean,
                    "action_std": self.alg.actor.output_std,
                    "value": self.alg.transition.values,
                    "actions_log_prob": self.alg.transition.actions_log_prob,
                }
            )
        )

    def record_environment_action(self, actions):
        """wrapper 클리핑 후 스케일 적용 전 환경 입력을 원래 정책 출력과 구분하여 보존함."""
        self._frames[-1]["data"]["environment_action"] = _copy_tensors(actions)

    def mark_env_step_started(self):
        """환경 실행의 시작을 기록하며 정상 반환 여부와 구분함."""
        if self._frames:
            self._frames[-1]["env_step_started"] = True

    def record_env_result(self, result):
        """환경 실행 결과와 reset 경계를 보존하고 에피소드 번호를 갱신함."""
        if not self._frames:
            return
        env = self.runner.env.unwrapped
        time_outs = result[3].get("time_outs")
        if hasattr(env, "reset_terminated"):
            terminated = env.reset_terminated
            self._frames[-1]["termination_source"] = "environment.reset_terminated"
        else:
            terminated = result[2].bool() & ~time_outs.bool() if time_outs is not None else None
            self._frames[-1]["termination_source"] = "derived_from_dones_and_time_outs"
        self._frames[-1]["env_executed"] = True
        if "applied_action" not in self._frames[-1]["data"]:
            self._frames[-1]["data"]["applied_action"] = _copy_tensors(
                env.action_manager.action if hasattr(env, "action_manager") else None
            )
        self._frames[-1]["data"].update(
            _copy_tensors(
                {
                    "rewards": result[1],
                    "dones": result[2],
                    "time_outs": time_outs,
                    "terminated": terminated,
                    "reset_time_outs": env.reset_time_outs if hasattr(env, "reset_time_outs") else None,
                    "episode_length_after": self.runner.env.episode_length_buf,
                    "command_counters_after": self._command_counters(),
                }
            )
        )
        self._episode_ids += result[2].to(self._episode_ids.dtype)

    def record_bootstrap_rewards(self, rewards):
        """timeout의 가치 보정까지 적용된 실제 학습 보상을 보존함."""
        if self._frames:
            self._frames[-1]["data"]["bootstrapped_rewards"] = rewards.clone()

    def record_applied_action(self):
        """reset이 액션 버퍼를 초기화하기 전에 실제 전달된 액션을 보존함."""
        env = self.runner.env.unwrapped
        if self._frames and hasattr(env, "action_manager"):
            self._frames[-1]["data"]["applied_action"] = _copy_tensors(env.action_manager.action)

    def record_physics(self, name, physics):
        """reset 전 물리 상태를 원본 버퍼와 독립적인 복사본으로 보관함."""
        if self._frames:
            self._frames[-1]["data"].setdefault("physics_after_step", {})[name] = _copy_tensors(physics)

    def record_reward_terms(self, reward_manager):
        """계산이 끝난 보상 항별 값과 이름을 동일 프레임에 저장함."""
        if self._frames:
            self._frames[-1]["data"]["reward_terms"] = reward_manager._step_reward.clone()
            self._frames[-1]["reward_term_names"] = list(reward_manager.active_terms)

    def begin_update(self, iteration: int):
        """PPO 업데이트 직전 모델·optimizer·난수 상태와 rollout 입력을 보존함."""
        storage = self.alg.storage
        self._latest_update = {
            "iteration": iteration,
            "algorithm_before": _copy_tensors(self.alg.save(), cpu=True),
            "rng_state": self._rng_state(),
            "rollout": _copy_tensors(
                {
                    "observations": storage.observations,
                    "actions": storage.actions,
                    "actions_log_prob": storage.actions_log_prob,
                    "distribution_params": storage.distribution_params,
                    "rewards": storage.rewards,
                    "dones": storage.dones,
                    "values": storage.values,
                    "returns": storage.returns,
                    "advantages": storage.advantages,
                },
                cpu=True,
            ),
        }

    def end_update(self, iteration: int, losses):
        """완료된 업데이트의 loss와 새 정책 버전을 연결함."""
        self._latest_update["loss_after"] = dict(losses)
        self._policy_version += 1
        self._remember_policy(iteration)

    def _rng_state(self):
        """샘플링 전 난수 상태를 읽으며 난수열 자체는 진행시키지 않음."""
        device = torch.device(self.alg.device)
        return {
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        }

    def _command_counters(self):
        env = self.runner.env.unwrapped
        if not hasattr(env, "command_manager"):
            return {}
        return {name: env.command_manager.get_term(name).command_counter for name in env.command_manager.active_terms}

    def _remember_policy(self, iteration_updated):
        """history가 참조하는 모델 버전과 바로 이전 버전을 독립 복사본으로 보관함."""
        self._policy_versions.append(
            {
                "version": self._policy_version,
                "iteration_updated": iteration_updated,
                "algorithm": _copy_tensors(self.alg.save(), cpu=True),
            }
        )
        if self._frames:
            oldest = self._frames[0]["policy_version"]
            while self._policy_versions[0]["version"] < oldest - 1:
                self._policy_versions.popleft()

    def start_frame(self, obs, *, iteration: int, rollout_step: int):
        """actor 입력을 action 샘플링 전에 복사하며 reset 경계를 지나도 이력을 유지함."""
        self._frames.append(
            {
                "absolute_step": self._absolute_step,
                "iteration": iteration,
                "rollout_step": rollout_step,
                "policy_version": self._policy_version,
                "env_step_started": False,
                "env_executed": False,
                "rng_state": self._rng_state(),
                "data": _copy_tensors(
                    {
                        "observations": obs,
                        "episode_length_before": self.runner.env.episode_length_buf,
                        "episode_id_before": self._episode_ids,
                        "command_counters_before": self._command_counters(),
                    }
                ),
            }
        )
        self._absolute_step += 1

    def save(self, env_ids, reason, *, iteration: int, rollout_step: int, minibatch: int):
        """문제가 생긴 환경의 최근 이력과 모델 버전, 마지막 PPO 입력을 저장함."""
        if not self._frames:
            return
        if env_ids is None:
            env_ids = list(range(self.runner.env.num_envs))
        report = {
            "reason": reason,
            "iteration": iteration,
            "rollout_step": rollout_step,
            "minibatch": minibatch,
            "env_ids": env_ids,
            "threshold": self.threshold,
            "threshold_target": "environment_action",
            "frame_count": len(self._frames),
            "first_step": self._frames[0]["absolute_step"],
            "last_step": self._frames[-1]["absolute_step"],
        }
        frames = [{**frame, "data": _copy_tensors(frame["data"], cpu=True, env_ids=env_ids)} for frame in self._frames]
        env = self.runner.env.unwrapped
        metadata = {
            "obs_groups": self.runner.cfg["obs_groups"],
            "runner_cfg": _copy_tensors(self.runner.cfg, cpu=True),
            "clip_actions": self.runner.cfg.get("clip_actions"),
            "threshold_target": "environment_action",
            "environment_action": "wrapper 클리핑 후 스케일·offset 적용 전 실제 환경 입력임.",
            "num_envs": self.runner.env.num_envs,
            "data_env_axis": "frame.data의 첫 축은 report.env_ids 순서임. latest_update는 전체 환경을 포함함.",
            "env_executed": "env.step()이 정상 반환한 경우 True임. 부분 실행은 env_step_started로 구분함.",
        }
        if hasattr(env, "step_dt"):
            metadata["step_dt"] = env.step_dt
        if hasattr(env, "observation_manager"):
            metadata["observation_terms"] = env.observation_manager.active_terms
            metadata["observation_term_shapes"] = env.observation_manager.group_obs_term_dim
        if hasattr(env, "scene"):
            metadata["articulation_joint_names"] = {
                name: asset.joint_names for name, asset in env.scene.articulations.items()
            }
        if hasattr(env, "command_manager"):
            metadata["command_annotations"] = {}
            for name in env.command_manager.active_terms:
                term = env.command_manager.get_term(name)
                metadata["command_annotations"][name] = {
                    key: _copy_tensors(getattr(term, key), cpu=True, env_ids=env_ids)
                    for key in ("is_transition_env", "transition_type")
                    if hasattr(term, key)
                }
        self.log_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "report": report,
                "frames": frames,
                "policy_versions": list(self._policy_versions),
                "latest_update": self._latest_update,
                "metadata": metadata,
            },
            self.log_dir / "action_history.pt",
        )
        (self.log_dir / "action_history.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        self.saved = True
