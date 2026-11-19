# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL의 검사 시점과 NaN·Inf 검출 및 중단 처리를 통합 관리함."""

from __future__ import annotations

import contextlib
import json
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from rsl_rl.algorithms import PPO

from ._tensor_utils import _tensors
from .rsl_rl_history_logger import RslRlHistoryLogger

if TYPE_CHECKING:
    from rsl_rl.runners import OnPolicyRunner


class RslRlFiniteCheck(contextlib.AbstractContextManager):
    """단일 장치 PPO의 관측·학습 목표·gradient·optimizer 상태를 검사함.

    검사는 값을 대체하지 않으며 NaN·Inf를 발견하면 진단 파일을 저장하고 예외를 발생시킴.
    물리 상태는 정책 스텝의 reset 전에 검사하며 solver 내부 substep은 검사하지 않음.
    gradient는 clipping 전에 검사함. 반환 loss는 PPO.update() 종료 후 검사하므로
    PPO 내부 모든 연산의 최초 비정상 값을 포착하는 것은 아님.
    역전파 도중 저장하는 parameter gradient는 누적이 완료되지 않은 부분 상태일 수 있음.
    GPU reduction과 CPU 동기화가 추가되므로 원인 분석용 실행에서 사용함.

    action history는 최근 정책 스텝만 보관하며 관측으로 추가하지 않음.
    원래 정책 출력은 NaN·Inf를 검사하고, 크기 임계값은 wrapper의 클리핑 후 환경 입력에 적용함.
    임계값 초과 시 물리 스텝 전에 기록하고 중단함. 과거 실행이나 물리 solver의 정확한 재생을 보장하지 않음.
    """

    def __init__(
        self,
        runner: OnPolicyRunner,
        log_dir: str,
        *,
        record_action_history: bool = False,
        action_history_steps: int = 64,
        action_history_threshold: float = 100.0,
    ):
        self.runner = runner
        self.alg = runner.alg
        self.log_dir = Path(log_dir) / "numerics"
        self.iteration = runner.current_learning_iteration
        self.stage = "initial"
        self.rollout_step = 0
        self.minibatch = 0
        self._stack = contextlib.ExitStack()
        self._history = (
            RslRlHistoryLogger(runner, self.log_dir, steps=action_history_steps, threshold=action_history_threshold)
            if record_action_history
            else None
        )

    def __enter__(self):
        if not isinstance(self.alg, PPO) or self.runner.is_distributed:
            raise ValueError("--check_finite는 단일 장치의 PPO 학습에서 지원함.")
        try:
            self._model_state()
            if self._history is not None:
                self._history.start()
                self.log_dir.mkdir(parents=True, exist_ok=True)
                (self.log_dir / "finite_check_source.py").write_text(Path(__file__).read_text())
                # 모듈 분리 후에도 진단 당시의 전체 실행 소스를 함께 보존함.
                source_dir = self.log_dir / "diagnostics_source"
                source_dir.mkdir(exist_ok=True)
                for filename in (
                    "__init__.py",
                    "_tensor_utils.py",
                    "rsl_rl_history_logger.py",
                    "rsl_rl_finite_check.py",
                ):
                    (source_dir / filename).write_bytes(Path(__file__).with_name(filename).read_bytes())
            env = self.runner.env.unwrapped
            self._replace(self.runner.env, "step", self._env_step)
            if self._history is not None and env is not self.runner.env:
                self._replace(env, "step", self._base_env_step)
            self._replace(self.alg, "act", self._act)
            self._replace(self.alg, "process_env_step", self._process_env_step)
            self._replace(self.alg, "compute_returns", self._compute_returns)
            self._replace(self.alg, "update", self._update)
            for name, model in (("actor", self.alg.actor), ("critic", self.alg.critic)):
                handle = model.register_forward_pre_hook(partial(self._forward_input, name))
                self._stack.callback(handle.remove)
                handle = model.register_forward_hook(partial(self._forward_output, name))
                self._stack.callback(handle.remove)
                parameters = tuple((key, param) for key, param in model.named_parameters() if param.requires_grad)
                if parameters:
                    handle = torch.autograd.graph.register_multi_grad_hook(
                        tuple(param for _, param in parameters), partial(self._gradients, name, parameters)
                    )
                    self._stack.callback(handle.remove)
            handle = self.alg.optimizer.register_step_pre_hook(self._before_optimizer)
            self._stack.callback(handle.remove)
            handle = self.alg.optimizer.register_step_post_hook(self._after_optimizer)
            self._stack.callback(handle.remove)
            if hasattr(env, "reward_manager"):
                self._replace(env.reward_manager, "compute", self._reward)
            return self
        except BaseException:
            self._stack.close()
            raise

    def __exit__(self, *exc):
        try:
            if self._history is not None and not self._history.saved:
                reason = f"exception: {exc[0].__name__}" if exc[0] is not None else "training completed"
                self._save_history(None, reason)
        finally:
            self._stack.__exit__(*exc)
        return False

    @contextlib.contextmanager
    def snapshot_probe(self):
        """Keep finite checks active without changing training action-history frames."""
        history, stage = self._history, self.stage
        self._history = None
        self.stage = "snapshot"
        try:
            yield
        finally:
            self._history, self.stage = history, stage
            if history is not None:
                history.record_external_reset()

    def _replace(self, owner, name, checker):
        """이 실행의 메서드만 감싸고 정상 종료·예외 모두에서 원래 속성을 복원함."""
        original = getattr(owner, name)
        if name in vars(owner):
            self._stack.callback(setattr, owner, name, vars(owner)[name])
        else:
            self._stack.callback(delattr, owner, name)
        setattr(owner, name, partial(checker, original))

    def _save_history(self, env_ids, reason):
        """검출 시점의 공통 카운터로 실행 이력을 저장함."""
        if self._history is not None:
            self._history.save(
                env_ids, reason, iteration=self.iteration, rollout_step=self.rollout_step, minibatch=self.minibatch
            )

    def _check(self, name, value, axes=()):
        """같은 장치의 검사 결과를 모아 동기화하며 이상 값이 있을 때만 상세 정보를 복사함."""
        by_device = {}
        for key, tensor in _tensors(value, name):
            by_device.setdefault(tensor.device, []).append((key, tensor))
        for items in by_device.values():
            flags = torch.stack([torch.isfinite(tensor).all() for _, tensor in items])
            if not flags.all().item():
                index = (~flags).nonzero()[0].item()
                self._fail(*items[index], axes)

    def _fail(self, name, tensor, axes):
        """원본 체크포인트와 별도로 오염된 값·학습 상태를 보존하고 학습을 중단함."""
        value = tensor.detach().cpu()
        axes = axes[: value.ndim]
        invalid = ~torch.isfinite(value)
        indices = invalid.nonzero()[:16].tolist()
        report = {
            "stage": self.stage,
            "iteration": self.iteration,
            "rollout_step": self.rollout_step,
            "minibatch": self.minibatch,
            "tensor": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "axes": list(axes),
            "indices": indices,
            "nan_count": torch.isnan(value).sum().item(),
            "inf_count": torch.isinf(value).sum().item(),
        }
        if "env" in axes:
            env_axis = axes.index("env")
            bad_envs = invalid.movedim(env_axis, 0).flatten(1).any(dim=1) if value.ndim > 1 else invalid
            report["env_ids"] = bad_envs.nonzero().flatten().tolist()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        (self.log_dir / "nonfinite.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        storage = self.alg.storage
        torch.save(
            {
                "report": report,
                "failure_tensor": value,
                "algorithm": self.alg.save(),
                "gradients": {
                    f"{model_name}.{key}": param.grad.detach().cpu()
                    for model_name, model in (("actor", self.alg.actor), ("critic", self.alg.critic))
                    for key, param in model.named_parameters()
                    if param.grad is not None
                },
                "observations": storage.observations,
                "actions": storage.actions,
                "actions_log_prob": storage.actions_log_prob,
                "distribution_params": storage.distribution_params,
                "rewards": storage.rewards,
                "dones": storage.dones,
                "values": storage.values,
                "returns": storage.returns,
                "advantages": storage.advantages,
                "torch_rng_state": torch.get_rng_state(),
            },
            self.log_dir / "snapshot.pt",
        )
        self._save_history(report.get("env_ids"), f"NaN/Inf: {name}")
        raise FloatingPointError(
            f"NaN/Inf detected: {name}; stage={self.stage}, iteration={self.iteration}, "
            f"rollout_step={self.rollout_step}, minibatch={self.minibatch}; "
            f"indices={indices}, axes={axes}. Report: {self.log_dir / 'nonfinite.json'}"
        )

    def _model_state(self):
        self._check("actor.state", self.alg.actor.state_dict())
        self._check("critic.state", self.alg.critic.state_dict())
        self._check("optimizer.state", list(self.alg.optimizer.state.values()))

    def _act(self, original, obs):
        self.stage = "rollout"
        self.rollout_step = self.alg.storage.step
        self.minibatch = 0
        if self._history is not None:
            self._history.start_frame(obs, iteration=self.iteration, rollout_step=self.rollout_step)
        result = original(obs)
        if self._history is not None:
            self._history.record_action(result)
        self._check(
            "transition",
            {
                "actions": result,
                "actions_log_prob": self.alg.transition.actions_log_prob,
                "distribution_params": self.alg.transition.distribution_params,
            },
            ("env", "component"),
        )
        return result

    def _base_env_step(self, original, actions):
        """클리핑이 끝난 실제 환경 입력을 검사한 뒤 물리 스텝을 실행함."""
        if self._history is not None and self._history.has_frames:
            self._history.record_environment_action(actions)
            exceeded = actions.abs().amax(dim=-1) > self._history.threshold
            if exceeded.any().item():
                self._save_history(exceeded.nonzero().flatten().tolist(), "environment action threshold exceeded")
                raise RuntimeError(
                    f"Action history trigger: abs(environment_action) > {self._history.threshold}; "
                    f"iteration={self.iteration}, rollout_step={self.rollout_step}. "
                    f"Stopped before physics step. Report: {self.log_dir / 'action_history.json'}"
                )
            self._history.mark_env_step_started()
        return original(actions)

    def _env_step(self, original, actions):
        self._check("environment.actions", actions, ("env", "component"))
        if self.runner.env is self.runner.env.unwrapped:
            result = self._base_env_step(original, actions)
        else:
            result = original(actions)
        if self._history is not None:
            self._history.record_env_result(result)
        self._check(
            "environment.outputs",
            {"observations": result[0], "rewards": result[1], "dones": result[2]},
            ("env", "component"),
        )
        return result

    def _process_env_step(self, original, obs, rewards, dones, extras):
        result = original(obs, rewards, dones, extras)
        if self._history is not None:
            self._history.record_bootstrap_rewards(self.alg.storage.rewards[self.rollout_step])
        self._check("bootstrap.rewards", self.alg.storage.rewards[self.rollout_step], ("env", "component"))
        return result

    def _compute_returns(self, original, obs):
        self.stage = "returns"
        result = original(obs)
        self._check(
            "targets",
            {"returns": self.alg.storage.returns, "advantages": self.alg.storage.advantages},
            ("step", "env", "component"),
        )
        return result

    def _update(self, original):
        self.stage = "update"
        self.minibatch = 0
        if self._history is not None:
            self._history.begin_update(self.iteration)
        result = original()
        self._check("loss", result)
        if self._history is not None:
            self._history.end_update(self.iteration, result)
        self.iteration += 1
        return result

    def _forward_input(self, name, model, args):
        axes = ("batch_row", "component") if self.stage == "update" else ("env", "component")
        self._check(f"{name}.input", args[0].select(*model.obs_groups), axes)

    def _forward_output(self, name, model, args, result):
        axes = ("batch_row", "component") if self.stage == "update" else ("env", "component")
        self._check(f"{name}.output", result, axes)

    def _gradients(self, name, parameters, gradients):
        with torch.no_grad():
            self._check(f"{name}.gradient", {key: grad for (key, _), grad in zip(parameters, gradients)})
            present = [grad for grad in gradients if grad is not None]
            if present:
                self._check(f"{name}.gradient_norm_before_clip", torch.nn.utils.get_total_norm(present))

    def _before_optimizer(self, optimizer, args, kwargs):
        self._check(
            "gradient_after_clip",
            [param.grad for group in optimizer.param_groups for param in group["params"]],
        )

    def _after_optimizer(self, optimizer, args, kwargs):
        self._model_state()
        self.minibatch += 1

    def _reward(self, original, dt):
        env = self.runner.env.unwrapped
        if self._history is not None:
            self._history.record_applied_action()
        for name, asset in env.scene.articulations.items():
            physics = {
                "root_link_pose_w": asset.data.root_link_pose_w.torch,
                "root_com_vel_w": asset.data.root_com_vel_w.torch,
                "joint_pos": asset.data.joint_pos.torch,
                "joint_vel": asset.data.joint_vel.torch,
                "computed_effort": asset.actuators.computed_effort.torch,
                "applied_effort": asset.actuators.applied_effort.torch,
            }
            if self._history is not None:
                self._history.record_physics(name, physics)
            self._check(
                f"physics.{name}",
                physics,
                ("env", "component"),
            )
        result = original(dt)
        if self._history is not None:
            self._history.record_reward_terms(env.reward_manager)
        self._check("reward.terms", env.reward_manager._step_reward, ("env", "reward_term"))
        self._check("reward.total", result, ("env",))
        return result
