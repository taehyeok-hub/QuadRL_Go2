# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import json
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("rsl_rl")

from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp import last_action

from isaaclab_rl.diagnostics.rsl_rl_finite_check import RslRlFiniteCheck
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

pytestmark = pytest.mark.unit


class _CpuEnv(VecEnv):
    """작은 실제 PPO가 사용할 관측과 보상만 제공하는 CPU 환경임."""

    def __init__(self):
        self.num_envs = 4
        self.num_actions = 2
        self.max_episode_length = 20
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.device = "cpu"
        self.cfg = {}
        self.step_calls = 0
        self.reward_override = None
        self.timeouts = False
        self.observations = TensorDict(
            {
                "actor_obs": torch.arange(16, dtype=torch.float32).reshape(4, 4) / 20.0,
                "critic_obs": torch.arange(20, dtype=torch.float32).reshape(4, 5) / 20.0,
            },
            batch_size=[self.num_envs],
        )

    @property
    def unwrapped(self):
        return self

    def get_observations(self):
        return self.observations.clone()

    def step(self, actions):
        self.step_calls += 1
        self.episode_length_buf += 1
        reward = 1.0 - 0.1 * actions.square().sum(dim=-1)
        if self.reward_override is not None:
            reward = torch.full((self.num_envs,), self.reward_override)
        dones = torch.full((self.num_envs,), self.timeouts, dtype=torch.long)
        return self.get_observations(), reward, dones, {"time_outs": dones.bool()}


def _runner(env=None):
    """외부 logger 없이 실제 MLP, PPO, rollout storage와 Adam을 생성함."""
    cfg = {
        "num_steps_per_env": 2,
        "save_interval": 100,
        "check_for_nan": True,
        "obs_groups": {"actor": ["actor_obs"], "critic": ["critic_obs"]},
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [16, 8],
            "activation": "elu",
            "obs_normalization": False,
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0},
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [16, 8],
            "activation": "elu",
            "obs_normalization": False,
        },
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 1,
            "num_mini_batches": 1,
            "schedule": "fixed",
            "learning_rate": 1.0e-3,
            "rnd_cfg": None,
        },
    }
    return OnPolicyRunner(env or _CpuEnv(), cfg, log_dir=None, device="cpu")


def _parameters(runner):
    return {
        f"{name}.{key}": value.detach().clone()
        for name, model in (("actor", runner.alg.actor), ("critic", runner.alg.critic))
        for key, value in model.state_dict().items()
    }


def _report(log_dir):
    """최초 실패의 위치 정보와 읽을 수 있는 snapshot을 확인함."""
    report = json.loads((log_dir / "numerics" / "nonfinite.json").read_text())
    assert {"stage", "iteration", "rollout_step", "minibatch", "tensor", "indices", "axes"} <= report.keys()
    assert report["iteration"] == 0
    snapshot = torch.load(log_dir / "numerics" / "snapshot.pt", map_location="cpu", weights_only=False)
    assert snapshot["report"] == report
    assert not torch.isfinite(snapshot["failure_tensor"]).all()
    return report


@pytest.mark.parametrize("record_action_history", [False, True])
def test_finite_run_preserves_learning_rng_and_restores_methods(tmp_path, record_action_history):
    """정상 학습의 실제 가중치와 난수 결과를 보존하고 문맥 종료 후 메서드를 복원함."""
    torch.manual_seed(23)
    baseline = _runner()
    baseline.learn(2)
    expected_parameters = _parameters(baseline)
    expected_random = torch.rand(8)

    torch.manual_seed(23)
    runner = _runner()
    original_step = runner.env.step
    original_update = runner.alg.update
    with RslRlFiniteCheck(runner, str(tmp_path), record_action_history=record_action_history):
        runner.learn(2)
    torch.testing.assert_close(_parameters(runner), expected_parameters, rtol=0.0, atol=0.0)
    torch.testing.assert_close(torch.rand(8), expected_random, rtol=0.0, atol=0.0)
    assert runner.env.step == original_step
    assert runner.alg.update == original_update
    for filename in ("nonfinite.json", "snapshot.pt"):
        assert not (tmp_path / "numerics" / filename).exists()
    if record_action_history:
        report = json.loads((tmp_path / "numerics" / "action_history.json").read_text())
        assert report["reason"] == "training completed"
        assert (tmp_path / "numerics" / "action_history.pt").exists()
    else:
        assert not (tmp_path / "numerics" / "action_history.json").exists()
        assert not (tmp_path / "numerics" / "action_history.pt").exists()


def test_action_history_retains_ordered_immutable_frames_and_policy_versions_before_execution(tmp_path):
    """이동 기록의 순서·원본 값·정책 버전을 보존하고 임계 액션의 환경 실행을 차단함."""

    class MutableObservationEnv(_CpuEnv):
        def __init__(self):
            super().__init__()
            self.reset_terminated = torch.zeros(4, dtype=torch.bool)
            self.reset_time_outs = torch.zeros(4, dtype=torch.bool)

        def get_observations(self):
            return self.observations

        def step(self, actions):
            result = super().step(actions)
            self.reset_terminated.zero_()
            self.reset_time_outs.zero_()
            if self.step_calls == 3:
                # 실제 종료와 timeout의 동시 발생도 각각 보존해야 함.
                self.reset_terminated[3] = True
                self.reset_time_outs[3] = True
                result[2][3] = 1
                result[3]["time_outs"][3] = True
                self.episode_length_buf[3] = 0
            # 같은 관측 버퍼를 재사용해 과거 기록이 현재 상태에 덮어써지지 않는지 확인함.
            self.observations["actor_obs"].add_(0.125)
            self.observations["critic_obs"].add_(0.125)
            return result

    torch.manual_seed(31)
    runner = _runner(MutableObservationEnv())
    original_act = runner.alg.act
    expected_observations = []
    expected_actions = []
    expected_parameters = []

    def inject_threshold_action(observations):
        expected_observations.append(observations.clone())
        expected_parameters.append(_parameters(runner))
        actions = original_act(observations)
        if runner.env.step_calls == 4:
            # 환경 번호와 액션 성분을 다르게 해 저장 시 환경 선택 순서도 검증함.
            actions[1, 1] = -102.0
            actions[3, 0] = 101.0
        expected_actions.append(actions.clone())
        return actions

    runner.alg.act = inject_threshold_action
    with (
        pytest.raises(RuntimeError, match="Action history trigger"),
        RslRlFiniteCheck(
            runner, str(tmp_path), record_action_history=True, action_history_steps=3, action_history_threshold=100.0
        ),
    ):
        runner.learn(3)

    assert runner.env.step_calls == 4
    report = json.loads((tmp_path / "numerics" / "action_history.json").read_text())
    assert report["env_ids"] == [1, 3]
    assert report["threshold"] == 100.0
    assert report["threshold_target"] == "environment_action"
    assert (report["frame_count"], report["first_step"], report["last_step"]) == (3, 2, 4)
    history = torch.load(tmp_path / "numerics" / "action_history.pt", map_location="cpu", weights_only=False)
    assert history["report"] == report
    frames = history["frames"]
    assert [frame["absolute_step"] for frame in frames] == [2, 3, 4]
    assert [frame["policy_version"] for frame in frames] == [1, 1, 2]
    assert [frame["env_executed"] for frame in frames] == [True, True, False]
    assert [frame["env_step_started"] for frame in frames] == [True, True, False]
    assert frames[0]["data"]["terminated"].tolist() == [False, True]
    assert frames[0]["data"]["reset_time_outs"].tolist() == [False, True]
    assert frames[0]["data"]["episode_id_before"].tolist() == [0, 0]
    assert frames[1]["data"]["episode_id_before"].tolist() == [0, 1]
    versions = {version["version"]: version for version in history["policy_versions"]}
    for frame in frames:
        step = frame["absolute_step"]
        data = frame["data"]
        for group, observations in expected_observations[step].items():
            torch.testing.assert_close(data["observations"][group], observations[[1, 3]], rtol=0.0, atol=0.0)
        torch.testing.assert_close(data["raw_action"], expected_actions[step][[1, 3]], rtol=0.0, atol=0.0)
        assert {"action_mean", "action_std", "value", "actions_log_prob", "episode_length_before"} <= data.keys()
        assert frame["rng_state"]["cpu"].dtype == torch.uint8
        # 이후 실제 PPO 업데이트가 과거 정책 사본을 바꾸지 않았는지 확인함.
        algorithm = versions[frame["policy_version"]]["algorithm"]
        for name in ("actor", "critic"):
            for key, value in algorithm[f"{name}_state_dict"].items():
                torch.testing.assert_close(value, expected_parameters[step][f"{name}.{key}"], rtol=0.0, atol=0.0)
        if frame["env_executed"]:
            assert {"rewards", "dones", "time_outs", "episode_length_after", "bootstrapped_rewards"} <= data.keys()
        else:
            assert "rewards" not in data
    assert history["latest_update"]["iteration"] == 1
    assert {"algorithm_before", "rollout", "rng_state", "loss_after"} <= history["latest_update"].keys()
    assert not (tmp_path / "numerics" / "nonfinite.json").exists()


def test_nonfinite_failure_flushes_action_history_including_pending_observation(tmp_path):
    """기존 Inf 검출도 직전 기록과 아직 실행하지 않은 실패 관측을 함께 보존함."""
    runner = _runner()
    with (
        pytest.raises(FloatingPointError),
        RslRlFiniteCheck(runner, str(tmp_path), record_action_history=True),
        torch.inference_mode(),
    ):
        observations = runner.env.get_observations()
        actions = runner.alg.act(observations)
        observations, rewards, dones, extras = runner.env.step(actions)
        runner.alg.process_env_step(observations, rewards, dones, extras)
        observations["actor_obs"][2, 1] = torch.inf
        runner.alg.act(observations)
    assert runner.env.step_calls == 1
    finite_report = _report(tmp_path)
    assert finite_report["env_ids"] == [2]
    history = torch.load(tmp_path / "numerics" / "action_history.pt", map_location="cpu", weights_only=False)
    assert history["report"]["env_ids"] == [2]
    assert [frame["absolute_step"] for frame in history["frames"]] == [0, 1]
    assert [frame["env_executed"] for frame in history["frames"]] == [True, False]
    assert torch.isfinite(history["frames"][0]["data"]["observations"]["actor_obs"]).all()
    assert torch.isinf(history["frames"][1]["data"]["observations"]["actor_obs"][0, 1])


def test_action_history_preserves_applied_action_before_episode_reset(tmp_path):
    """보상 계산 뒤 환경이 액션을 초기화해도 실제 전달된 액션을 기록에 보존함."""

    class ActionResetEnv(_CpuEnv):
        def __init__(self):
            super().__init__()
            self.action_manager = SimpleNamespace(action=torch.zeros(4, 2))
            self.scene = SimpleNamespace(articulations={})
            self.reset_terminated = torch.zeros(4, dtype=torch.bool)
            self.reward_manager = SimpleNamespace(
                compute=lambda dt: torch.ones(4), _step_reward=torch.zeros(4, 1), active_terms=["constant"]
            )

        def step(self, actions):
            self.step_calls += 1
            self.episode_length_buf += 1
            self.action_manager.action.copy_(actions)
            rewards = self.reward_manager.compute(0.02)
            # 실제 환경과 같은 순서로 보상을 계산한 뒤 종료 환경의 액션 버퍼를 초기화함.
            self.reset_terminated.fill_(True)
            self.action_manager.action.zero_()
            self.episode_length_buf.zero_()
            return self.get_observations(), rewards, self.reset_terminated.long(), {"time_outs": torch.zeros(4).bool()}

    torch.manual_seed(47)
    runner = _runner(ActionResetEnv())
    with RslRlFiniteCheck(runner, str(tmp_path), record_action_history=True), torch.inference_mode():
        observations = runner.env.get_observations()
        actions = runner.alg.act(observations)
        expected = actions.clone()
        observations, rewards, dones, extras = runner.env.step(actions)
        runner.alg.process_env_step(observations, rewards, dones, extras)

    assert expected.abs().max() > 0.0
    assert torch.count_nonzero(runner.env.action_manager.action) == 0
    history = torch.load(tmp_path / "numerics" / "action_history.pt", map_location="cpu", weights_only=False)
    frame = history["frames"][0]
    assert frame["env_executed"]
    assert frame["data"]["dones"].bool().all()
    torch.testing.assert_close(frame["data"]["raw_action"], expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(frame["data"]["applied_action"], expected, rtol=0.0, atol=0.0)


def test_archived_diagnostics_import_and_detect_nonfinite_without_original_package_name(tmp_path, monkeypatch):
    """보관된 진단 소스를 별도 패키지로 읽어 실제 PPO의 Inf를 검출하고 이력을 저장함."""
    runner = _runner()
    with RslRlFiniteCheck(runner, str(tmp_path), record_action_history=True), torch.inference_mode():
        observations = runner.env.get_observations()
        actions = runner.alg.act(observations)
        observations, rewards, dones, extras = runner.env.step(actions)
        runner.alg.process_env_step(observations, rewards, dones, extras)

    archive = tmp_path / "numerics" / "diagnostics_source"
    assert (tmp_path / "numerics" / "finite_check_source.py").is_file()
    package_name = f"_archived_rsl_diagnostics_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(
        package_name, archive / "__init__.py", submodule_search_locations=[str(archive)]
    )
    assert spec is not None and spec.loader is not None
    package = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = package
    # 원래 모듈을 우연히 재사용해 보관 소스의 누락이나 절대 import를 숨기지 않도록 함.
    for name in tuple(sys.modules):
        if name == "isaaclab_rl.diagnostics" or name.startswith("isaaclab_rl.diagnostics."):
            monkeypatch.setitem(sys.modules, name, None)
    try:
        spec.loader.exec_module(package)
        archived_module = importlib.import_module(f"{package_name}.rsl_rl_finite_check")
        assert Path(archived_module.__file__).parent == archive
        archived_check = archived_module.RslRlFiniteCheck
        fresh_runner = _runner()
        fresh_runner.env.observations["actor_obs"][1, 0] = torch.inf
        replay_dir = tmp_path / "archived_run"
        with (
            pytest.raises(FloatingPointError),
            archived_check(fresh_runner, str(replay_dir), record_action_history=True),
        ):
            fresh_runner.learn(1)
        assert fresh_runner.env.step_calls == 0
        assert _report(replay_dir)["env_ids"] == [1]
        assert (replay_dir / "numerics" / "action_history.pt").is_file()
    finally:
        for name in tuple(sys.modules):
            if name == package_name or name.startswith(f"{package_name}."):
                del sys.modules[name]


def test_action_threshold_uses_clipped_environment_input_without_masking_infinity(tmp_path):
    """실제 wrapper의 제한 후 입력을 검사하고 제한으로 숨겨질 raw Inf는 먼저 차단함."""

    class CpuManagerEnv(ManagerBasedRLEnv):
        def __init__(self):
            cpu_env = _CpuEnv()
            self._is_closed = True
            self.cfg = SimpleNamespace(
                is_finite_horizon=False, episode_length_s=20.0, decimation=1, sim=SimpleNamespace(dt=0.02)
            )
            self.sim = SimpleNamespace(device="cpu")
            self.scene = SimpleNamespace(num_envs=4, articulations={})
            self.episode_length_buf = cpu_env.episode_length_buf
            self.obs_buf = cpu_env.observations
            self.action_manager = SimpleNamespace(total_action_dim=2, action=torch.zeros(4, 2))
            self.reward_manager = SimpleNamespace(
                compute=lambda dt: torch.ones(4), _step_reward=torch.zeros(4, 1), active_terms=["constant"]
            )
            self.step_calls = 0

        def reset(self, **kwargs):
            return self.obs_buf, {}

        def step(self, actions):
            self.step_calls += 1
            self.episode_length_buf += 1
            self.action_manager.action.copy_(actions)
            rewards = self.reward_manager.compute(0.02)
            self.obs_buf = self.obs_buf.clone()
            self.obs_buf["actor_obs"][:, -2:] = last_action(self)
            return self.obs_buf, rewards, torch.zeros(4).bool(), torch.zeros(4).bool(), {}

    torch.manual_seed(53)
    env = CpuManagerEnv()
    runner = _runner(RslRlVecEnvWrapper(env, clip_actions=10.0))
    original_base_step = env.step
    original_wrapper_step = runner.env.step
    with torch.no_grad():
        last_linear = [module for module in runner.alg.actor.modules() if isinstance(module, torch.nn.Linear)][-1]
        last_linear.weight.zero_()
        last_linear.bias.copy_(torch.tensor([-111.0, 111.0]))
    expected_input = torch.tensor([[-10.0, 10.0]]).repeat(4, 1)
    with RslRlFiniteCheck(runner, str(tmp_path), record_action_history=True), torch.inference_mode():
        actions = runner.alg.act(runner.env.get_observations())
        raw_actions = actions.clone()
        assert raw_actions.abs().min() > 100.0
        observations, rewards, dones, extras = runner.env.step(actions)
        runner.alg.process_env_step(observations, rewards, dones, extras)
        torch.testing.assert_close(observations["actor_obs"][:, -2:], expected_input, rtol=0.0, atol=0.0)
        torch.testing.assert_close(runner.alg.storage.actions[0], raw_actions, rtol=0.0, atol=0.0)
        # Inf는 wrapper가 유한한 10으로 바꾸기 전에 검사되어야 함.
        invalid_actions = raw_actions.clone()
        invalid_actions[2, 0] = torch.inf
        with pytest.raises(FloatingPointError):
            runner.env.step(invalid_actions)

    assert env.step_calls == 1
    assert env.step == original_base_step
    assert runner.env.step == original_wrapper_step
    torch.testing.assert_close(actions, raw_actions, rtol=0.0, atol=0.0)
    assert _report(tmp_path)["env_ids"] == [2]
    history = torch.load(tmp_path / "numerics" / "action_history.pt", map_location="cpu", weights_only=False)
    data = history["frames"][0]["data"]
    torch.testing.assert_close(data["raw_action"], raw_actions[2:3], rtol=0.0, atol=0.0)
    torch.testing.assert_close(data["environment_action"], expected_input[2:3], rtol=0.0, atol=0.0)
    torch.testing.assert_close(data["applied_action"], expected_input[2:3], rtol=0.0, atol=0.0)


@pytest.mark.parametrize("fault", ["observation", "actions"])
def test_environment_inf_stops_before_environment_step(tmp_path, fault):
    """입력 관측과 액션의 Inf가 환경 step에 도달하기 전에 중단되는지 검증함."""
    runner = _runner()
    with pytest.raises(FloatingPointError), RslRlFiniteCheck(runner, str(tmp_path)):
        if fault == "observation":
            runner.env.observations["actor_obs"][2, 1] = torch.inf
            runner.learn(1)
        else:
            actions = torch.zeros(4, 2)
            actions[2, 1] = torch.inf
            runner.env.step(actions)
    assert runner.env.step_calls == 0
    report = _report(tmp_path)
    assert report["env_ids"] == [2]
    assert [2, 1] in report["indices"]
    if fault == "observation":
        assert "input" in report["tensor"] or "observation" in report["tensor"]
    else:
        assert report["tensor"] == "environment.actions"


@pytest.mark.parametrize("fault", ["bootstrap", "returns"])
def test_finite_rollout_overflow_is_caught_before_update(tmp_path, fault):
    """유한한 입력의 timeout 보정과 GAE 계산에서 발생한 Inf를 각각 업데이트 전에 검출함."""
    runner = _runner()
    runner.env.reward_override = 3.0e38
    runner.env.timeouts = fault == "bootstrap"
    with torch.no_grad():
        for parameter in runner.alg.critic.parameters():
            parameter.zero_()
        last_linear = [module for module in runner.alg.critic.modules() if isinstance(module, torch.nn.Linear)][-1]
        last_linear.bias.fill_(3.0e38 if fault == "bootstrap" else 0.0)
    before = _parameters(runner)
    with pytest.raises(FloatingPointError), RslRlFiniteCheck(runner, str(tmp_path)):
        runner.learn(1)
    torch.testing.assert_close(_parameters(runner), before, rtol=0.0, atol=0.0)
    report = _report(tmp_path)
    assert report["tensor"] == ("bootstrap.rewards" if fault == "bootstrap" else "targets.returns")
    assert "env" in report["axes"]


@pytest.mark.parametrize("gradient_value", [float("inf"), 1.0e30])
def test_invalid_raw_gradient_or_norm_stops_before_optimizer_step(tmp_path, gradient_value):
    """실제 역전파의 원소 Inf와 유한 원소의 norm overflow를 clipping 전에 차단함."""
    runner = _runner()
    parameter = next(runner.alg.critic.parameters())
    fault = parameter.register_hook(lambda gradient: torch.full_like(gradient, gradient_value))
    steps = []
    step_hook = runner.alg.optimizer.register_step_post_hook(lambda *_: steps.append(True))
    before = _parameters(runner)
    try:
        with pytest.raises(FloatingPointError), RslRlFiniteCheck(runner, str(tmp_path)):
            runner.learn(1)
    finally:
        fault.remove()
        step_hook.remove()
    assert not steps
    torch.testing.assert_close(_parameters(runner), before, rtol=0.0, atol=0.0)
    report = _report(tmp_path)
    assert report["stage"] == "update"
    assert "critic.gradient" in report["tensor"]
    if gradient_value == 1.0e30:
        assert report["tensor"] == "critic.gradient_norm_before_clip"


def test_optimizer_corruption_is_caught_while_cached_distribution_std_is_finite(tmp_path):
    """Adam 갱신 직후 raw std의 NaN을 이전 forward의 유한한 std 캐시와 구분함."""
    runner = _runner()
    actor = runner.alg.actor
    steps = []

    def corrupt_std(*_):
        steps.append(True)
        with torch.no_grad():
            actor.distribution.std_param[0] = torch.nan
        assert torch.isfinite(actor.output_std).all()

    fault = runner.alg.optimizer.register_step_post_hook(corrupt_std)
    try:
        with pytest.raises(FloatingPointError), RslRlFiniteCheck(runner, str(tmp_path)):
            runner.learn(1)
    finally:
        fault.remove()
    assert steps == [True]
    assert torch.isfinite(actor.output_std).all()
    report = _report(tmp_path)
    assert report["tensor"] == "actor.state.distribution.std_param"
    assert report["stage"] == "update"


def test_raw_physics_inf_is_caught_before_reward_and_reset_can_hide_it(tmp_path):
    """정상 관측으로 교체하는 reset 전에 물리 상태의 Inf를 env ID와 함께 포착함."""

    class ResetEnv(_CpuEnv):
        def __init__(self):
            super().__init__()
            self.reward_calls = 0
            self.reset_called = False
            self.velocity = torch.zeros(4, 6)

            def proxy(tensor):
                return SimpleNamespace(torch=tensor)

            asset = SimpleNamespace(
                data=SimpleNamespace(
                    root_link_pose_w=proxy(torch.zeros(4, 7)),
                    root_com_vel_w=proxy(self.velocity),
                    joint_pos=proxy(torch.zeros(4, 2)),
                    joint_vel=proxy(torch.zeros(4, 2)),
                ),
                actuators=SimpleNamespace(
                    computed_effort=proxy(torch.zeros(4, 2)), applied_effort=proxy(torch.zeros(4, 2))
                ),
            )
            self.scene = SimpleNamespace(articulations={"robot": asset})
            self.reward_manager = SimpleNamespace(compute=self.compute_reward, _step_reward=torch.zeros(4, 1))

        def compute_reward(self, dt):
            self.reward_calls += 1
            return torch.ones(4)

        def step(self, actions):
            self.step_calls += 1
            self.velocity[1, 0] = torch.inf
            reward = self.reward_manager.compute(0.02)
            self.reset_called = True
            self.velocity.zero_()
            return self.get_observations(), reward, torch.zeros(4, dtype=torch.long), {}

    control = ResetEnv()
    assert torch.isfinite(control.step(torch.zeros(4, 2))[0]["actor_obs"]).all()
    assert control.reset_called and control.reward_calls == 1

    runner = _runner(ResetEnv())
    with pytest.raises(FloatingPointError), RslRlFiniteCheck(runner, str(tmp_path)):
        runner.env.step(torch.zeros(4, 2))
    assert not runner.env.reset_called
    assert runner.env.reward_calls == 0
    report = _report(tmp_path)
    assert report["tensor"] == "physics.robot.root_com_vel_w"
    assert report["env_ids"] == [1]


def test_snapshot_probe_preserves_training_history_and_checks_nonfinite(tmp_path):
    """Probe steps cannot overwrite a training frame or bypass finite checks."""
    runner = _runner()
    with RslRlFiniteCheck(runner, str(tmp_path), record_action_history=True) as check:
        with torch.inference_mode():
            obs = runner.env.get_observations()
            actions = runner.alg.act(obs)
            runner.env.reward_override = 1.0
            runner.env.step(actions)
            with check.snapshot_probe():
                runner.env.reward_override = 9.0
                runner.env.step(actions)
            actions = runner.alg.act(obs)
            runner.env.reward_override = 1.0
            runner.env.step(actions)
            with pytest.raises(FloatingPointError), check.snapshot_probe():
                runner.env.reward_override = float("inf")
                runner.env.step(actions)
    history = torch.load(tmp_path / "numerics" / "action_history.pt", map_location="cpu", weights_only=False)
    assert len(history["frames"]) == 2
    for frame in history["frames"]:
        torch.testing.assert_close(frame["data"]["rewards"], torch.ones(runner.env.num_envs))
    assert (history["frames"][0]["data"]["episode_id_before"] == 0).all()
    assert (history["frames"][1]["data"]["episode_id_before"] == 1).all()
    assert _report(tmp_path)["tensor"] == "environment.outputs.rewards"
