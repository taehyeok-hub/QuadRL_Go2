# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Online curriculum notifications at the installed RSL-RL PPO update boundary."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from isaaclab_rl.utils.online_curriculum import RslRlOnlineCurriculum
from isaaclab_rl.utils.snapshot_curriculum import RslRlSnapshotCurriculum

pytestmark = pytest.mark.unit


class OnlineTerm:
    """Record the controller's public notifications without implementing window sampling."""

    online_window_enabled = True
    online_controller_active = False
    training_policy_updates = None

    def __init__(self):
        self.starts = []
        self.updates = []
        self.fail_callback = False

    def begin_online_window(self, completed_updates):
        self.starts.append(completed_updates)
        self.training_policy_updates = completed_updates

    def complete_online_update(self, completed_updates):
        if self.fail_callback:
            raise RuntimeError("curriculum callback failed")
        assert self.online_controller_active
        assert completed_updates == self.training_policy_updates + 1
        self.updates.append(completed_updates)
        self.training_policy_updates = completed_updates


class OnlineEnv:
    num_envs = 2
    num_actions = 1
    device = "cpu"

    def __init__(self):
        self.unwrapped = self
        self.term = OnlineTerm()
        self.curriculum_manager = SimpleNamespace(_class_term_cfgs=[SimpleNamespace(func=self.term)])
        self.steps = 0
        self.obs = torch.zeros(self.num_envs, 2)
        self.reset = Mock(side_effect=AssertionError("online windows must not reset the environment"))

    def get_observations(self):
        return TensorDict({"policy": self.obs.clone()}, batch_size=[self.num_envs])

    def step(self, actions):
        assert self.term.online_controller_active
        self.steps += 1
        self.obs.add_(0.001)
        return self.get_observations(), torch.ones(self.num_envs), torch.zeros(self.num_envs), {"log": {}}


def _make_runner():
    env = OnlineEnv()
    obs = env.get_observations()
    actor = MLPModel(
        obs,
        {"actor": ["policy"]},
        "actor",
        1,
        hidden_dims=(4,),
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.5},
    )
    critic = MLPModel(obs, {"critic": ["policy"]}, "critic", 1, hidden_dims=(4,))
    runner = OnPolicyRunner.__new__(OnPolicyRunner)
    runner.env = env
    runner.alg = PPO(
        actor, critic, RolloutStorage("rl", env.num_envs, 2, obs, [1]), num_learning_epochs=1, num_mini_batches=1
    )
    runner.device = "cpu"
    runner.is_distributed = False
    runner.current_learning_iteration = 0
    runner.cfg = {"num_steps_per_env": 2, "algorithm": {}, "save_interval": 100}
    runner.logger = SimpleNamespace(
        init_logging_writer=Mock(), process_env_step=Mock(), log=Mock(), writer=None, save_model=Mock()
    )
    return runner


def test_actual_ppo_notifies_every_completed_update_without_extra_rollouts_or_initial_measurement():
    runner = _make_runner()
    term = runner.env.term
    actor = runner.alg.get_policy()
    actor_state = {key: value.clone() for key, value in actor.state_dict().items()}
    rng = torch.get_rng_state()
    original_compute_returns = runner.alg.compute_returns
    with RslRlOnlineCurriculum(runner), RslRlSnapshotCurriculum(runner):
        assert term.starts == [0] and term.updates == []
        assert runner.env.steps == 0 and runner.alg.storage.step == 0
        assert runner.alg.get_policy() is actor and actor.training
        assert torch.equal(torch.get_rng_state(), rng)
        for key, value in actor_state.items():
            torch.testing.assert_close(actor.state_dict()[key], value, rtol=0, atol=0)
        assert runner.alg.compute_returns == original_compute_returns
        runner.learn(101)
    assert term.updates == list(range(1, 102))
    assert term.training_policy_updates == 101
    assert runner.env.steps == runner.logger.process_env_step.call_count == 202
    runner.env.reset.assert_not_called()
    assert runner.alg.storage.step == 0
    assert not term.online_controller_active and "update" not in vars(runner.alg)


@pytest.mark.parametrize("failure", ["ppo", "storage", "callback"])
def test_failed_update_does_not_commit_a_completed_curriculum_event_and_restores_hook(failure):
    runner = _make_runner()
    term = runner.env.term
    if failure == "ppo":
        runner.alg.update = Mock(side_effect=RuntimeError("PPO failed"))
        message = "PPO failed"
    elif failure == "storage":
        runner.alg.update = Mock(return_value={})
        message = "clears rollout storage"
    else:
        term.fail_callback = True
        message = "curriculum callback failed"
    original_update = runner.alg.update
    with pytest.raises(RuntimeError, match=message):
        with RslRlOnlineCurriculum(runner):
            runner.learn(1)
    assert term.training_policy_updates == 0 and term.updates == []
    assert not term.online_controller_active
    assert runner.alg.update == original_update
    runner.env.reset.assert_not_called()


def test_persisted_update_count_survives_two_actual_runner_resumes(tmp_path):
    runner = _make_runner()
    with RslRlOnlineCurriculum(runner):
        runner.learn(3)
    assert runner.current_learning_iteration == 2
    for completed in (4, 5):
        path = str(tmp_path / f"resume_{completed}.pt")
        runner.save(path, infos={"completed_updates": runner.env.term.training_policy_updates})
        runner = _make_runner()
        infos = runner.load(path)
        runner.env.term.training_policy_updates = infos["completed_updates"]
        with RslRlOnlineCurriculum(runner, resumed=True):
            assert runner.env.term.starts == [completed - 1]
            runner.learn(1)
        assert runner.current_learning_iteration == 2
        assert runner.env.term.updates == [completed]
        assert runner.env.term.training_policy_updates == completed


def test_legacy_resume_uses_completed_count_and_begin_failure_restores_activation():
    runner = _make_runner()
    runner.current_learning_iteration = 136
    term = runner.env.term
    with RslRlOnlineCurriculum(runner, resumed=True):
        assert term.starts == [137]
    term.online_controller_active = True
    term.begin_online_window = Mock(side_effect=ValueError("invalid restored window"))
    original_update = runner.alg.update
    with pytest.raises(ValueError, match="invalid restored window"):
        with RslRlOnlineCurriculum(runner, resumed=True):
            pass
    assert term.online_controller_active
    assert runner.alg.update == original_update


def test_other_measurement_modes_are_noop_and_unsupported_runners_fail_before_activation():
    runner = _make_runner()
    term = runner.env.term
    term.online_window_enabled = False
    original_update = runner.alg.update
    rng = torch.get_rng_state()
    with RslRlOnlineCurriculum(runner):
        assert runner.alg.update == original_update
        assert term.starts == [] and not term.online_controller_active
        assert torch.equal(torch.get_rng_state(), rng)
    term.online_window_enabled = True
    runner.is_distributed = True
    with pytest.raises(ValueError, match="single-device"):
        RslRlOnlineCurriculum(runner)
    assert not term.online_controller_active
