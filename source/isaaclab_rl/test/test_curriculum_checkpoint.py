# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Checkpoint metadata and lifetime contracts for environment curriculum state."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.utils.curriculum_checkpoint import RslRlCurriculumCheckpoint

pytestmark = pytest.mark.unit


@pytest.fixture
def runner():
    runner = OnPolicyRunner.__new__(OnPolicyRunner)
    manager = SimpleNamespace(
        state_dict=Mock(return_value={"lp": {"probabilities": torch.tensor([0.25, 0.75])}}),
        load_state_dict=Mock(),
    )
    runner.env = SimpleNamespace(unwrapped=SimpleNamespace(curriculum_manager=manager))
    runner.alg = SimpleNamespace(save=lambda: {"policy": torch.tensor([1.0])}, load=lambda *args: True)
    runner.logger = SimpleNamespace(save_model=Mock())
    runner.current_learning_iteration = 7
    return runner


@pytest.fixture
def snapshot_runner(runner):
    runner.env.clip_actions = 1.0
    runner.env.unwrapped.curriculum_manager._class_term_cfgs = [
        SimpleNamespace(func=SimpleNamespace(snapshot_enabled=True))
    ]
    return runner


def test_curriculum_is_uploaded_with_checkpoint_and_restored_without_losing_infos(runner, tmp_path):
    """Use the installed runner's actual torch serialization and upload call."""
    path = str(tmp_path / "model.pt")
    caller_infos = {"experiment": "rough"}
    with RslRlCurriculumCheckpoint(runner):
        runner.save(path, infos=caller_infos)
        saved = torch.load(path, weights_only=False)
        assert saved["infos"]["experiment"] == "rough"
        assert "isaaclab_curriculum_runtime_contract" not in saved["infos"]
        torch.testing.assert_close(
            saved["infos"]["isaaclab_curriculum"]["lp"]["probabilities"], torch.tensor([0.25, 0.75])
        )
        assert saved["iter"] == 7
        runner.logger.save_model.assert_called_once_with(path, 7)
        restored = runner.load(path)
        runner.env.unwrapped.curriculum_manager.load_state_dict.assert_called_once_with(restored["isaaclab_curriculum"])
    assert caller_infos == {"experiment": "rough"}
    assert "save" not in runner.__dict__ and "load" not in runner.__dict__


def test_snapshot_checkpoint_persists_action_clipping_and_restores_matching_state(snapshot_runner, tmp_path):
    path = str(tmp_path / "snapshot.pt")
    with RslRlCurriculumCheckpoint(snapshot_runner):
        snapshot_runner.save(path)
        saved = torch.load(path, weights_only=False)
        assert saved["infos"]["isaaclab_curriculum_runtime_contract"] == {"clip_actions": 1.0}
        restored = snapshot_runner.load(path)
        snapshot_runner.env.unwrapped.curriculum_manager.load_state_dict.assert_called_once_with(
            restored["isaaclab_curriculum"]
        )


def test_online_window_checkpoint_checks_action_clipping_before_restoring_state(runner, tmp_path):
    runner.env.clip_actions = 1.0
    manager = runner.env.unwrapped.curriculum_manager
    manager._class_term_cfgs = [SimpleNamespace(func=SimpleNamespace(online_window_enabled=True))]
    path = str(tmp_path / "online_window.pt")
    with RslRlCurriculumCheckpoint(runner):
        runner.save(path)
        saved = torch.load(path, weights_only=False)
        assert saved["infos"]["isaaclab_curriculum_runtime_contract"] == {"clip_actions": 1.0}
        restored = runner.load(path)
        manager.load_state_dict.assert_called_once_with(restored["isaaclab_curriculum"])
        manager.load_state_dict.reset_mock()
        runner.env.clip_actions = 2.0
        with pytest.raises(ValueError, match="--reset_curriculum"):
            runner.load(path)
        manager.load_state_dict.assert_not_called()


@pytest.mark.parametrize("incompatibility", ["changed_clipping", "missing_contract"])
def test_snapshot_checkpoint_rejects_incompatible_runtime_before_restoring_curriculum(
    snapshot_runner, tmp_path, incompatibility
):
    path = str(tmp_path / "snapshot.pt")
    with RslRlCurriculumCheckpoint(snapshot_runner):
        snapshot_runner.save(path)
        if incompatibility == "changed_clipping":
            snapshot_runner.env.clip_actions = 2.0
        else:
            saved = torch.load(path, weights_only=False)
            saved["infos"].pop("isaaclab_curriculum_runtime_contract", None)
            torch.save(saved, path)
        with pytest.raises(ValueError, match="--reset_curriculum"):
            snapshot_runner.load(path)
        snapshot_runner.env.unwrapped.curriculum_manager.load_state_dict.assert_not_called()


def test_old_checkpoint_starts_fresh_and_methods_are_restored_on_error(runner, tmp_path, caplog):
    path = str(tmp_path / "old.pt")
    runner.save(path)
    with pytest.raises(RuntimeError, match="training stopped"), RslRlCurriculumCheckpoint(runner):
        assert runner.load(path) is None
        assert "no curriculum state" in caplog.text
        runner.env.unwrapped.curriculum_manager.load_state_dict.assert_not_called()
        raise RuntimeError("training stopped")
    assert "save" not in runner.__dict__ and "load" not in runner.__dict__


def test_reset_curriculum_loads_policy_training_state_and_saves_fresh_statistics(snapshot_runner, tmp_path, caplog):
    """Resetting curriculum must not skip the installed runner's model, optimizer, or iteration restoration."""
    runner = snapshot_runner
    path = str(tmp_path / "previous.pt")
    runner.alg.save = lambda: {"policy": torch.tensor([1.0]), "optimizer": {"step": 12}}
    runner.alg.load = Mock(return_value=True)
    with RslRlCurriculumCheckpoint(runner):
        runner.save(path, infos={"experiment": "rough"})
    manager = runner.env.unwrapped.curriculum_manager
    manager.state_dict.return_value = {"lp": {"probabilities": torch.tensor([0.5, 0.5])}}
    manager._class_term_cfgs.append(SimpleNamespace(func=SimpleNamespace(reset_on_checkpoint_load=True)))
    runner.env.reset = Mock()
    runner.env.clip_actions = 2.0
    runner.current_learning_iteration = 0
    load_cfg = {"optimizer": True}
    with RslRlCurriculumCheckpoint(runner, restore_curriculum=False):
        infos = runner.load(path, load_cfg=load_cfg)
        runner.alg.load.assert_called_once()
        loaded, forwarded_cfg, strict = runner.alg.load.call_args.args
        torch.testing.assert_close(loaded["policy"], torch.tensor([1.0]))
        assert loaded["optimizer"] == {"step": 12}
        assert forwarded_cfg is load_cfg and strict is True
        assert runner.current_learning_iteration == 7
        assert infos["experiment"] == "rough"
        manager.load_state_dict.assert_not_called()
        assert "Curriculum reset requested" in caplog.text
        runner.env.reset.assert_not_called()
        fresh_path = str(tmp_path / "fresh.pt")
        runner.save(fresh_path)
        saved = torch.load(fresh_path, weights_only=False)
        assert saved["infos"]["isaaclab_curriculum_runtime_contract"] == {"clip_actions": 2.0}
        torch.testing.assert_close(
            saved["infos"]["isaaclab_curriculum"]["lp"]["probabilities"], torch.tensor([0.5, 0.5])
        )
    assert "save" not in runner.__dict__ and "load" not in runner.__dict__


def test_restored_terrain_resets_robot_after_all_manager_state_loads(runner, tmp_path):
    """The physical reset must use restored terrain placement, not the pre-load tile."""
    manager = runner.env.unwrapped.curriculum_manager
    manager._class_term_cfgs = [SimpleNamespace(func=SimpleNamespace(reset_on_checkpoint_load=True))]
    placement = {"terrain_origin": torch.tensor([8.0, 2.0, 1.0]), "history": torch.tensor([-1, 1])}
    physical_origin = torch.zeros(3)
    events = []
    manager.state_dict = lambda: {"terrain": {key: value.clone() for key, value in placement.items()}}

    def restore(state):
        placement.update(state["terrain"])
        events.append("manager restored")

    def reset():
        events.append("physical reset")
        physical_origin.copy_(placement["terrain_origin"])
        assert placement["history"].tolist() == [-1, 1]

    manager.load_state_dict = restore
    runner.env.reset = reset
    path = str(tmp_path / "terrain.pt")
    with RslRlCurriculumCheckpoint(runner):
        runner.save(path)
        placement["terrain_origin"] = torch.zeros(3)
        placement["history"] = torch.tensor([-1, -1])
        runner.load(path)
    torch.testing.assert_close(physical_origin, torch.tensor([8.0, 2.0, 1.0]))
    assert events == ["manager restored", "physical reset"]
