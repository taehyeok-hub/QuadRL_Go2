# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Incomplete training must not become a completed comparison result."""

import os

import pytest
import torch

from isaaclab_rl.evaluation.checkpoints import evaluation_checkpoint
from isaaclab_rl.utils.training_interrupt import training_interrupt_exit

pytestmark = pytest.mark.unit


def test_latest_selects_numeric_checkpoint_despite_modified_old_file(tmp_path):
    run = tmp_path / "2026-11-16_10-00-00"
    run.mkdir()
    torch.save({"iter": 0}, run / "model_0.pt")
    torch.save({"iter": 9999}, run / "model_9999.pt")
    os.utime(run / "model_0.pt", (2_000_000_000, 2_000_000_000))
    assert evaluation_checkpoint("latest", tmp_path, 10000).name == "model_9999.pt"


def test_interrupted_newer_run_is_rejected_instead_of_using_older_complete_run(tmp_path):
    old = tmp_path / "2026-11-16_10-00-00"
    new = tmp_path / "2026-11-16_11-00-00"
    old.mkdir()
    new.mkdir()
    torch.save({"iter": 9999}, old / "model_9999.pt")
    torch.save({"iter": 100}, new / "model_100.pt")
    with pytest.raises(ValueError, match="101 iterations; expected 10000"):
        evaluation_checkpoint("latest", tmp_path, 10000)
    (new / "model_100.pt").unlink()
    with pytest.raises(FileNotFoundError, match="Newest run has no checkpoint"):
        evaluation_checkpoint("latest", tmp_path, 10000)


def test_ctrl_c_keeps_nonzero_status_and_runs_cleanup():
    cleanup = []
    with pytest.raises(SystemExit) as error:
        with training_interrupt_exit():
            try:
                raise KeyboardInterrupt
            finally:
                cleanup.append(True)
    assert error.value.code == 130
    assert cleanup == [True]
    with training_interrupt_exit():
        pass
