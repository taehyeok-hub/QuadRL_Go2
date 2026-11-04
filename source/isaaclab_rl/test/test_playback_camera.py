# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot-follow camera behavior at playback's viewer boundary."""

import gc
import weakref
from types import SimpleNamespace

import torch

from isaaclab.visualizers import KeyEventSource

from isaaclab_rl.entrypoints.backends._playback_camera import PlaybackCamera


def test_playback_camera_tracks_robot_without_commands_and_releases_keyboard():
    """An explicit robot supports tracking and free navigation without command managers."""
    source = KeyEventSource()
    backend = SimpleNamespace(visualizer_type="kit", origin_type="asset")
    views = []
    cfg = SimpleNamespace(eye=(4.0, -3.0, 2.0), lookat=(0.0, 0.0, 0.5), backend=backend)

    def set_camera_view(eye, target):
        views.append((eye, target))
        # Newton records each applied world pose in its config.
        cfg.eye, cfg.lookat = eye, target

    viz = SimpleNamespace(
        cfg=cfg,
        key_event_source=source,
        set_camera_view=set_camera_view,
    )
    position = torch.tensor([[1.0, 2.0, 0.5]])
    env = SimpleNamespace(
        sim=SimpleNamespace(visualizers=[viz]),
        scene={"controlled_robot": SimpleNamespace(data=SimpleNamespace(root_pos_w=SimpleNamespace(torch=position)))},
    )
    camera = PlaybackCamera(env, asset_name="controlled_robot")
    assert views[-1] == ((5.0, -1.0, 2.5), (1.0, 2.0, 1.0))
    assert backend.origin_type == "world"

    position[0] = torch.tensor([2.0, 4.0, 0.75])
    camera.update()
    assert views[-1] == ((6.0, 1.0, 2.75), (2.0, 4.0, 1.25))

    source._emit_key("KeyY", True)
    source._emit_key("KeyY", True)
    position[0] = torch.tensor([4.0, 6.0, 1.0])
    camera.update()
    assert len(views) == 2
    source._emit_key("KeyY", False)
    source._emit_key("KeyY", True)
    camera.update()
    assert views[-1] == ((8.0, 3.0, 3.0), (4.0, 6.0, 1.5))

    camera.close()
    assert backend.origin_type == "asset"
    camera_ref = weakref.ref(camera)
    del camera
    gc.collect()
    assert camera_ref() is None
