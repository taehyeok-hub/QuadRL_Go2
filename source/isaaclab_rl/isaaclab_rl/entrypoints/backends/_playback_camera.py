# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot-follow camera for interactive policy playback."""

from isaaclab.visualizers import add_key_callbacks


class PlaybackCamera:
    """Toggle robot tracking with Y, leaving the viewer camera free when disabled."""

    def __init__(self, env, *, follow: bool = True) -> None:
        self._visualizers = tuple(env.sim.visualizers)
        self._follow = follow
        self._closed = False
        self._views = [(viz, tuple(viz.cfg.eye), tuple(viz.cfg.lookat)) for viz in self._visualizers]
        self._kit_origins = []
        for viz in self._visualizers:
            backend = viz.cfg.backend
            if backend.visualizer_type == "kit":
                self._kit_origins.append((backend, backend.origin_type))
                # Playback owns tracking until close; do not reposition the free camera.
                backend.origin_type = "world"
        self._asset = None
        if self._visualizers:
            asset_name = env.cfg.commands.base_velocity.asset_name
            self._asset = env.scene[asset_name]
        self._remove_keys = add_key_callbacks(self._visualizers, {"Y": self.toggle})
        self.update()

    def toggle(self) -> None:
        """Switch between tracking and free camera navigation."""
        if not self._closed:
            self._follow = not self._follow

    def update(self) -> None:
        """Move every viewer with the controlled robot in environment zero."""
        if self._closed or not self._follow or self._asset is None:
            return
        position = self._asset.data.root_pos_w.torch[0].detach().cpu().tolist()
        for viz, eye, target in self._views:
            viz.set_camera_view(
                tuple(value + offset for value, offset in zip(position, eye)),
                tuple(value + offset for value, offset in zip(position, target)),
            )

    def close(self) -> None:
        """Remove keyboard callbacks and restore viewer tracking configuration."""
        if self._closed:
            return
        self._closed = True
        self._remove_keys()
        for backend, origin_type in self._kit_origins:
            backend.origin_type = origin_type
