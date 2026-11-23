# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record a grid of live training robots grouped by their assigned terrain."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from isaaclab.envs.utils.video_recorder import VideoRecorder
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

logger = logging.getLogger(__name__)


class TerrainVideoRecorder(VideoRecorder):
    """Record four environments per terrain using the Newton GL perspective camera."""

    def __init__(
        self,
        cfg: VideoRecorderCfg,
        env: "ManagerBasedRLEnv",
        terrain_names: list[str],
        terrain_columns: dict[str, list[int]],
    ):
        super().__init__(cfg, env)
        self._visualizer = next(viz for viz in env.sim.visualizers if viz.cfg.backend.visualizer_type == "newton_gl")
        self._render = self._capture
        self._capture = self._capture_grid
        self._terrain_names = terrain_names
        self._terrain_columns = terrain_columns
        self._selected_envs: list[int] = []

    def _capture_grid(self) -> np.ndarray | None:
        if not self._frames:
            # A curriculum can reassign environments, so sample the current assignments for each clip.
            assignments = self._env.scene.terrain.terrain_types.cpu().numpy()
            self._selected_envs = []
            for name in self._terrain_names:
                candidates = np.flatnonzero(np.isin(assignments, self._terrain_columns[name]))
                if len(candidates) < 4:
                    logger.warning(
                        "Skipping terrain video: %s has %d assigned environments; need 4.", name, len(candidates)
                    )
                    self._recording = False
                    return None
                # Spread the views across the available environments without touching training RNGs.
                self._selected_envs.extend(candidates[np.linspace(0, len(candidates) - 1, 4, dtype=int)].tolist())

        positions = self._env.scene["robot"].data.root_pos_w.torch[self._selected_envs].cpu().numpy()
        visualizer = self._visualizer
        eye, target = visualizer.cfg.eye, visualizer.cfg.lookat
        visible_envs = visualizer.get_visualized_env_ids()
        tiles = []
        try:
            # Apply the same selection to GL geometry and to Isaac Lab overlays.
            visualizer._viewer.set_visible_worlds(self._selected_envs)
            visualizer._env_ids = self._selected_envs
            for position in positions:
                visualizer.set_camera_view(position + (2.8, -2.4, 1.9), position + (0.0, 0.0, -0.15))
                frame = self._render()
                stride = max(1, frame.shape[1] // 384)
                tiles.append(frame[::stride, ::stride].copy())
        finally:
            visualizer.set_camera_view(eye, target)
            visualizer._env_ids = visible_envs
            visualizer._viewer.set_visible_worlds(visible_envs)
        if len(self._terrain_names) == 1:
            return np.concatenate((np.concatenate(tiles[:2], axis=1), np.concatenate(tiles[2:], axis=1)), axis=0)
        top_row = np.concatenate(tiles[:4], axis=1)
        bottom_row = np.concatenate(tiles[4:], axis=1)
        return np.concatenate((top_row, bottom_row), axis=0)


def route_terrain_videos_to_wandb(training_logger, terrain_by_directory: dict[str, str]) -> None:
    """Keep each terrain in its own W&B series while preserving ordinary video uploads."""
    import wandb

    directories = {Path(directory).resolve(): name for directory, name in terrain_by_directory.items()}
    init_writer = training_logger.init_logging_writer

    def init_terrain_writer():
        init_writer()
        writer = training_logger.writer
        if writer is None:
            return
        save_video = writer.save_video
        uploaded = set()

        def save_terrain_video(video: Path, iteration: int):
            terrain = directories.get(video.parent.resolve())
            if terrain is None:
                return save_video(video, iteration)
            if video not in uploaded:
                wandb.log({f"TerrainVideo/{terrain}": wandb.Video(str(video), format="mp4")}, step=iteration)
                uploaded.add(video)

        writer.save_video = save_terrain_video

    training_logger.init_logging_writer = init_terrain_writer
