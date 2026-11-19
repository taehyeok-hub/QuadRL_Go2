# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Body stability, contact occupancy, and completed-swing statistics in physical units."""

import math

import torch


class GaitMetrics:
    """Keep continuous swing state across logging windows, but never across resets."""

    def __init__(self, num_envs: int, foot_names: list[str], device: str, *, debug_metrics: bool = False):
        self._foot_names = foot_names
        self._debug = debug_metrics
        shape = (num_envs, len(foot_names))
        self._contact = torch.zeros(shape, dtype=torch.bool, device=device)
        self._active = torch.zeros_like(self._contact)
        self._peak_valid = torch.zeros_like(self._contact)
        self._peak = torch.zeros(shape, device=device)
        # Body rows: signed vz, squared vz, squared roll, squared pitch; sum and valid time.
        self._body = torch.zeros(2, 4, dtype=torch.float64, device=device)
        # Foot rows: height, peak, duration, contact occupancy, horizontal stance speed.
        self._feet = torch.zeros(2, 5, len(foot_names), dtype=torch.float64, device=device)
        self._airborne_time = torch.zeros(len(foot_names), dtype=torch.float64, device=device)

    def reset(self, env_ids):
        """Discard incomplete swings for reset environments, retaining already observed statistics."""
        self._contact[env_ids] = False
        self._active[env_ids] = False
        self._peak_valid[env_ids] = False
        self._peak[env_ids] = 0

    def sample(
        self,
        vz: torch.Tensor,
        roll_pitch: torch.Tensor,
        contact: torch.Tensor,
        height: torch.Tensor,
        height_valid: torch.Tensor,
        foot_speed: torch.Tensor,
        last_air_time: torch.Tensor,
        dt: float,
    ):
        """Sample world vertical speed, angles [rad], and foot-link heights/speeds [m, m/s]."""
        body = torch.cat((vz[:, None], vz.square()[:, None], roll_pitch.square()), dim=1)
        valid_body = torch.isfinite(body)
        self._body[0] += torch.where(valid_body, body, 0).double().sum(dim=0) * dt
        self._body[1] += valid_body.sum(dim=0) * dt
        airborne = ~contact
        valid_height = height_valid & torch.isfinite(height)

        # A complete swing starts with observed stance and finishes with observed touchdown.
        liftoff = self._contact & airborne
        touchdown = self._active & contact
        self._active |= liftoff
        self._peak = torch.where(liftoff, -torch.inf, self._peak)
        self._peak_valid = torch.where(liftoff, True, self._peak_valid)
        self._peak_valid &= ~self._active | ~airborne | valid_height
        self._peak = torch.where(self._active & airborne & valid_height, torch.maximum(self._peak, height), self._peak)

        values = (
            (height, airborne & valid_height, dt),
            (self._peak, touchdown & self._peak_valid & torch.isfinite(self._peak), 1.0),
            (last_air_time, touchdown & torch.isfinite(last_air_time) & (last_air_time > 0), 1.0),
            (contact.float(), torch.ones_like(contact), dt),
            (foot_speed, contact & torch.isfinite(foot_speed), dt),
        )
        for index, (value, valid, weight) in enumerate(values):
            self._feet[0, index] += torch.where(valid, value, 0).double().sum(dim=0) * weight
            self._feet[1, index] += valid.sum(dim=0) * weight
        self._airborne_time += airborne.sum(dim=0) * dt
        self._active &= airborne
        self._contact.copy_(contact)

    def pop(self) -> dict[str, float]:
        """Emit valid statistics only; absent swings or ray coverage are not reported as zero."""
        body = self._body.cpu().tolist()
        feet = self._feet.cpu().clone()
        airborne_time = self._airborne_time.cpu().clone()
        self._body.zero_()
        self._feet.zero_()
        self._airborne_time.zero_()
        result = {}
        for index, name in enumerate(("base_vz_mean", "base_vz_rms", "roll_rms", "pitch_rms")):
            if body[1][index] > 0:
                value = body[0][index] / body[1][index]
                if index > 0:
                    value = math.sqrt(value)
                result[f"Gait/{name}"] = math.degrees(value) if index > 1 else value

        names = (
            "swing_height_mean",
            "swing_peak_height_mean",
            "swing_time_mean",
            "stance_ratio",
            "stance_foot_speed_mean",
        )
        groups = [(None, feet.sum(dim=2))]
        if self._debug:
            groups.extend((name, feet[:, :, index]) for index, name in enumerate(self._foot_names))
        for foot_name, totals in groups:
            prefix = "Gait" if foot_name is None else f"Gait/feet/{foot_name}"
            for index, name in enumerate(names):
                if totals[1, index] > 0:
                    value = (totals[0, index] / totals[1, index]).item()
                    result[f"{prefix}/{name}"] = value * 100 if name == "stance_ratio" else value
        if self._debug:
            result["Gait/debug/completed_swings"] = feet[1, 2].sum().item()
            result["Gait/debug/valid_swing_peaks"] = feet[1, 1].sum().item()
            if airborne_time.sum() > 0:
                result["Gait/debug/swing_height_coverage"] = (feet[1, 0].sum() / airborne_time.sum()).item()
        return result
