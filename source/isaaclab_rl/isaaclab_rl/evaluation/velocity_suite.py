# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Serializable, ordered evaluation cases and episode-weighted summaries."""

import hashlib
import json
import math


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def build_cases(catalog, protocol="both", traversal_speed=0.5):
    """Tracking covers every task; traversal tests both stair types in three directions."""
    if protocol not in ("tracking", "traversal", "both"):
        raise ValueError("protocol must be tracking, traversal, or both.")
    cases = []
    if protocol in ("tracking", "both"):
        cases += [
            {"task_id": i, "protocol": "tracking", "heading": 0.0, "phase": i % 2, "yaw_variant": 1 if i % 2 else -1}
            for i in range(catalog.num_tasks)
        ]
    if protocol in ("traversal", "both"):
        limits = catalog.signature["command_ranges"][0]
        if not (0 < traversal_speed <= limits[1] and traversal_speed >= limits[0]):
            raise ValueError("Traversal speed must be positive and inside the configured forward velocity range.")
        for type_id, name in enumerate(catalog.terrain_names):
            if name not in ("pyramid_stairs", "pyramid_stairs_inv"):
                continue
            mask = (catalog.terrain_type_ids == type_id) & (catalog.modes == 2)
            for level in sorted(set(catalog.terrain_levels[mask].tolist())):
                task = int(((catalog.terrain_levels == level) & mask).nonzero()[0])
                for heading in (0.0, math.pi / 4, math.pi / 2):
                    cases.append(
                        {
                            "task_id": task,
                            "protocol": "traversal",
                            "heading": heading,
                            "phase": 0,
                            "yaw_variant": 1,
                            "command": [traversal_speed, 0.0, 0.0],
                        }
                    )
    if not cases:
        raise ValueError("No evaluation cases match this terrain catalog and protocol.")
    output = []
    for index, case in enumerate(cases):
        task = case["task_id"]
        group = catalog.column_groups[int(catalog.terrain_type_ids[task])]
        traversal = case["protocol"] == "traversal"
        column = group[(int(catalog.terrain_levels[task]) if traversal else task) % len(group)]
        fractions = [0.5] * 3 if traversal else [0.25 if (task + axis) % 2 == 0 else 0.75 for axis in range(3)]
        output.append(dict(case, case_id=index, column=column, fractions=fractions))
    return output


def summarize(rows):
    """Keep censored metrics out of averages and report their denominators explicitly."""

    def group(records):
        result = {
            "episodes": len(records),
            "failure_rate": sum(r["failed"] for r in records) / len(records),
            "boundary_rate": sum(r["boundary"] for r in records) / len(records),
            "mean_steps": sum(r["steps"] for r in records) / len(records),
        }
        traversal = [r for r in records if r["protocol"] == "traversal"]
        if traversal:
            result["traversal_success_rate"] = sum(r["success"] for r in traversal) / len(traversal)
        for metric in ("epte_sp", "normalized_epte_sp", "zero_command_drift"):
            for axis, name in enumerate(("vx", "vy", "yaw")):
                values = [r[metric][axis] for r in records if r[metric][axis] is not None]
                result[f"{metric}/{name}"] = sum(values) / len(values) if values else None
                result[f"{metric}_count/{name}"] = len(values)
        return result

    if not rows:
        raise ValueError("Cannot summarize an empty evaluation.")
    groups = {}
    for row in rows:
        key = f"{row['protocol']}/{row['terrain']}/level_{row['level']}/{row['mode']}"
        groups.setdefault(key, []).append(row)
    return {
        "overall": group(rows),
        "protocols": {
            protocol: group([row for row in rows if row["protocol"] == protocol])
            for protocol in sorted({row["protocol"] for row in rows})
        },
        "groups": {key: group(records) for key, records in sorted(groups.items())},
    }
