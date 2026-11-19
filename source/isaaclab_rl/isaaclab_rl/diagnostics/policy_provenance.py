# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record starting policy identity for controlled training comparisons."""

import hashlib
import json
from pathlib import Path

import torch


def write_policy_fingerprint(runner, log_dir, resume_path=None):
    """Hash initial/restored model tensors before any learning update."""
    record = {"resume_path": resume_path}
    for name in ("actor", "critic"):
        model = getattr(runner.alg, name, None)
        if model is None:
            continue
        digest = hashlib.sha256()
        for key, value in sorted(model.state_dict().items()):
            tensor = value.detach().cpu().contiguous()
            digest.update(f"{key}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        record[f"{name}_sha256"] = digest.hexdigest()
    path = Path(log_dir) / "diagnostics" / "initial_policy.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2))
