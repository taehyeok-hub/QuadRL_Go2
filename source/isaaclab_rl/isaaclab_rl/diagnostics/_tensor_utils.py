# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""진단에 필요한 중첩 텐서 순회와 독립 복사를 제공함."""

from collections.abc import Mapping

import torch


def _tensors(value, prefix: str = ""):
    """중첩된 관측·모델 상태에서 텐서와 그 이름을 순회함."""
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, Mapping):
        for name, item in value.items():
            yield from _tensors(item, f"{prefix}.{name}" if prefix else str(name))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _tensors(item, f"{prefix}.{index}")
    elif isinstance(value, float):
        yield prefix, torch.tensor(value, dtype=torch.float64)


def _copy_tensors(value, *, cpu: bool = False, env_ids: list[int] | None = None):
    """이후 환경 갱신과 optimizer 변경의 영향을 받지 않는 텐서 복사본을 만듦."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach() if env_ids is None else value.detach()[env_ids]
        return tensor.to(device="cpu", copy=True) if cpu else tensor.clone()
    if isinstance(value, Mapping):
        return {key: _copy_tensors(item, cpu=cpu, env_ids=env_ids) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_copy_tensors(item, cpu=cpu, env_ids=env_ids) for item in value)
    if isinstance(value, list):
        return [_copy_tensors(item, cpu=cpu, env_ids=env_ids) for item in value]
    return value
