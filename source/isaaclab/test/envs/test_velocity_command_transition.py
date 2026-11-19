# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace

import pytest
import torch

from isaaclab.envs.mdp import NormalVelocityCommandCfg, UniformVelocityCommandCfg
from isaaclab.utils import instantiate

pytestmark = pytest.mark.unit


def _environment(num_envs):
    """명령 갱신과 추적 지표에 필요한 CPU 로봇 상태만 제공함."""
    gravity = torch.tensor([[0.0, 0.0, -1.0]]).repeat(num_envs, 1)
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_lin_vel_b=SimpleNamespace(torch=torch.zeros(num_envs, 3)),
            root_ang_vel_b=SimpleNamespace(torch=torch.zeros(num_envs, 3)),
            projected_gravity_b=SimpleNamespace(torch=gravity),
            heading_w=SimpleNamespace(torch=torch.zeros(num_envs)),
        )
    )
    return SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        step_dt=0.02,
        dt=0.02,
        scene={"robot": robot},
        extras={},
        reset_terminated=torch.zeros(num_envs, dtype=torch.bool),
        sim=SimpleNamespace(vis_marker_registry=SimpleNamespace(clear_debug_vis_callback=lambda owner: None)),
    )


def _configuration(**overrides):
    """축마다 다른 양방향 끝값을 사용해 단순 부호 반전과 끝값 선택을 구분함."""
    ranges = overrides.pop(
        "ranges",
        UniformVelocityCommandCfg.Ranges(lin_vel_x=(-1.0, 2.0), lin_vel_y=(-0.5, 1.0), ang_vel_z=(-1.2, 1.8)),
    )
    return UniformVelocityCommandCfg(
        asset_name="robot", resampling_time_range=(10.0, 10.0), debug_vis=False, ranges=ranges, **overrides
    )


def test_transition_population_is_balanced_seeded_and_persistent():
    """전환 환경의 고정 인원과 여섯 역할, 모드 배타성 및 초기화 후 역할 유지를 검증함."""
    cfg = _configuration(rel_transition_envs=0.5, rel_standing_envs=0.2, rel_rotating_envs=0.2)
    torch.manual_seed(731)
    term = instantiate(cfg, _environment(77))
    term.reset()
    transition = term.is_transition_env.clone()
    roles = term.transition_type.clone()
    first_command = term.command.clone()
    first_time = term.time_left.clone()

    assert transition.sum().item() == 38
    assert torch.equal(transition, roles >= 0)
    assert torch.all(roles[~transition] == -1)
    counts = torch.bincount(roles[transition], minlength=6)
    assert counts.min().item() > 0
    assert (counts.max() - counts.min()).item() <= 1
    assert not torch.any(term.is_standing_env & term.is_rotating_env)
    assert not torch.any(transition & (term.is_standing_env | term.is_rotating_env))

    torch.manual_seed(731)
    reproduced = instantiate(
        _configuration(rel_transition_envs=0.5, rel_standing_envs=0.2, rel_rotating_envs=0.2), _environment(77)
    )
    reproduced.reset()
    assert torch.equal(reproduced.transition_type, roles)
    torch.testing.assert_close(reproduced.command, first_command, rtol=0.0, atol=0.0)
    torch.testing.assert_close(reproduced.time_left, first_time, rtol=0.0, atol=0.0)

    term.reset()
    assert torch.equal(term.is_transition_env, transition)
    assert torch.equal(term.transition_type, roles)


def test_transition_endpoints_cycles_timing_and_heading_independence():
    """실제 시간 만료를 통해 끝값과 양쪽 대각선 반전을 검증하며 heading 덮어쓰기를 방지함."""
    cfg = _configuration(
        rel_transition_envs=0.6,
        rel_standing_envs=0.1,
        rel_rotating_envs=0.1,
        heading_command=True,
        heading_control_stiffness=0.5,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 2.0), lin_vel_y=(-0.5, 1.0), ang_vel_z=(-1.2, 1.8), heading=(0.0, 0.0)
        ),
    )
    env = _environment(60)
    env.scene["robot"].data.heading_w.torch.fill_(0.5)
    torch.manual_seed(19)
    term = instantiate(cfg, env)
    term.reset()
    selected = term.is_transition_env
    assert torch.all((term.time_left[selected] >= 2.0) & (term.time_left[selected] <= 2.5))
    torch.testing.assert_close(term.time_left[~selected], torch.full_like(term.time_left[~selected], 10.0))
    history = [term.command.clone()]
    for step in range(4):
        term.compute(dt=2.5)
        history.append(term.command.clone())
        assert torch.all((term.time_left[selected] >= 2.0) & (term.time_left[selected] <= 2.5))
        if step == 0:
            torch.testing.assert_close(term.time_left[~selected], torch.full_like(term.time_left[~selected], 7.5))
    history = torch.stack(history)

    # 명령의 허용 집합은 역할의 공개 계약이며 현재 구현의 위상 계산을 복제하지 않음.
    allowed = {
        0: [[-1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        1: [[0.0, -0.5, 0.0], [0.0, 1.0, 0.0]],
        2: [[-1.0, -0.5, 0.0], [2.0, 1.0, 0.0]],
        3: [[-1.0, 1.0, 0.0], [2.0, -0.5, 0.0]],
        4: [[-1.0, 0.0, -1.2], [-1.0, 0.0, 1.8], [2.0, 0.0, -1.2], [2.0, 0.0, 1.8]],
        5: [[0.0, -0.5, -1.2], [0.0, -0.5, 1.8], [0.0, 1.0, -1.2], [0.0, 1.0, 1.8]],
    }
    for role in range(6):
        values = history[:, term.transition_type == role]
        assert values.shape[1] > 0
        candidates = torch.tensor(allowed[role])
        matches = torch.isclose(values.unsqueeze(-2), candidates).all(dim=-1)
        assert torch.all(matches.any(dim=-1))
        torch.testing.assert_close(values[4], values[0], rtol=0.0, atol=0.0)
        assert torch.all((values[1] != values[0]).any(dim=-1))
        torch.testing.assert_close(values[2], values[0], rtol=0.0, atol=0.0)
        active_axes = {0: [0], 1: [1], 2: [0, 1], 3: [0, 1], 4: [0, 2], 5: [1, 2]}[role]
        endpoint_sums = torch.tensor([1.0, 0.5, 0.6])[active_axes]
        torch.testing.assert_close(
            values[0][:, active_axes] + values[1][:, active_axes], endpoint_sums.expand(values.shape[1], -1)
        )


def test_partial_reset_preserves_other_commands_roles_timers_and_tracking_metrics():
    """부분 초기화가 대상 외 환경의 명령, 역할, 남은 시간과 추적 오차 누적을 보존함을 검증함."""
    env = _environment(24)
    torch.manual_seed(53)
    term = instantiate(_configuration(rel_transition_envs=0.5), env)
    term.reset()
    initial_command = term.command.clone()
    initial_roles = term.transition_type.clone()
    term.compute(dt=2.5)
    selected = torch.nonzero(term.is_transition_env).flatten()[:3]
    selected = torch.cat((selected, torch.nonzero(~term.is_transition_env).flatten()[:1]))
    untouched = torch.ones(24, dtype=torch.bool)
    untouched[selected] = False
    before_command = term.command.clone()
    before_time = term.time_left.clone()
    before_counter = term.command_counter.clone()

    term.reset(selected)
    assert torch.equal(term.transition_type, initial_roles)
    torch.testing.assert_close(term.command[untouched], before_command[untouched], rtol=0.0, atol=0.0)
    torch.testing.assert_close(term.time_left[untouched], before_time[untouched], rtol=0.0, atol=0.0)
    assert torch.equal(term.command_counter[untouched], before_counter[untouched])
    transition_ids = selected[term.is_transition_env[selected]]
    torch.testing.assert_close(term.command[transition_ids], initial_command[transition_ids], rtol=0.0, atol=0.0)

    # 초기화하지 않은 환경은 최초 명령에 대해 측정한 한 번의 추적 오차를 보고해야 함.
    extras = term.reset(torch.nonzero(untouched).flatten())
    torch.testing.assert_close(extras["error_vel_xy"], initial_command[untouched, :2].norm(dim=-1).mean())
    torch.testing.assert_close(extras["error_vel_yaw"], initial_command[untouched, 2].abs().mean())


def test_disabled_transition_options_preserve_seeded_sampling_and_rng():
    """전환을 끄면 전환 전용 시간 설정이 기존 명령과 난수 스트림에 영향을 주지 않음을 검증함."""
    ranges = UniformVelocityCommandCfg.Ranges(lin_vel_x=(1.0, 2.0), lin_vel_y=(0.1, 0.3), ang_vel_z=(0.2, 0.4))

    def sample(cfg):
        torch.manual_seed(97)
        term = instantiate(cfg, _environment(20))
        term.reset()
        history = [term.command.clone()]
        for dt in (0.02, 10.0):
            term.compute(dt)
            history.append(term.command.clone())
        return torch.stack(history), term.time_left.clone(), torch.rand(8)

    baseline = sample(_configuration(ranges=ranges, rel_standing_envs=0.2, rel_rotating_envs=0.2))
    disabled = sample(
        _configuration(
            ranges=ranges,
            rel_standing_envs=0.2,
            rel_rotating_envs=0.2,
            rel_transition_envs=0.0,
            transition_resampling_time_range=(0.5, 0.75),
        )
    )
    for expected, actual in zip(baseline, disabled, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_invalid_transition_configuration_and_normal_generator_are_rejected():
    """전환 비율, 확률 합계, 시간과 양방향 범위를 검증하고 정규분포 생성기 사용을 거부함."""
    bad_configs = [
        _configuration(rel_transition_envs=-0.1),
        _configuration(rel_transition_envs=1.1),
        _configuration(rel_transition_envs=float("nan")),
        _configuration(rel_transition_envs=0.4, rel_standing_envs=0.4, rel_rotating_envs=0.4),
        _configuration(rel_transition_envs=0.2, transition_resampling_time_range=(0.0, 2.0)),
        _configuration(rel_transition_envs=0.2, transition_resampling_time_range=(2.5, 2.0)),
        _configuration(rel_transition_envs=0.2, transition_resampling_time_range=(2.0, float("inf"))),
    ]
    for axis, bounds in (("lin_vel_x", (0.0, 2.0)), ("lin_vel_y", (-0.5, 0.0)), ("ang_vel_z", (-float("inf"), 1.8))):
        ranges = _configuration().ranges.replace(**{axis: bounds})
        bad_configs.append(_configuration(rel_transition_envs=0.2, ranges=ranges))
    for cfg in bad_configs:
        with pytest.raises(ValueError):
            instantiate(cfg, _environment(12))

    with pytest.raises(ValueError, match="rounded transition quota"):
        instantiate(
            _configuration(rel_transition_envs=0.5, rel_standing_envs=0.25, rel_rotating_envs=0.25), _environment(3)
        )

    normal_cfg = NormalVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        debug_vis=False,
        rel_transition_envs=0.2,
        ranges=NormalVelocityCommandCfg.Ranges(
            mean_vel=(0.0, 0.0, 0.0), std_vel=(0.5, 0.5, 0.5), zero_prob=(0.0, 0.0, 0.0)
        ),
    )
    with pytest.raises(ValueError, match="transition|전환"):
        instantiate(normal_cfg, _environment(12))
