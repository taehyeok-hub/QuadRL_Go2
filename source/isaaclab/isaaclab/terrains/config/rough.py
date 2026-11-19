# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rough terrains configuration."""

from ... import terrains as terrain_gen
from ..terrain_generator_cfg import TerrainGeneratorCfg

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),    
    border_width=20.0,      # 전체 지형 배열 바깥을 둘러싼 {border_width} m 폭의 평지
    num_rows=10,
    num_cols=20,            # [rows, cols] = [terrain level, terrain type]  : type은 랜덤이 아니라 
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,

    # proportion 으로 지형 배치 비율 설정 -> 정규화를 진행하지만 웬만해선 합 1로.
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width_range=(0.30, 0.25),
            platform_width=3.0,     # 중앙의 평평한 정사각형 플랫폼
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width_range=(0.30, 0.25),
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2,
            grid_width=0.45,
            grid_height_range=(0.05, 0.2),  # 각 블록 윗면 높이는 해당 크기 안에서 양수·음수로 무작위 배정
            platform_width=2.0,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
            # noise_range  : 생성 기준 높이 샘플의 범위  
            # noise_step   : 초기 높이 샘플을 {noise_step} cm 간격으로 선택
            # border_width : 지형 조각 가장자리의 평지 폭
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
    },
)
"""Rough terrains configuration."""
