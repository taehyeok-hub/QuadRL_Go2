# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
import math
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import torch

from ..managers import CommandManager, CurriculumManager, RewardManager, TerminationManager
from ..utils import index_fill_
from .common import VecEnvStepReturn
from .manager_based_env import ManagerBasedEnv
from .manager_based_rl_env_cfg import ManagerBasedRLEnvCfg

logger = logging.getLogger(__name__)


# 강화학습 환경의 기본 설정
class ManagerBasedRLEnv(ManagerBasedEnv, gym.Env): 
    """ Manager-based workflow를 사용하는 강화학습 환경의 상위 클래스.
    
    이 클래스는 ManagerBasedEnv를 상속하며, 강화학습 환경의 핵심 기능을 구현함.
    해당 알고리즘은 특정 강화학습 라이브러리에 종속되지 않도록 설계되어 있음.

    여러 하위 환경을 병렬로 실행하는 벡터화 환경을 전제로 하며, 이 하위 환경의 개수는 ''num_envs''로 지정함.
    environment에서 반환하는 각 observation들은 모든 sub-env 의 관측을 모은 batch임.
    그런 의미에서 `step`이라 함은 각 하위 환경에 적응할 행동을 batch 형태로 받게 됨.

    이 환경들 자체가 벡터화 되어있다고 했지만, `gym.vector.VectorEnv`는 상속하지 않음.
    해당 클래스에는 대기 및 비동기 업데이트처럼 필요하지 않은 여러 메서드가 포함되어있기 때문임.
    강화학습 라이브러리마다 환경의 정의하는 방식이 다르지만, 여기서는 복잡성을 줄이기 위해 `gym.Env`를 직접 사용함.

    Note:
        벡터화 환경에서는 환경 생성 후 첫 `step` 호출 전에 `reset` 메서드를 한 번만 호출하는 것을 권장함.
        그 이후에는 step함수가 종료된 하위 환경의 초기화를 처리함.
        이는 시뮬레이터가 벡터화 환경 내의 개별 하위 환경의 reset 호출을 지원하기 않기 때문임.  
    """

    is_vector_env: ClassVar[bool] = True
    # 환경이 벡터화된 환경인지 아닌지 확인용.
    metadata: ClassVar[dict[str, Any]] = {              # 환경에 대한 메타데이터
        "render_modes": [None, "human", "rgb_array"],
        "autoreset_mode": gym.vector.AutoresetMode.SAME_STEP,
    }   

    cfg: ManagerBasedRLEnvCfg
    """ 환경에 대한 구성 """

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        """ 환경을 초기화하는 함수.

        Args:
            cfg: 환경 설정 객체.
            render_mode: 환경의 렌더링 모드.
            기본값은 None이며, "human" 모드와 유사하게 동작함.
        """
        # -- 커리큘럼 등에서 사용하는 전체 환경 공통 스텝 카운터
        self.common_step_counter = 0

        # 매니저를 로드하기 전에 MDP 함수에서 episode length buffer를 사용할 수 있도록 먼저 버퍼를 생성.
        # --> 각 sub_env의 현재 에피소드 진행 스텝 수
        self.episode_length_buf = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device, dtype=torch.long)
        # --> 각 sub_env의 초기화 필요 여부
        self.reset_buf = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device, dtype=torch.bool)

        # base class의 생성자를 호출해서 scene(시뮬레이션 장면, 구성요소)을 setup함.
        super().__init__(cfg=cfg)
        
        # 렌더링 모드 저장
        self.render_mode = render_mode

        # 데이터 및 상수 초기화
        # --> Gym 영상 녹화 래퍼의 프레임 속도(FPS)를 환경 스텝 주기의 역수로 설정 (녹화 영상의 재생 속도 == 시뮬레이션 속도)
        self.metadata["render_fps"] = 1 / self.step_dt
        # --> 시뮬레이션 설정에서 RTX 센서 사용 여부를 읽어 저장
        self.has_rtx_sensors = self.sim.get_setting("/isaaclab/render/rtx_sensors")
        
        logger.info("Completed setting up the environment...")

    """
    Properties.
    """

    @property
    def max_episode_length_s(self) -> float:
        """Maximum episode length in seconds."""
        return self.cfg.episode_length_s

    @property
    def max_episode_length(self) -> int:
        """Maximum episode length in environment steps."""
        return math.ceil(self.max_episode_length_s / self.step_dt)

    """
    Operations - Setup.
    """

    def load_managers(self):
        """ 매니저 생성 순서가 중요함:

        관측 매니저는 명령 매니저와 행동 매니저에 접근할 수 있어야 하고,
        상 매니저는 종료 매니저에 접근할 수 있어야 함.

        command -> observation, action -> termination -> reward -> curriculum
        """

        # -- command manager : 로봇 명령 관리
        self.command_manager: CommandManager = CommandManager(self.cfg.commands, self)
        logger.info(f"Command Manager: {self.command_manager}")

        # 부모 클래스에서 관측 및 행동 매니저 등을 생성
        super().load_managers()

        # -- termination manager : 각 하위 환경의 에피소드 종료 조건을 관리
        self.termination_manager = TerminationManager(self.cfg.terminations, self)
        logger.info(f"Termination Manager: {self.termination_manager}")

        # -- reward manager : 설정된 보상 항목을 계산하고 합산
        self.reward_manager = RewardManager(self.cfg.rewards, self)
        logger.info(f"Reward Manager: {self.reward_manager}")

        # -- curriculum manager : 학습 진행에 따른 난이도 및 설정 변경을 관리
        self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
        logger.info(f"Curriculum Manager: {self.curriculum_manager}")

        # Gym 인터페이스에 맞게 action_space와 obs_space를 설정
        self._configure_gym_env_spaces()

        # 시뮬레이션 시작 시 실행하는 "startup" 모드의 이벤트가 있으면 적용
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")



    def setup_manager_visualizers(self):
        """ 매니저의 항목과 보상·에피소드 길이를 시뮬레이션 화면에서 실시간 그래프로 볼 수 있도록 연결하는 함수.

        시뮬레이션 컨텍스트에 등록된 모든 시각화 도구의 add_live_plots를 호출한다.
        매니저 항목과 함께 episode/mean_reward와 episode/episode_length를
        스칼라 지표로 등록하여 개별 항목 옆에서 확인할 수 있도록 한다.

        Kit 백엔드에서는 BaseEnvWindow가 UI 패널을 구성할 수 있도록
        ManagerLiveVisualizer 객체들을 manager_visualizers에 저장한다.
        """

        # 시각화할 매니저들 모음.
        managers = {
            "action_manager": self.action_manager,
            "observation_manager": self.observation_manager,
            "command_manager": self.command_manager,
            "termination_manager": self.termination_manager,
            "reward_manager": self.reward_manager,
            "curriculum_manager": self.curriculum_manager,
        }
        # 보상과 에피소드 길이를 별도의 그래프 항목으로 등록
        scalars = {
            "episode": {
                "mean_reward": lambda: float(getattr(self, "reward_buf", None).mean())
                if getattr(self, "reward_buf", None) is not None
                else 0.0,
                "episode_length": lambda: float(self.episode_length_buf.float().mean()),
            }
        }
        # 활성화된 모든 시각화 도구에 그래프 연결
        for viz in self.sim.visualizers:
            viz.add_live_plots(managers, scalars=scalars)

        # BaseEnvWindow가 Kit UI 패널을 구성할 때 사용할 매니저 시각화 객체들을 모음
        # kit_manager_visualizers 속성이 없는 시각화 도구는 빈 딕셔너리로 처리
        self.manager_visualizers = {
            name: mlv for v in self.sim.visualizers for name, mlv in getattr(v, "kit_manager_visualizers", {}).items()
        }

    """
    Operations - MDP
    """

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """ 환경의 dynamics를 한 스텝 진행하고, 종료된 하위 환경을 초기화하는 함수.
        
        '여기에서 말하는 환경 스텝은 동일한 정책에서 액션이 변경되는 한 단위'라고 생각하면 될 것이다. 
        `ManagerBasedEnv`클래스의 step 함수와 달리, 이 함수는 다음 작업을 순서대로 수행한다.

        1. 입력 액션을 처리함.
        2. decimation 설정에 따라 물리 시뮬레이션을 진행함.
        3. 렌더링 조건에 따라 화면과 센서를 갱신함.
        4. 환경 스텝 카운터를 갱신하고 종료 조건과 보상을 계산함.
        5. 종료되거나 시간 제한에 도달한 하위 환경을 초기화함.
        6. 명령과 주기적 이벤트를 갱신하고 관측값을 계산함.
        7. 관측값, 보상, 종료 여부, 시간 초과 여부, 추가 정보를 반환함.

        환경의 제어 스텝 시간은 physics_dt * decimation이다.
        예를 들어 (물리 시뮬레이터 주기) physics_dt=0.005, decimation=4이면
        step 호출 한 번에 시뮬레이션 시간 0.02초가 진행된다(제어 주파수 50 Hz).

        render_enabled=False이면 Kit 앱 루프를 실행하지 않으므로
        카메라·RTX 센서 렌더링과 GUI 뷰포트 갱신을 생략한다.
        초기화 직후 RTX 센서를 위한 추가 렌더링도 생략한다.
        독립 시각화 도구(Newton, Rerun, Viser)는 계속 갱신된다.

        Args:
            action: 각 하위 환경에 적용할 액션. 형태는 (num_envs, action_dim).
        
        Returns:
            observation(다음 관측), reward, termination, 시간 초과 여부 및 추가 정보를 담은 tuple.
        """

        # 액션을 환경의 연산 장치로 옮기고 액션 매니저에서 전처리
        self.extras["log"] = {}
        self.action_manager.process_action(action.to(self.device))
        # 환경 스텝 시작 전 데이터를 기록
        self.recorder_manager.record_pre_step()
        # 렌더링 필요 여부를 환경 스텝마다 한 번 확인
        is_rendering = self.sim.is_rendering


        # 물리 step 실행
        # --> 물리 엔진이 decimation을 처리하므로, 한 번의 호출로 모든 물리 서브스텝을 진행함.
        steps_per_call = self.cfg.decimation if self._physics_handles_decimation else 1
        for _ in range(self.cfg.decimation // steps_per_call):
            # --> 실행할 물리 스텝 수만큼 시뮬레이션 스텝 카운터 증가  
            self._sim_step_counter += steps_per_call
            # --> 처리된 액션을 로봇의 제어 입력에 적용
            self.action_manager.apply_action()
            # --> 장면의 제어 입력 및 상태 변경 사항을 시뮬레이터에 전달
            self.scene.write_data_to_sim()
            # --> 자동 렌더링 없이 물리 시뮬레이션 진행 
            self.sim.step(render=False)
            # --> 물리 스텝 실행 후 데이터를 기록
            self.recorder_manager.record_post_physics_decimation_step()

            # 지정된 물리 스텝 간격마다 렌더링 수행
            # --> render_enabled=False이면 Kit 앱 루프를 생략하고 독립 시각화 도구는 갱신
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render(skip_app_pumping=not self.render_enabled)
            self.scene.update(dt=self.physics_dt * steps_per_call)


        # post-step:
        # --> env counter 업데이트
        self.episode_length_buf += 1    # 현재 에피소드 진행 스텝 수 증가 (per env)
        self.common_step_counter += 1   # 전체 환경 스텝 카운터 증가 (common for all envs)
        
        # --> 종료 조건 계산 및 일반 종료와 시간 초과 여부를 각각 저장
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs

        # --> 이번 환경 스텝의 보상 계산
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        # Collect curriculum statistics from the terminal state before resetting the scene.
        self.curriculum_manager.update(dt=self.step_dt)

        # --> 활성화된 항목에 대해 기록용 관측값을 계산하고 스텝 이후 데이터를 기록
        if len(self.recorder_manager.active_terms) > 0:
            # update observations for recording if needed
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # --> reset 환경 처리 : 종료되거나 시간 제한에 도달하여 초기화가 필요한 하위 환경의 인덱스 추출
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1).int()
        if len(reset_env_ids) > 0:
            # 설정이 활성화되어 있으면 초기화 직전의 관측값을 추가 정보에 저장
            if self.cfg.compute_final_obs:
                self.extras["final_obs"] = self.observation_manager.compute()
            self.recorder_manager.record_pre_reset(reset_env_ids)           # 초기화 직전 데이터를 기록
            
            # 해당 하위 환경을 초기화하고 에피소드 정보를 정리
            self._reset_idx(reset_env_ids)
            
            # 센서가 scene에 add될 시, 무조건 reset에서 해당 변화를 반영하도록 렌더링 해야함.
            if self.render_enabled and is_rendering and self.has_rtx_sensors and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()
            self.recorder_manager.record_post_reset(reset_env_ids)


        # --> 시각화 UI에서 요청한 수동 초기화 처리
        if self.sim.consume_reset_request():
            # Only reset envs not already reset this step to avoid redundant resets.
            not_yet_reset = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            index_fill_(not_yet_reset, reset_env_ids, False)
            manual_reset_ids = not_yet_reset.nonzero(as_tuple=False).squeeze(-1).int()
            if len(manual_reset_ids) > 0:
                # mark as terminated so RL wrappers observe the episode boundary
                index_fill_(self.reset_terminated, manual_reset_ids, True)
                # mirror the recorder lifecycle used for normal resets
                self.recorder_manager.record_pre_reset(manual_reset_ids)
                self._reset_idx(manual_reset_ids)
                self.recorder_manager.record_post_reset(manual_reset_ids)


        # 환경 스텝 시간에 따라 목표 명령을 갱신
        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.active_terms:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # --> 렌더링과 초기화가 끝난 뒤 영상 녹화기를 한 스텝 진행
        for recorder in self.video_recorders:
            recorder.step()
        # --> 초기화된 환경의 상태와 갱신된 명령이 반영된 관측값을 계산
        self.obs_buf = self.observation_manager.compute(update_history=True)

        # 관측값, 보상, 일반 종료 여부, 시간 초과 여부, 추가 정보 반환
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def render(self, recompute: bool = False) -> np.ndarray | None:
        """Run rendering without stepping through the physics.

        By convention, if mode is:

        - **human**: Render to the current display and return nothing. Usually for human consumption.

        .. note::
            ``render_mode="rgb_array"`` is no longer supported.  Use
            :class:`~isaaclab.envs.utils.video_recorder_cfg.VideoRecorderCfg` on
            ``env_cfg.video_recorders`` instead.

        Args:
            recompute: Whether to force a render even if the simulator has already rendered the scene.
                Defaults to False.

        Returns:
            None.

        Raises:
            RuntimeError: If mode is set to "rgb_data" and simulation render mode does not support it.
                In this case, the simulation render mode must be set to ``RenderMode.PARTIAL_RENDERING``
                or ``RenderMode.FULL_RENDERING``.
            NotImplementedError: If an unsupported rendering mode is specified.
        """
        # run a rendering step of the simulator
        # if we have rtx sensors, we do not need to render again since step already rendered
        if not self.has_rtx_sensors and not recompute:
            self.sim.render()
        # decide the rendering mode
        if self.render_mode == "rgb_array":
            import warnings

            warnings.warn(
                "render_mode='rgb_array' is deprecated and will be removed in a future release. "
                "Use VideoRecorderCfg on env_cfg.video_recorders to capture frames instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return None
        if self.render_mode == "human" or self.render_mode is None:
            return None
        else:
            raise NotImplementedError(
                f"Render mode '{self.render_mode}' is not supported. Please use: {self.metadata['render_modes']}."
            )

    def close(self):
        if not self._is_closed:
            # destructor is order-sensitive
            del self.command_manager
            del self.reward_manager
            del self.termination_manager
            del self.curriculum_manager
            # Drop the observation/action space objects. gymnasium's wrapper chain keeps
            # the env referenced past close, so without this they leak — and for image
            # observations each space holds a large gym.spaces.Box bounds array.
            self.single_observation_space = None
            self.single_action_space = None
            self.observation_space = None
            self.action_space = None
            # call the parent class to close the environment
            super().close()

    """
    Helper functions.
    """

    def _configure_gym_env_spaces(self):
        """Configure the action and observation spaces for the Gym environment."""
        # observation space (unbounded since we don't impose any limits)
        self.single_observation_space = gym.spaces.Dict()
        for group_name, group_term_names in self.observation_manager.active_terms.items():
            # extract quantities about the group
            has_concatenated_obs = self.observation_manager.group_obs_concatenate[group_name]
            group_dim = self.observation_manager.group_obs_dim[group_name]
            # check if group is concatenated or not
            # if not concatenated, then we need to add each term separately as a dictionary
            if has_concatenated_obs:
                self.single_observation_space[group_name] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=group_dim)
            else:
                group_term_cfgs = self.observation_manager._group_obs_term_cfgs[group_name]
                term_dict = {}
                for term_name, term_dim, term_cfg in zip(group_term_names, group_dim, group_term_cfgs):
                    low = -np.inf if term_cfg.clip is None else term_cfg.clip[0]
                    high = np.inf if term_cfg.clip is None else term_cfg.clip[1]
                    term_dict[term_name] = gym.spaces.Box(low=low, high=high, shape=term_dim)
                self.single_observation_space[group_name] = gym.spaces.Dict(term_dict)
        # action space (unbounded since we don't impose any limits)
        action_dim = sum(self.action_manager.action_term_dim)
        self.single_action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(action_dim,))

        # batch the spaces for vectorized environments
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

    def _reset_idx(self, env_ids: torch.Tensor | slice):
        """Reset environments based on specified indices.

        Args:
            env_ids: A slice or environment indices on the environment device.
        """
        # update the curriculum for environments that need a reset
        self.curriculum_manager.compute(env_ids=env_ids)
        # reset the internal buffers of the scene elements
        self.scene.reset(env_ids)
        # apply events such as randomizations for environments that need a reset
        if "reset" in self.event_manager.active_terms:
            env_step_count = self._sim_step_counter // self.cfg.decimation
            self.event_manager.apply(mode="reset", env_ids=env_ids, global_env_step_count=env_step_count)

        # iterate over all managers and reset them
        # this returns a dictionary of information which is stored in the extras
        # note: This is order-sensitive! Certain things need be reset before others.
        self.extras["log"] = dict()
        # -- observation manager
        info = self.observation_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- action manager
        info = self.action_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- rewards manager
        info = self.reward_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- curriculum manager
        info = self.curriculum_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- command manager
        info = self.command_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- event manager
        info = self.event_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- termination manager
        info = self.termination_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- recorder manager
        info = self.recorder_manager.reset(env_ids)
        self.extras["log"].update(info)

        index_fill_(self.episode_length_buf, env_ids, 0)
