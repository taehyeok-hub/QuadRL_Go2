:orphan:

Success-History Terrain Curriculum
=================================

Go2 rough uses a per-environment, two-episode success history to update terrain
levels. The LP sampler is disabled. The original two-trial implementation was adapted from the public
`Curriculum-Based Reinforcement Learning for Quadrupedal Jumping implementation
<https://github.com/vassil-atn/Curriculum-Quadruped-Jumping-DRL/blob/master/legged_gym/legged_gym/envs/base/legged_robot.py#L1526>`_.
That work was published in IEEE Robotics & Automation Magazine in 2025,
DOI ``10.1109/MRA.2024.3487325``. Its jump-success label is replaced here by the
explicit locomotion goal below; this is an adaptation, not an exact reproduction
of the paper's jumping task.

Decision rules
--------------

Defaults match the archived ``2026-11-18_12-36-40_success_history_slides_X``
run: each environment collects two eligible completed training episodes. At least
one success promotes one level; two failures demote one level. Each completed
window is consumed. PPO continues updating normally. These are training episode
histories, not separate frozen-policy evaluations. ``promotion_success_rate=None``
selects the original any-success rule, and ``demotion_success_rate=0.0`` requires
all failures before demotion.

The optional rate-based experiment uses ``history_length=10`` (or 20),
``promotion_success_rate=0.8`` and ``demotion_success_rate=0.4``. Intermediate
rates retain the level and consume the window. Those are local experimental
settings, not defaults or paper settings.

Goal and episode ownership
--------------------------

Success is ``goal_visited AND NOT failure_at_episode_end``. A goal visit requires
body position and all four foot XY positions beyond the **same** inner rectangle
face, with 5 cm clearance, while remaining inside the assigned tile. The body
must be within 45 degrees of upright. Any of the four exit faces qualifies, so
ordinary omnidirectional commands can retain their existing yaw and resampling
settings. A visit is latched until episode end; a later fall overrides it.
This is a position/posture predicate, not a contact or landing detector.

Pyramid stairs use their actual flat outer border as the goal strip. With the
current 8 by 8 m tiles and 1 m stair border, the inner face is 3 m from the tile
center; the body and every foot must enter the same border before the tile exit.
For other terrains, ``goal_region_width`` defines an outer strip geometrically;
that strip is **not assumed flat**, and reaching it does not certify that every
obstacle in the tile was traversed.

The robot still starts around the terrain center. Therefore center-to-border on
``pyramid_stairs`` is **descent**, and on ``pyramid_stairs_inv`` it is **ascent**.
The implementation logs their success rates separately. No spawn location,
command range, reward, observation, or action configuration is changed by this
curriculum. Curving or reversing velocity commands may never reach an exit;
this curriculum measures actual goal traversal, not just command tracking.

Assigned-tile exits are timeouts, preventing further training on a neighboring
tile before reset. Exiting without a valid goal visit is a failure of the goal
objective, even if it is not a physical fall. Full-horizon episodes without a
goal also contribute zero. Initial/manual resets do not vote. Episodes with no
planar commanded speed above ``min_command_speed`` do not vote; standing/rotation
can therefore retain their normal command behavior without being classified as
failed traversals.

Configuration and reuse
-----------------------

The reusable classes live in ``velocity.mdp.curriculum.success`` and
``velocity.mdp.curriculum.success_history``. Goal geometry is separate in
``velocity.mdp.evaluation.geometry``. The Go2 rough configuration selects:

.. code-block:: python

   from isaaclab.managers import CurriculumTermCfg
   from isaaclab_tasks.core.velocity.mdp.curriculum.success import (
       SuccessTerrainCfg, SuccessTerrainCurriculum,
   )

   self.curriculum.learning_progress = None
   self.curriculum.terrain_levels = CurriculumTermCfg(
       func=SuccessTerrainCurriculum,
       params={"settings": SuccessTerrainCfg(
           history_length=2,
           promotion_success_rate=None,
           demotion_success_rate=0.0,
           goal_region_width=1.0,
           min_command_speed=0.2,
       )},
   )

The original two-trial rule follows the public source. Goal width, clearance, upright condition,
and minimum command speed are explicit local design choices, not paper settings.
Other robots need four resolvable foot bodies; ``asset_name``,
``foot_body_pattern``, and ``command_name`` are curriculum-term parameters.
Stair configurations require a positive outer border. Flat tasks and interactive
playback disable this terrain curriculum. The existing distance-based function
``mdp.terrain_levels_vel`` remains available as a separate baseline.

Run and inspect
---------------

Start a fresh comparison run:

.. code-block:: bash

   uv run isaaclab train --rl_library rsl_rl \
       --task Isaac-Velocity-Rough-UnitreeGo2 \
       --num_envs 4096 --max_iterations 10000 --seed 42 \
       --logger wandb --log_project_name go2_lp_compare \
       --run_name success_history \
       agent.experiment_name=go2_success_history

WandB and TensorBoard receive ``Curriculum/terrain_levels/level``,
``eligible_fraction``, ``success_rate``, ``promotion_rate``, ``demotion_rate``,
``ascent_success_rate``, and ``descent_success_rate``. The success rate is computed
over eligible episodes ending in that reset batch; it is not a held-out score.
Promotion/demotion rates also include episodes waiting for their second result.
Detailed curriculum scalars stay out of the training console but remain available
in both dashboard writers. Terrain-level averages describe the assigned training distribution, not policy
quality. Compare saved policies on fixed terrain geometry, commands, and seeds.

Checkpoint state preserves unfinished histories together with terrain levels
and columns. RSL-RL restores the entire manager, then resets physical episodes
onto those restored tiles before collecting training data. Settings, terrain,
termination, observation, command, event, horizon, and environment-count changes
require ``--reset_curriculum``. To fine-tune an older LP/distance checkpoint,
pass ``--checkpoint /path/to/model.pt --reset_curriculum``; otherwise its
curriculum-state layout is incompatible. Fine-tuning is a separate experiment
from training a fresh comparison policy.
