:orphan:

Learning-Progress Locomotion Curriculum
======================================

The optional velocity curriculum uses signed learning progress (LP) to sample terrain,
terrain difficulty, and velocity-command tasks. It replaces distance-based terrain
promotion; no per-terrain promotion distance or fixed standing/rotating quota is
required. The implementation follows the sampling principle in
`Scaling Rough Terrain Locomotion with Automatic Curriculum Reinforcement Learning
<https://arxiv.org/abs/2601.17428>`_, with the extensions described below.

The ordinary Go2 flat task keeps its existing command generator. Go2 rough policy
playback also uses ordinary commands, so keyboard and joystick control do not
compete with an adaptive task sampler.

Configuration
-------------

LP-specific integration is grouped under ``velocity/mdp/curriculum/lp/``:

* ``config.py``: LP task discretization and sampler/measurement settings.
* ``commands.py``: velocity commands constrained by the LP-assigned task.
* ``learning_progress.py``: LP task assignment, episode accounting, and sampler updates.
* ``metrics.py``: LP sampling and EPTE-SP diagnostics.
* ``snapshot_probe.py``: first-episode accounting for frozen-policy LP measurement.

``curriculum/catalog.py`` is shared by LP and fixed evaluation.
``curriculum/terminations.py`` is shared by LP and the success-history curriculum.
``curriculum/success.py`` and ``success_history.py`` implement the separate
success-history curriculum. Physical Gait/Velocity telemetry lives in
``velocity/mdp/metrics/`` and is independent of the curriculum algorithm.

The old ``curriculum.config``, ``commands``, ``learning_progress``, ``metrics``,
and ``snapshot_probe`` import paths are deprecated compatibility aliases for saved
configurations. New code should use ``curriculum.lp.<module>``; the runtime
``env.curriculum.learning_progress`` setting and checkpoint state format are unchanged.

Define the terrain geometry in ``ROUGH_TERRAINS_CFG`` and velocity limits in
``commands.base_velocity.ranges`` as usual. Existing terrain rows supply the finite
set of geometric difficulty levels. For example, increasing stair height and
decreasing stair width across their configured ranges describe a family of tasks;
they do not prescribe the order in which LP will sample those tasks.

Go2 rough defaults to the success-history terrain curriculum described in
:doc:`success_terrain_curriculum`. To enable
LP, explicitly configure these reusable components as shown below:

* ``LearningProgressTerrainCurriculum`` connects the sampler to terrain placement,
  completed-episode rewards, and metric logging.
* ``LearningProgressVelocityCommand`` applies the selected command mode and bounds
  throughout an episode.
* ``LearningProgressCfg`` configures task discretization and the LP update cadence.

The defaults are:

.. code-block:: python

   from isaaclab_tasks.core.velocity.mdp.curriculum.lp.config import LearningProgressCfg

   settings = LearningProgressCfg(
       command_bins=(3, 1, 3),
       modes=("standing", "rotating", "moving", "transition"),
       temperature=0.2,
       exploration=0.1,
       reward_scale=15.0,
       min_samples=8,
       update_interval_steps=2500,
       measurement_mode="online_window",
       update_interval_iterations=100,
       evaluation_interval_iterations=100,
       evaluation_seed=101,
       group_sizes=None,
       progress_ema_alpha=1.0,
       max_uniform_kl=0.6931471805599453,
       zero_command_scales=(0.1, 0.1, 0.1),
       command_threshold=1.0e-4,
   )

``command_bins`` splits the configured signed X, Y, and yaw ranges into uniform
intervals. A fixed range produces one interval. Standing tasks have no velocity
bins; rotating tasks use only yaw bins; moving tasks use the applicable velocity
intervals. Transition tasks use the configured full ranges and the existing six
reversal patterns: X, Y, same-sign XY, opposite-sign XY, X+yaw, and Y+yaw. A pattern
is omitted if any active axis lacks both positive and negative limits. These
patterns are predefined command protocols, not automatically discovered behaviors.

The task catalog groups equivalent terrain columns by terrain type. More mesh
columns for one type therefore do not give it extra initial task probability.
Every task initially has equal probability, which does not imply equal probability
for each command mode: moving can have more distinct tasks than standing.
``proportion`` still controls physical mesh-column allocation, so every requested
terrain type needs at least one generated column.

In LP mode, ``heading_command`` and the ``rel_standing_envs``,
``rel_rotating_envs``, and ``rel_transition_envs`` quotas must be disabled. The Go2
rough configuration sets these values. The sampler owns every command-mode
assignment, including standing and transition tasks.

Sampling and episode ownership
------------------------------

The task space is constructed from terrain types, generated difficulty levels,
and valid command tasks. At reset the curriculum selects a task, updates the
terrain origin, and assigns command constraints before the robot reset event.
Command resampling stays within those constraints until the next episode.

Fixed online windows
^^^^^^^^^^^^^^^^^^^^

The default ``measurement_mode="online_window"`` reuses completed training
episodes. It collects their returns for ``update_interval_iterations=100`` PPO
updates and closes the window only after the hundredth PPO update succeeds.
It adds no evaluation rollouts, actor copy, or environment resets.
The current PPO-boundary integration supports single-device RSL-RL training;
distributed training is rejected rather than combining unsynchronized windows.

An episode belongs to the window in which it completes. Its complete accumulated
return includes its terminal step, even if the episode began in the previous
window. Closing a window does not truncate returns or reset ongoing episodes.
The policy continues learning throughout collection, so these are bounded online
measurements, not evaluations of one frozen policy. The completion window is
100 PPO updates; an episode's trajectory can begin before that interval.

For task :math:`\zeta`, let :math:`n_j(\zeta)` be the number of episodes completed
in window :math:`j`. A reliable mean requires at least ``min_samples=8``:

.. math::

   \bar R_j(\zeta) =
   \frac{1}{s n_j(\zeta)}\sum_{i=1}^{n_j(\zeta)} R_i(\zeta),
   \qquad
   LP_j(\zeta) = \bar R_j(\zeta) - \bar R_{j-1}(\zeta).

Here :math:`s` is ``reward_scale``. The signed difference is valid only when
**both immediately adjacent windows** have enough episodes for that task.
At each boundary, every task's pending sum and count are cleared, including tasks
below the threshold. Incomplete windows are never pooled until eight samples
arrive, and a missing previous window cannot be bridged using an older mean.

The first 100 PPO updates establish the first baseline. The first possible
signed LP comparison is after 200 updates, for tasks with reliable means in both
windows. A missing comparison receives a neutral softmax input score of zero,
with an explicit validity mask. This default score is not evidence of measured
zero progress. A reliable mean after a gap establishes a new baseline; its LP
remains missing until the next adjacent reliable mean. Non-finite returns are
errors, not missing samples or zero progress.

After a window closes, the new probabilities apply to subsequent episode resets.
Tasks already assigned to ongoing episodes stay assigned. Sampling does not
guarantee eight episodes per task per window: track counts, valid-comparison
coverage, consecutive missing windows, and probability mass on missing tasks.
The 10 percent uniform mixture and KL limit control the allocation distribution;
they do not guarantee fresh samples for every task or policy improvement.

The paper specifies consecutive mean returns, signed LP, and softmax. The fixed
100-PPO window, eight-episode threshold, missing-data handling, and temperature
are engineering choices, **not paper hyperparameters or validated optimal values**.
Change the online cadence with
``env.curriculum.learning_progress.params.settings.update_interval_iterations=200``.
``update_interval_steps`` is ignored in this mode: LP and optional window-file
diagnostics close together at the PPO boundary. It remains the legacy ``online``
update clock and the ``snapshot`` training-diagnostic cadence. No frozen-policy
measurements run unless explicitly selected.

Optional frozen-policy measurement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set ``measurement_mode="snapshot"`` to evaluate every task with one frozen
stochastic actor. The single-device RSL-RL MLP runner establishes a baseline before
the first PPO update, then evaluates after every
``evaluation_interval_iterations=100`` completed updates. This setting is separate
from ``update_interval_iterations``. Actor parameters and observation-normalization
buffers are copied for each round; evaluation steps never enter PPO storage,
optimization, or training episode statistics.

Each task contributes exactly ``min_samples`` complete episodic returns. Only a
complete finite task-by-repetition matrix updates the baseline and signed LP.
The plan repeats task ordering, cohort seeds, reset/event schedules, and command
ranges. It retains observation noise, command resampling, horizon, reward, and
termination rules; early endings contribute actual returns without extrapolation.

The evaluator reuses the training environment serially. Completed and padding slots
receive zero actions and reset through the normal per-environment path on each
step, preserving live cases' Python, NumPy, and Torch random streams. Live cases
cannot be manually reset. The Newton CPU MuJoCo solver is rejected because its
resets cannot be isolated between environments. After evaluation, training RNG,
counters, and persistent event timers are restored; training episodes are reset
and partial episodes are discarded. The physical backend clock is not rewound.
Use headless training and separately validate custom simulation-time features.

Selecting standing, rotating, and moving modes gives 780 tasks and 6,240
episodes per snapshot round. All four modes give 1,140 tasks and 9,120 episodes.
This adds runtime and is not the default training loop. Fixed panels can overfit;
final performance comparisons still need held-out seeds and the fixed evaluation
suite. The frozen all-task protocol is an implementation extension, not an
execution schedule prescribed by the paper.

The previous ``measurement_mode="online"`` remains available for controlled
ablations. It consumes training episode returns and updates on control steps via
``update_interval_steps``. Its sparse pending samples can span multiple update
windows, so it does not provide the freshness contract of ``online_window``.

Sampling distribution
^^^^^^^^^^^^^^^^^^^^^

All measurement modes turn selection scores :math:`S_j` into probabilities:

.. math::

   q_{j+1}(\zeta) =
   \frac{\exp(S_j(\zeta)/\beta)}
        {\sum_{\zeta'}\exp(S_j(\zeta')/\beta)},
   \qquad
   p_{j+1}(\zeta) = (1-e)q_{j+1}(\zeta) + e/|T|.

In ``online_window`` with the default EMA setting, :math:`S_j` is the signed LP
where its comparison is valid and zero where it is missing. Validity is recorded
separately.
``temperature`` is :math:`\beta` in scaled-return units and ``exploration`` is
:math:`e`. Defaults ``reward_scale=15`` and ``temperature=0.2`` give a raw-return
temperature of 3.0. Scaling changes only LP statistics; PPO receives the original
rewards. The default mixture reserves 10 percent of probability mass for uniform
sampling. ``sampling_strategy="uniform"`` uses a fully uniform training sampler
while retaining the same measurement protocol.

``progress_ema_alpha=1`` uses the latest signed difference directly. Lower values
smooth it as :math:`L_j=\alpha LP_j+(1-\alpha)L_{j-1}` once comparisons exist.
``group_sizes`` optionally fixes each contiguous group's total probability to its
fraction of task IDs. These are supplied-code extensions rather than paper
settings. Neither grouping nor EMA smoothing is enabled by default.

The integration retains ``max_uniform_kl=log(2)`` as a concentration limit.
Since :math:`KL(p\Vert U)=\log(N)-H(p)`, it keeps effective task count at least
:math:`N/2`. It does not guarantee a minimum sample count for each task, constrain
successive distributions directly, or prove policy improvement. Set
``max_uniform_kl=None`` to disable it in an ablation.

Online-window checkpoints store the independent completed-PPO-update count and
the window baseline and validity state. A checkpoint saved exactly at a closed
window boundary can retain a compatible baseline. If it contains a partial
window, resume discards that window and resets all LP statistics, including its
baseline, while keeping policy and completed training-update state. The next
100 updates establish a fresh baseline; another adjacent reliable window is
needed before LP is valid. Unfinished physical episodes restart on resume.
This avoids joining a partial recorded window to restarted simulator episodes.

Snapshot checkpoints separately retain complete evaluation baselines, measurement
policy update numbers, and policy digests. Older algorithm state is not
reinterpreted as a fixed online window. Use ``--reset_curriculum`` when switching
measurement modes or rebuilding incompatible baselines; this keeps model weights
and the selected training state.

Changes to task ranges, rewards, terrain and generation seed,
environment seed, robot and actuator settings, action scaling or wrapper clipping,
physics settings, observation noise, events, terminations, command timing, or
measurement settings require an explicit ``--reset_curriculum`` when they change
a retained measurement contract. Both fixed online-window and snapshot checkpoints
validate wrapper action clipping before restoring curriculum state. Earlier
checkpoints may lack the expanded measurement or runtime contract; resume them
with ``--reset_curriculum`` to rebuild LP baselines.

Detailed file diagnostics are opt-in (``diagnostics=false`` by default).
With ``diagnostics=true``, ``diagnostics/online_windows.jsonl`` records online
window start/end PPO counts, per-task completion counts, means, LP validity,
missing streaks, LP scores, probabilities, and the core scalar metrics. Aggregate coverage and missing-task
probability metrics remain available without detailed debug metrics.
``mean_valid`` marks counts of at least ``min_samples`` (eight by default);
``progress_valid`` requires that condition in both adjacent windows.
``missing_streak`` counts consecutive windows
with an insufficient current mean and resets when the current count is reliable,
even if another window is still needed for a valid LP comparison. Missing means
and progress use numeric zeros accompanied by these masks.
``diagnostics/snapshot_rounds.jsonl`` instead records snapshot per-task counts,
raw and scaled means, standard deviations, standard errors, policy update number,
and actor digest. ``task_windows.jsonl`` records training allocation and episode
outcomes, including online episode start/end control-step summaries for checking
how trajectories cross window boundaries. Snapshot LP does not consume its training samples. Debug metrics label
the snapshot core round clock ``lp_measurement_ticks``.

LP allocates experience within the task space; it does not create obstacle goals
or prove obstacle traversal. A standing task on a central platform measures
standing there, not stair climbing. Geometry and reset distribution still decide
which terrain contacts occur.

Tracking metrics
----------------

The reusable ``EpisodeVelocityTracking`` accumulator implements the stability
penalty from Eq. (8):

.. math::

   \widetilde\epsilon =
   \frac{\epsilon k_f + (K-k_f)}{K}.

Here :math:`K` is the full episode horizon and :math:`k_f` is the number of observed
steps before failure. Remaining steps after failure contribute 100 percent error.
Lower values are better. The implementation uses the clipped component-wise
absolute relative error
:math:`\min(|v-v^*|/|v^*|, 1)` as its per-step error convention. The paper does not
fully specify that convention; it must be considered when comparing results.

The logged quantities deliberately distinguish zero commands:

* ``epte_sp/vx``, ``epte_sp/vy``, and ``epte_sp/yaw`` use relative error only when
  that command axis stays above ``command_threshold`` in magnitude throughout the
  observed episode. Values are percentages.
* ``normalized_epte_sp/*`` extends the metric to zero and mixed commands using
  :math:`\max(|v^*|, s)` as the denominator. ``zero_command_scales`` supplies
  :math:`s` in m/s, m/s, and rad/s. This is an implementation extension.
* ``zero_command_drift/*`` reports mean absolute velocity error on zero-command
  steps in the original physical units.

Failure-terminated and full-horizon episodes can receive EPTE-SP values.
Non-failure partial episodes, including boundary timeouts, are excluded from
EPTE-SP aggregates rather than assigned an artificial perfect score. Drift is
still available for their observed zero-command steps. Validity masks prevent
undefined relative errors or non-finite values from becoming logged scalars.

Training and logging
--------------------

After enabling the LP components below, start a fresh Go2 rough run with both
WandB and local TensorBoard scalar output:

.. code-block:: bash

   uv run isaaclab train --rl_library rsl_rl \
       --task Isaac-Velocity-Rough-UnitreeGo2 \
       --num_envs 4096 \
       --logger wandb --log_project_name go2_lp

The installed RSL-RL WandB writer writes each scalar to both backends. Use
``--logger tensorboard`` instead for local TensorBoard output only. Existing
WandB authentication and project settings apply. Logs remain under
``logs/rsl_rl/unitree_go2_rough/<run>/`` unless the experiment name is overridden.

Legacy ``online`` and optional ``snapshot`` default
``Curriculum/learning_progress/`` output is limited to at most 22 scalar names
for the standard six terrain types: per-axis EPTE-SP and its normalized
extension, EPTE eligibility fractions, failure/boundary-truncation rates, the LP
update counter, maximum task probability, effective task count, uniform KL, and
six terrain sampling probabilities.

The default ``online_window`` mode also logs the following essential measurements
when ``debug_metrics=false``; its output is not subject to the 22-name limit:

* ``lp_last_window_count_min/mean/max`` and ``lp_last_window_episodes`` summarize
  completed samples per task and their total in the last closed window.
* ``lp_current_mean_coverage_fraction`` measures reliable current means;
  ``lp_progress_coverage_fraction`` measures valid adjacent-window LP comparisons.
* ``lp_missing_streak_mean/max`` measures consecutive insufficient-mean windows.
  ``lp_unmeasured_probability_mass`` is allocation mass on tasks without a reliable
  current mean; ``lp_no_progress_probability_mass`` includes every task lacking a
  valid LP comparison, including a reliable first baseline after a gap.
* ``lp_temperature`` is the configured scaled-return temperature;
  ``lp_effective_temperature`` is the temperature after KL limiting. Multiply by
  ``reward_scale`` (15 by default) for raw-return units. The
  ``lp_temperature_capped`` and ``lp_temperature_is_uniform_limit`` flags distinguish
  representational limits from a finite temperature.
* ``lp_window_start_policy_updates`` identifies the current window's start.
  ``lp_update_seconds`` measures LP boundary handling, including validation and
  enabled diagnostics; it is not PPO optimization or an evaluation-rollout time.

Other task reward logs are unaffected.

Enable additional LP summaries, command-mode/difficulty marginals,
terrain-mode combinations, drift, and historical task means only when debugging:

.. code-block:: bash

   uv run isaaclab train --rl_library rsl_rl \
       --task Isaac-Velocity-Rough-UnitreeGo2 --num_envs 4096 \
       --logger wandb --checkpoint latest \
       env.curriculum.learning_progress.params.settings.debug_metrics=true

Alternatively, set ``debug_metrics=True`` in ``LearningProgressCfg``. Its default
is ``False``. This option affects diagnostics only, never LP task selection, and
can change when resuming an existing checkpoint. Existing dashboard history is
not deleted. Running processes need a restart to pick up the new setting.

Debug ``historical_observed_task_macro_normalized_epte_sp/*`` summarizes historical
training episodes and mixes policy versions. It is not a fixed held-out evaluation;
compare policies on the same separate evaluation distribution before claiming an
improvement over another curriculum.

Continue from the latest compatible run with:

.. code-block:: bash

   uv run isaaclab train --rl_library rsl_rl \
       --task Isaac-Velocity-Rough-UnitreeGo2 \
       --num_envs 4096 --logger tensorboard \
       --checkpoint latest

When switching an earlier snapshot or legacy run to fixed online windows, start
from its checkpoint with explicit overrides. An old launch argument such as
``measurement_mode=snapshot`` overrides the new default and must be replaced:

.. code-block:: bash

   uv run isaaclab train --rl_library rsl_rl \
       --task Isaac-Velocity-Rough-UnitreeGo2 \
       --num_envs 4096 --logger tensorboard \
       --checkpoint /path/to/model.pt --reset_curriculum \
       env.curriculum.learning_progress.params.settings.measurement_mode=online_window \
       env.curriculum.learning_progress.params.settings.update_interval_iterations=100 \
       env.curriculum.learning_progress.params.settings.temperature=0.2

``--reset_curriculum`` rebuilds LP statistics while retaining model weights and
the selected training state; the first new complete window establishes a baseline.

An explicit ``--checkpoint /path/to/model_1000.pt`` is also accepted; there is no
separate ``--resume`` flag. RSL-RL checkpoints include curriculum statistics and
validate the task catalog, terrain configuration, and LP settings on restoration.
The sampler saves its private random-generator state for task-draw continuation
on the same CPU/CUDA device backend. Physical simulator state and unfinished
episodes are restarted. A checkpoint
without curriculum state initializes fresh LP statistics and emits a warning.
Legacy sampler checkpoints also start fresh LP statistics while retaining policy
weights; their raw-return history and RNG format cannot be mixed with this sampler.
In explicit online mode, distributed resume discards unsynchronized rank-local pending windows
to avoid duplicating rank-zero samples; synchronized baselines and probabilities
are restored. Training episode metrics are rank-local.

Reusing the components
---------------------

For another manager-based rough velocity environment, add these assignments after
its ordinary configuration has been assembled:

.. code-block:: python

   from isaaclab.managers import CurriculumTermCfg, TerminationTermCfg
   from isaaclab.utils import replace
   from isaaclab_tasks.core.velocity.mdp.curriculum.lp.config import LearningProgressCfg
   from isaaclab_tasks.core.velocity.mdp.curriculum.lp.learning_progress import LearningProgressTerrainCurriculum
   from isaaclab_tasks.core.velocity.mdp.curriculum.terminations import assigned_terrain_boundary

   self.curriculum.terrain_levels = None
   self.curriculum.learning_progress = CurriculumTermCfg(
       func=LearningProgressTerrainCurriculum,
       params={"settings": LearningProgressCfg(), "command_name": "base_velocity"},
   )
   self.commands.base_velocity = replace(
       self.commands.base_velocity,
       class_type="isaaclab_tasks.core.velocity.mdp.curriculum.lp.commands:LearningProgressVelocityCommand",
       rel_standing_envs=0.0,
       rel_rotating_envs=0.0,
       rel_transition_envs=0.0,
       heading_command=False,
   )
   self.scene.terrain.terrain_generator.curriculum = True
   self.terminations.assigned_terrain_boundary = TerminationTermCfg(
       func=assigned_terrain_boundary, time_out=True
   )

This adapter expects a generated terrain grid and a robot with body-frame base
velocity state. It does not require changing the policy observation or reward
terms. The generic sampler, task catalog, and tracking accumulator live in
``isaaclab.utils.curriculum`` and have no dependency on RSL-RL or Go2. The RSL-RL
checkpoint adapter lives separately in
``isaaclab_rl.utils.curriculum_checkpoint``. Other RL libraries can reuse the
same manager serialization interface but must connect their own checkpoint hooks.
Fixed online windows require the completed-PPO-update hook in
``isaaclab_rl.utils.online_curriculum``; other libraries need their own equivalent
window-close integration. Snapshot measurement requires its separate RSL-RL
controller. The snapshot serial controller is scoped to one LP term with no additional
stateful curriculum or recorder terms; custom simulation-time features need
validation because the physics backend clock advances during measurement.

For fixed-policy assessment and controlled uniform/LP comparisons, see
:doc:`locomotion_comparison`.

Recovery without discarding policy weights
-----------------------------------------

Use ``--checkpoint /path/to/model.pt --reset_curriculum`` to restore the policy
and critic while starting fresh LP statistics with the current sampling settings.
The runner retains its normal optimizer-loading behavior; ``--no-resume_optimizer_state``
can disable that separately. Existing runs need a restart to use changed settings.
Training reward under a changing task distribution is not a fixed performance
benchmark; compare saved policies with the fixed evaluator before selecting a
recovery checkpoint.

Using the standard terrain curriculum
-------------------------------------

The standard distance-based terrain curriculum remains an explicit baseline.
Go2 rough currently uses :doc:`success_terrain_curriculum` instead.
Its configuration uses the following assignments instead of LP overrides. This
is different from uniform task sampling.

.. code-block:: python

   self.curriculum.learning_progress = None
   self.curriculum.terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
   self.commands.base_velocity.class_type = mdp.UniformVelocityCommand
   self.scene.terrain.terrain_generator.curriculum = True
   self.terminations.assigned_terrain_boundary = None

The standard command generator uses the configured ``rel_*`` quotas again.
Terrain promotion is checked at episode reset: displacement from the tile origin
greater than half its X size promotes the environment. Insufficient displacement
relative to commanded speed demotes it. This is a predefined curriculum rule,
not a fixed terrain level. Start a separate fresh training run to compare methods;
resuming an LP-trained policy measures fine-tuning instead.

Dashboard grouping
------------------

LP diagnostics are recorded under ``LP/`` in W&B and TensorBoard. Ordinary
terrain metrics remain under ``Curriculum/terrain_levels/``; physical telemetry
remains under ``Gait/`` and ``Velocity/``. This applies to newly started processes.
Existing saved runs retain their previous metric names.
