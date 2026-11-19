:orphan:

Locomotion Gait and Velocity Metrics
===================================

Go2 rough and flat RSL-RL training records physical measurements under ``Gait/`` and
``Velocity/``. They use existing robot state, contact sensors, and terrain rays;
they add no rewards, policy observations, random draws, or evaluation rollouts.
The normal configuration emits at most 15 scalars and no additional terminal rows.
The selected native writer records them in TensorBoard, or both TensorBoard and W&B
when ``--logger wandb`` is selected.

Metrics and units
-----------------

.. list-table:: Default measurements
   :header-rows: 1
   :widths: 38 12 50

   * - Tag
     - Unit
     - Definition
   * - ``Gait/base_vz_mean``
     - m/s
     - Signed mean world-Z velocity of the base link.
   * - ``Gait/base_vz_rms``
     - m/s
     - RMS world-Z velocity; upward and downward motion cannot cancel.
   * - ``Gait/swing_height_mean``
     - m
     - Mean foot-link height above locally sampled ground while airborne.
   * - ``Gait/swing_peak_height_mean``
     - m
     - Mean peak height of completed swings with valid ground samples throughout.
   * - ``Gait/swing_time_mean``
     - s
     - Mean contact-sensor air time at touchdown of completed observed swings.
   * - ``Gait/stance_ratio``
     - %
     - Contact foot-time divided by total observed foot-time, multiplied by 100.
   * - ``Gait/roll_rms``, ``Gait/pitch_rms``
     - degrees
     - RMS base tilt relative to world gravity.
   * - ``Gait/stance_foot_speed_mean``
     - m/s
     - Mean horizontal world speed of contacting foot links, a slip proxy.
   * - ``Velocity/x_mae``, ``Velocity/y_mae``
     - m/s
     - Mean absolute body-frame velocity tracking errors.
   * - ``Velocity/x_rmse``, ``Velocity/y_rmse``
     - m/s
     - Root mean squared body-frame velocity tracking errors.
   * - ``Velocity/yaw_mae``, ``Velocity/yaw_rmse``
     - rad/s
     - Absolute and RMS body-frame yaw-rate tracking errors.

Velocity errors compare the post-physics state with the command that was applied
during that control step, before resets and command resampling. They use no reward
weights, exponential transformation, or division by commanded speed. Zero commands
therefore remain valid measurements. Signed errors from different robots do not
cancel. Samples are pooled by observed time, rather than averaging per-episode RMS.

Swing duration and peak are pooled by completed swing events. A swing must begin
with an observed stance-to-air transition and end with an observed touchdown.
Initial airborne fragments and reset-truncated swings do not count as complete
swings. A logging boundary preserves ongoing swings. Mean airborne height can
include valid samples from partial swings. No completed event or valid height
means the corresponding scalar is omitted for that window, rather than set to zero.

Height is the **foot-link origin height**, not calibrated sole or toe clearance.
On rough terrain, the nearest finite XY ray hit from the existing height scanner
is used only within ``max_ground_sample_distance`` (default 0.1 m). Missing coverage
has no plane-zero fallback. The 0.1 m ray grid approximates local terrain and can
misrepresent sharp stair edges. An explicitly configured plane uses ``plane_z``.
Contact uses the existing sensor's normal-force threshold, without adding filtering.

Configuration and training
--------------------------

The reusable configuration lives in
``isaaclab_tasks.core.velocity.mdp.metrics.LocomotionMetricsCfg``. Go2 enables it
through ``UnitreeGo2RoughEnvCfg.locomotion_metrics``; flat inherits that setting.
Other compatible velocity tasks can add the same config explicitly.

.. code-block:: python

   from isaaclab_tasks.core.velocity.mdp.metrics import LocomotionMetricsCfg

   locomotion_metrics: LocomotionMetricsCfg = LocomotionMetricsCfg(
       enabled=True,
       debug_metrics=False,
       max_ground_sample_distance=0.1,
       playback_interval_steps=24,
   )

Existing RSL-RL train commands need no additional flag. Values are reported once
per PPO rollout/update, with the same iteration axis as the training logger.
``env.locomotion_metrics.enabled=false`` disables collection.
``env.locomotion_metrics.debug_metrics=true`` additionally records per-foot metrics,
completed swing counts, valid peak counts, and valid airborne-height coverage in
``Gait/debug`` (coverage is a fraction from 0 to 1). Debug output is disabled by default.
Frozen-policy LP probes are excluded from training telemetry. Multi-GPU logging
uses the writer process's environments, not a reduction across all GPU shards.

Collection adds GPU tensor operations each control step. Rough-terrain ground
lookup scales with environment count, feet, and existing ray count; scalar transfer
to CPU occurs when the reporting window closes. No additional simulation is run.

Measure an existing policy
--------------------------

Playback is opt-in and performs no PPO updates:

.. code-block:: bash

   uv run isaaclab play --rl_library rsl_rl \
     --task Isaac-Velocity-Rough-UnitreeGo2 \
     --checkpoint /absolute/path/to/model_9999.pt \
     --num_envs 16 --viz newton_gl --real-time \
     --log_metrics --logger tensorboard

Add ``--gamepad`` or ``--keyboard`` to measure manual commands. Add
``--metrics_steps 3000`` to end after a bounded number of control steps.
For W&B, replace the logger options with
``--logger wandb --log_project_name go2_policy_evaluation``.
W&B playback creates a separate run and also writes local TensorBoard events.

The directory printed at startup is
``<checkpoint-directory>/evaluations/<timestamp>_<model-name>/``. It includes
``checkpoint.txt`` with the exact source checkpoint and ``params/`` with the
resolved environment and agent configurations. Playback's horizontal axis
is the number of measured control steps; windows default to 24 steps (0.48 s at
the Go2 50 Hz control rate). The final partial window is flushed on normal exit
or Ctrl+C. Training and playback series are separate runs.

.. code-block:: bash

   uv run tensorboard --logdir /absolute/path/to/checkpoint-directory/evaluations

Open the URL printed by TensorBoard and expand ``Gait`` and ``Velocity``.
Previously completed training runs cannot retrospectively acquire these measurements;
replay their checkpoints or start a new process with this code.

Interpretation
--------------

Lower tracking errors indicate closer command following. More swing clearance or
longer swing time is not universally better, and climbing intentionally increases
vertical body speed. These groups describe gait and tracking, not terrain success.
Compare policies under the same terrain levels, commands, seeds, and evaluation
duration: curriculum changes can change the pooled mean without a policy regression.
