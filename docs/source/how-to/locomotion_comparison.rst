:orphan:

Diagnosing Locomotion Curricula
==============================

Use three separate tools: controlled training experiments, a fixed-policy
tracking/stair suite, and local task-window diagnostics. A low training failure
rate is not evidence of broad terrain competence when task probabilities change.
No module below changes reward weights or claims improved locomotion performance.

Prepare comparison training
---------------------------

.. code-block:: bash

   uv run python -m isaaclab_rl.evaluation.compare prepare \
       --output experiments/go2_curriculum_v1 \
       --num-envs 4096 --iterations 10000 \
       --seeds 42 --variants uniform lp --logger wandb

This writes ``experiments.json``, a shared ``evaluation_env.yaml``, resolved settings, ``run_train.sh``, and
``run_evaluate.sh``. It does not launch long jobs. Execute:

.. code-block:: bash

   bash experiments/go2_curriculum_v1/run_train.sh
   bash experiments/go2_curriculum_v1/run_evaluate.sh

Generated training scripts reject changes to the selected training sources after
preparation. Prepare a new plan after intentional code changes. Starting actor and
critic tensor hashes are saved in ``diagnostics/initial_policy.json`` before the
first learning update (or after explicit checkpoint restoration).

Generated jobs use separate experiment names. For interactive playback, select
the resulting model with an explicit checkpoint path; the ordinary task
``--checkpoint latest`` selector still searches its original experiment folder.

Both variants start fresh and use the same task space, rewards, observations,
network, seed list, environment count, and iteration budget. Generated evaluations require the requested final iteration count, so interrupted
training checkpoints are rejected. Ctrl+C stops the training script with status
130 instead of continuing to the next variant.

``uniform`` still
measures signed LP but keeps sampling probabilities uniform. ``lp`` reproduces
the supplied LP sampler: temperature=0.1, reward_scale=15, and 10 percent
uniform exploration, with eight episodes per adequate window and a 2500-control-step
update interval. The softmax part has raw-return temperature 1.5. Evaluation defaults to seed 101 and 128 environments;
use ``--evaluation-seeds 101 202 303`` to test additional terrain realizations.
The default plan has one training seed and two jobs, each with 10,000 iterations.
Use ``--seeds 42 43 44`` for three independent runs per variant after the initial
comparison; this expands the plan to six training jobs. A 300-iteration smoke run
only checks wiring and gross failures, not curriculum quality or convergence.

Additional one-variable ablations are ``lp_beta10`` (temperature=10),
``lp_raw_return`` (reward_scale=1), ``lp_no_exploration`` (no uniform mixture),
and ``lp_min32`` (32 samples per adequate LP window). These are diagnostic experiments, not validated
fixes or recommended optimal values. ``--overrides`` applies common Hydra
settings to all jobs. Keep geometry, policy, and reward settings identical across
variants while identifying the sampling failure.

Audit task sampling
-------------------

.. code-block:: bash

   uv run python -m isaaclab_rl.evaluation.audit \
       --checkpoint logs/rsl_rl/EXPERIMENT/RUN/model_19999.pt \
       --windows logs/rsl_rl/EXPERIMENT/RUN/diagnostics/task_windows.jsonl \
       --output experiments/go2_curriculum_v1/audit.json

Omit ``--windows`` for old runs that did not record diagnostics. Historical reward
variance cannot be recovered from a checkpoint that only stores sums and counts.
The audit reports entropy, effective task count, peak/top-ten probability mass,
and the dominant terrain/difficulty/command tasks. Window diagnostics add raw
raw return versus scaled LP-score means/deviations, episode lengths, terminations, and
weighted time-integrated reward components, indexed by task. A companion
``task_catalog.json`` maps task IDs to the configured families.

Diagnostics are enabled by prepared experiments, or manually with:

.. code-block:: text

   env.curriculum.learning_progress.params.settings.diagnostics=true

These files do not add scalar dashboard panels. They are bounded sufficient
statistics per fixed update window, not unbounded per-step histories. In
multi-GPU training, diagnostic episode statistics are rank-local while sampling
probabilities and LP use synchronized global sufficient statistics. Do not
interpret one rank's standard deviations as global standard deviations.

Fixed-policy evaluation
-----------------------

.. code-block:: bash

   uv run python -m isaaclab_rl.evaluation.locomotion \
       --checkpoint logs/rsl_rl/EXPERIMENT/RUN/model_19999.pt \
       --output logs/evaluation/policy_19999_seed101 \
       --num-envs 128 --seed 101 --protocol both

Use ``--env-config /path/to/evaluation_env.yaml`` to evaluate all policies on
one explicitly frozen environment, even if their training terrain ranges differ.
Prepared evaluation commands include this option automatically.

The evaluator requires the checkpoint's saved ``params/env.yaml`` and
``params/agent.yaml``. It restores these configurations, then explicitly replaces
training curricula, random events, observation noise, reset poses, and command
resampling with the evaluation protocol. Inference is deterministic PPO/MLP
inference: no policy updates, LP updates, exploration actions, or curriculum-state
restoration occur. Newton and the saved physics settings remain in use.
This first adapter targets the existing manager-based Go2 PPO task; reusable
catalog, geometry, statistics, and summary components are independent modules.

``tracking`` visits each discrete task once with fixed within-bin 25%/75%
representatives; transition tasks use fixed phase, relative yaw sign, and timing.
For the standard catalog there are 1,140 tracking cases. This samples the finite
catalog, not every possible continuous command inside every bin.

``traversal`` adds 60 stair cases: two stair types, ten difficulty levels, and
three headings (0, 45, and 90 degrees). All headings for a given terrain/level use
the same mesh column. The robot starts at the central platform at its default
joint pose with zero velocity, and receives a forward 0.5 m/s command. Ordinary
pyramid stairs test descent; inverted pyramid stairs test ascent. Success requires
forward progress to the stair-region edge, all four feet beyond that region by
5 cm, a tilt below 45 degrees for at least 0.5 seconds, and no failure before the
recorded episode end. This is an explicit task metric, not the curriculum's
promotion rule. It does not measure ascent for arbitrary random spawn positions.

Boundary exits are timeouts, not falls. EPTE-SP and the zero/mixed-command
normalized extension use the existing documented convention. Non-failure early
boundary episodes are censored; summaries preserve valid counts rather than
scoring these episodes as perfect. Always inspect boundary rates and coverage
alongside conditional error means. Stair success includes censored boundary
cases if the safe exit condition was met.

Outputs include ``suite.json``, ``manifest.json``, ``episodes.jsonl``, and
``summary.json`` with protocol/terrain/difficulty/mode groups. Manifests record
checkpoint hashes, source hashes, fixed evaluation configuration, and a contract
hash. Evaluation output directories must be new to prevent mixing episodes from
different policies. The simulator may have small numerical nondeterminism; fixed
seeds and cases establish controlled conditions, not bitwise reproducibility.

For a short connectivity check, use ``--episode-seconds 0.5`` and a few
``--case-ids`` from the suite. This is not a valid stair competence evaluation.
For visual failure inspection, rerun selected cases with ``--viz newton_gl`` and
``--trace-case-ids``. Traces contain root state, commands, gravity, joints, and
foot positions before resets; they do not claim to be rendered videos.

Compare completed reports
-------------------------

.. code-block:: bash

   uv run python -m isaaclab_rl.evaluation.compare report \
       --results logs/evaluation/uniform_s42_e101 logs/evaluation/lp_s42_e101 \
       --output experiments/go2_curriculum_v1/comparison.json

Only matching evaluation contracts are grouped together. Different seeds,
geometry, case lists, physics settings, or source versions remain separate.
Compare protocol-specific failure/error/coverage and stair success, then inspect
task-window evidence before changing the sampler or reward. A single evaluation
seed or training seed is a diagnostic baseline, not a statistical performance
claim. The tools prepare a comparison; they do not determine causality from one
policy or guarantee that a proposed ablation will succeed.

Resume and evaluator limits
---------------------------

LP checkpoints record reward configuration, episode horizon, and control step
size. Changes to these conditions reject LP-statistic restoration. Legacy
checkpoints without this contract retain their policy weights but start fresh LP
statistics with a warning. The evaluator currently requires MLP actors, and its
comparison contract includes action clipping and the common runtime source hash.
