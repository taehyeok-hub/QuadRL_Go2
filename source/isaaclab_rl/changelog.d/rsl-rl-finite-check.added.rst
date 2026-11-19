* Added opt-in ``--check_finite`` diagnostics for single-device RSL-RL PPO training to
  detect NaN and infinity in environment states, learning targets, gradients and
  optimizer state, and save a diagnostic snapshot before stopping.
* Added bounded ``--record_action_history`` diagnostics to preserve recent policy
  inputs, actions, reset boundaries, model versions and the latest PPO update inputs
  before stopping on an oversized raw action or a nonfinite value.
