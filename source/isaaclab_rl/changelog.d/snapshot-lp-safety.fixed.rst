* Fixed frozen-policy LP evaluation to reset completed slots safely without consuming
  live-case random streams, and rejected solvers without isolated environment resets.
* Validated action clipping before restoring snapshot LP baselines and changed the
  default comparison cadence to 100 completed PPO updates.
