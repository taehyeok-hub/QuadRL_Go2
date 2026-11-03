* **Breaking:** Replaced the shared velocity task's ``policy`` observations with ``actor_obs``,
  ``critic_base_obs``, and ``privileged_obs`` groups. Actors receive proprioceptive inputs without
  simulator linear velocity or height scans; critics additionally receive those privileged inputs
  without observation noise. Updated the registered RSL-RL configurations and flat-terrain settings
  for Go2, G1, H1, Cassie, and ANYmal D. Custom integrations must use the new group names, and policies
  trained with the previous observation layout require retraining.
* Restored ANYmal D's foot air-time reward and contact-body selectors in its robot-specific reward
  configuration so that the flat and rough environment configurations can be constructed independently
  of Go2's customized shared defaults.
