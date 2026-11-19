* Extended snapshot LP resume validation to action, robot, physics, and seed settings
  so environment changes could not reuse incompatible progress baselines.
* Retired completed snapshot cases through normal environment resets and changed
  the default measurement interval to 100 completed PPO updates.
