* Switched the default velocity LP curriculum to 100-PPO online windows at temperature
  0.2, retaining uniform exploration and the KL concentration limit.
* Discarded partial LP windows and incompatible baselines on resume while preserving
  ongoing episodes during ordinary window updates and including failure returns.
