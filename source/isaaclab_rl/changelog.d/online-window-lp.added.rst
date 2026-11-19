* Added an online LP controller that closed measurement windows after successful PPO
  updates without extra evaluation episodes, actor copies, or environment resets.
* Included wrapper action clipping in online-window checkpoint compatibility checks.
