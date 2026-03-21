## Iteration 2 results

Iteration 2 kept the DQN formulation from iteration 1, but expanded the observation with trap-aware structural features and changed the training pipeline to save periodic checkpoints and evaluate them with snake-specific metrics.

Main run configuration:

- algorithm: `stable_baselines3.DQN`
- policy: `MlpPolicy`
- observation: `46`-dim egocentric feature vector
- actions: relative `left / straight / right`
- timesteps: `20000`
- eval frequency: `5000`
- eval episodes per checkpoint: `20`
- checkpoint-selection episodes: `200`
- final benchmark episodes: `1000`
- learning rate: `1e-4`
- exploration fraction: `0.25`
- network: `[256, 256, 256, 128]`
- seed: `7`

What changed in the environment:

- kept the iteration-1 danger bits, relative food coordinates, length feature, and 8 directional rays
- added current reachable-space and food-reachability features
- added tail-relative coordinates
- added one-step lookahead features for each relative action:
  - blocked or not
  - next food distance
  - reachable-space ratio
  - food reachability
- kept the iteration-1 reward structure and added a mild reachable-space delta term

Checkpoint evaluations during training:

- `5000` timesteps: avg length `2.000`, avg steps `56.000`, score `0.035714`
- `10000` timesteps: avg length `22.800`, avg steps `146.800`, score `0.155313`
- `15000` timesteps: avg length `22.650`, avg steps `135.350`, score `0.167344`
- `20000` timesteps: avg length `23.000`, avg steps `138.900`, score `0.165587`

Best model selection:

- per the iteration plan, checkpoint selection prioritized:
  - higher average snake length
  - then lower average steps
  - then higher score
- on the `200`-episode selection pass, the `10000`-step checkpoint had the highest average length and was selected as the best model
- selected checkpoint metrics: avg length `23.725`, avg steps `155.745`, score `0.152332`
- training time: `24.48s`
- copied to `best/iteration2.zip`

Final benchmark over `1000` games:

- avg snake length: `23.575`
- avg steps: `154.932`
- avg return: `290.676`
- median snake length: `24`
- median steps: `153.5`
- max snake length: `41`
- min steps: `12`
- score: `0.152164`

Comparison vs iteration 1:

- avg snake length improved from `20.233` to `23.575` (`+3.342`, about `16.52%`)
- avg steps worsened from `120.153` to `154.932` (`+34.779`, about `28.94%`)
- score fell from `0.168394` to `0.152164`

Takeaways:

- the richer lookahead observation reliably increased snake length relative to iteration 1
- the same change also made the learned policy more willing to spend extra steps staying safe, so efficiency regressed
- within this run, later checkpoints around `15000-20000` had better score than the selected `10000` checkpoint, but they did not exceed it on average length
- the main unresolved problem for the next iteration is recovering iteration-1 efficiency without giving back the new length gains
