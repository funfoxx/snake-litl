## Iteration 3 plan

Iteration 2 improved average snake length from `20.233` to `23.575`, but it also increased average steps from `120.153` to `154.932`. The observation upgrade looks useful, so the next iteration should keep the trap-aware features and instead push the DQN toward faster food collection and better checkpoint selection.

This iteration will keep the model family as DQN and focus on efficiency recovery:

1. Keep the iteration-2 observation, but make the environment tunable from the training script.
   - Preserve the `46`-dim egocentric observation and relative 3-action control.
   - Expose reward and starvation-limit coefficients so short probes can compare efficiency-oriented settings without rewriting the environment each time.

2. Rebalance reward shaping toward decisive progress.
   - Keep the large positive reward for food and the strong death penalty.
   - Increase the pressure against wasted movement by tuning:
     - per-step cost
     - move-away-from-food penalty versus move-toward-food reward
     - revisit penalty
     - reachable-space bonus scale
   - Tighten the starvation budget so policies that wander safely for too long get cut off earlier.
   - Add a mild no-progress penalty when the agent spends several consecutive moves without reducing food distance or eating.

3. Change best-checkpoint selection so efficiency matters directly.
   - Iteration 2 selected the `10000`-step checkpoint because it had the highest average length, even though later checkpoints had much better step efficiency and score.
   - For iteration 3, compare checkpoints with:
     - primary: higher score (`avg_length / avg_steps`)
     - secondary: higher average snake length
     - tertiary: lower average steps
   - This should better align the chosen model with the project objective of increasing length while reducing steps.

4. Use short probes before the main run.
   - Run several `10000`-step probes with `200` benchmark episodes to compare candidate shaping settings.
   - Promote the best probe configuration to one main run.

Planned probe settings:

- common DQN settings:
  - algorithm: `stable_baselines3.DQN`
  - policy: `MlpPolicy`
  - network: `[256, 256, 256, 128]`
  - learning rate: `1e-4`
  - buffer size: `100000`
  - learning starts: `5000`
  - batch size: `256`
  - gamma: `0.99`
  - train frequency: `4`
  - gradient steps: `1`
  - target update interval: `2000`
  - exploration fraction: `0.20` to `0.25`
  - exploration final epsilon: `0.02`

- probe focus:
  - slightly larger step penalty than iteration 2
  - stronger penalty for moving away from food than reward for moving closer
  - smaller reachable-space bonus so the agent does not overpay for extreme caution
  - smaller starvation budget than iteration 2

Initial candidate values to test:

- candidate A:
  - step penalty: `-0.03`
  - closer reward: `+0.25`
  - farther penalty: `-0.35`
  - reachable-space delta scale: `0.10`
  - revisit penalty: `-0.25`
  - no-progress threshold: `6`
  - no-progress penalty: `-0.15`
  - starvation limit: `30 + 6 * snake_len`
  - exploration fraction: `0.20`

- candidate B:
  - step penalty: `-0.025`
  - closer reward: `+0.25`
  - farther penalty: `-0.35`
  - reachable-space delta scale: `0.15`
  - revisit penalty: `-0.25`
  - no-progress threshold: `8`
  - no-progress penalty: `-0.12`
  - starvation limit: `32 + 6 * snake_len`
  - exploration fraction: `0.20`

- candidate C:
  - step penalty: `-0.03`
  - closer reward: `+0.20`
  - farther penalty: `-0.40`
  - reachable-space delta scale: `0.10`
  - revisit penalty: `-0.30`
  - no-progress threshold: `6`
  - no-progress penalty: `-0.18`
  - starvation limit: `28 + 6 * snake_len`
  - exploration fraction: `0.18`

Main-run target:

- choose the best probe by score first, then average length
- train one main run for `20000` timesteps unless the probes clearly show that a shorter run is already peaking
- evaluate checkpoints every `5000` timesteps
- select the best main-run checkpoint with the new score-first rule
- benchmark the selected checkpoint over `1000` games

Success criterion:

- beat iteration 2 on average steps by a meaningful margin
- keep average snake length at or above iteration 1
- ideally improve the overall score above both iteration 1 (`0.168394`) and iteration 2 (`0.152164`)
