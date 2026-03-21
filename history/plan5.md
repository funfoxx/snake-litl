## Iteration 5 plan

Iteration 4 preserved more snake length than iteration 3, but it still peaked before the end of training and finished with a slightly worse score (`0.164717`) than iteration 3 (`0.166265`). The iteration-4 notes already point at the likely issue: the asymmetric trap shaping is useful, but the DQN training schedule is still letting the policy drift after the best efficiency/length balance has been reached.

Hypothesis:

- the iteration-4 reward design is close enough that another large shaping rewrite is more likely to thrash than to help
- fixed optimizer pressure plus coarse `5000`-step checkpoint spacing make it too easy to miss a narrow performance peak
- a schedule-focused iteration should be lower risk:
  - decay the learning rate over training so later updates are smaller
  - reduce terminal exploration so replay data gets less noisy once the policy is competent
  - evaluate more frequently and use a larger checkpoint-selection pass so the chosen model better matches the final benchmark

This iteration will keep the model family as DQN and focus on four changes:

1. Keep the iteration-4 environment and observation fixed.
   - Preserve the `46`-feature egocentric state.
   - Preserve relative `left / straight / right` actions.
   - Preserve the asymmetric trap shaping introduced in iteration 4.

2. Make the DQN schedule more tunable from `train_test.py`.
   - Add support for a linearly decaying learning rate.
   - Expose terminal exploration epsilon as a command-line parameter.
   - Expose target-network update interval as a command-line parameter so probes can slightly stabilize value updates if needed.

3. Increase checkpoint resolution and selection reliability.
   - Use denser periodic evaluation (`2500` timesteps) in probes and the main run.
   - Increase checkpoint-selection episodes for the main run so the chosen checkpoint is less sensitive to evaluation noise.
   - Keep the existing score-first selection rule because it was still the best selector across iterations 3 and 4.

4. Run short schedule-focused probes before one main run.
   - Keep reward coefficients fixed to the iteration-4 main configuration.
   - Compare schedule variants rather than mixing in a new reward search.

Planned probe settings:

- common environment coefficients:
  - step penalty: `-0.027`
  - closer reward: `+0.25`
  - farther penalty: `-0.38`
  - space-loss scale: `0.20`
  - food-unreachable penalty: `-0.20`
  - safe-progress bonus: `+0.15`
  - safe-progress margin: `0.10`
  - revisit penalty: `-0.22`
  - starvation limit: `32 + 6 * snake_len`
  - starvation penalty: `-3.0`
  - no-progress threshold: `6`
  - no-progress penalty: `-0.12`

- common DQN settings unless overridden:
  - algorithm: `stable_baselines3.DQN`
  - policy: `MlpPolicy`
  - network: `[256, 256, 256, 128]`
  - buffer size: `100000`
  - learning starts: `5000`
  - batch size: `256`
  - gamma: `0.99`
  - train frequency: `4`
  - gradient steps: `1`
  - eval frequency: `2500`
  - probe timesteps: `12000`
  - probe benchmark episodes: `200`
  - checkpoint selection rule: higher score, then higher average length, then lower average steps

- candidate A:
  - learning rate: `1e-4 -> 3e-5`
  - exploration fraction: `0.20`
  - exploration final epsilon: `0.01`
  - target update interval: `2000`

- candidate B:
  - learning rate: `1e-4 -> 2e-5`
  - exploration fraction: `0.18`
  - exploration final epsilon: `0.005`
  - target update interval: `1500`

- candidate C:
  - learning rate: `8e-5 -> 2e-5`
  - exploration fraction: `0.20`
  - exploration final epsilon: `0.005`
  - target update interval: `1500`

Main-run target:

- promote the best probe by score, then average length
- run one main training at `20000` timesteps unless probes clearly show an earlier peak that is already stable
- evaluate checkpoints every `2500` timesteps
- use `400` selection episodes in the main run
- benchmark the selected checkpoint over `1000` games

Success criterion:

- improve the benchmark score above iteration 4 (`0.164717`)
- ideally recover or beat iteration 3 (`0.166265`)
- keep average snake length around iteration-4 territory while pulling average steps back down
