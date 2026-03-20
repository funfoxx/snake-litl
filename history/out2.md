## Iteration 2 Results

### Summary

This iteration replaced the mostly local egocentric observation with action-conditioned successor-state features for the three relative DQN actions (`left`, `straight`, `right`). The new observation exposes immediate safety, food reachability, path distance, reachable free space, local branching, tail reachability, future safe-move count, and food-progress information for each action.

The result was a better score than iteration 1 on the final `1000`-game benchmark, while still using `stable_baselines3.DQN`.

### Smoke Run

Before the main run, a short validation run was executed with the same configuration but only `60000` timesteps and lighter evaluation:

- artifact: `artifacts/iteration2/smoke1/summary.json`
- best smoke eval score: `4.250402477` at `50000` steps
- smoke benchmark over `200` games:
- avg snake length: `24.335`
- avg steps: `145.615`
- score: `4.066835319`

That was strong enough to justify keeping the new observation design for the full run.

### Main Run Configuration

- algorithm: `stable_baselines3.DQN`
- seed: `7`
- training timesteps: `200000`
- eval frequency: every `5000` steps
- eval episodes per checkpoint: `50`
- final benchmark episodes: `1000`
- policy: `MlpPolicy`
- network: `[256, 256, 128]`
- learning rate: `1e-4`
- replay buffer size: `200000`
- batch size: `256`
- learning starts: `5000`
- exploration fraction: `0.30`
- target update interval: `4000`

### Best Model During Training

- best eval timestep: `115000`
- best eval score: `4.216517024`
- best eval avg snake length: `25.86`
- best eval avg steps: `158.60`

### Final Benchmark

Benchmark of the saved best model over `1000` games:

- avg snake length: `24.978`
- avg steps: `156.250`
- score: `3.992963098`
- max snake length observed: `45`

### Comparison To Iteration 1

- iteration 1 score: `3.616569364`
- iteration 2 score: `3.992963098`
- absolute gain: `0.376393734`
- multiplicative gain: about `1.104x`

Iteration 2 improved the project metric, but not by as much as the highest-smoke estimate suggested. The larger `50`-episode checkpoint evaluations and `1000`-episode final benchmark showed a real but smaller gain.

### Evaluation Curve Notes

Selected checkpoints from the main run:

- `10000` steps: score `3.708642110`
- `45000` steps: score `3.750105952`
- `80000` steps: score `3.820890538`
- `95000` steps: score `3.992945001`
- `115000` steps: score `4.216517024` (best)
- `125000` steps: score `4.064733728`
- `175000` steps: score `3.611038717`
- `200000` steps: score `2.981676669`

The action-aware observation improved the score-efficient regime in the middle of training, but later checkpoints kept growing the snake while using too many extra steps, which reduced `(avg length)^2 / avg steps`. Saving the best checkpoint instead of the final checkpoint remained necessary.

### Tests

Executed:

```bash
XDG_CACHE_HOME=.cache MPLCONFIGDIR=.cache/matplotlib ./.venv/bin/python -m pytest snake/tests/base snake/tests/solver snake/tests/util
```

Result:

- `18 passed, 1 skipped`

### Saved Artifacts

- best model copied to `best/iteration2.zip`
- main training summary written to `artifacts/iteration2/summary.json`
- smoke validation summary written to `artifacts/iteration2/smoke1/summary.json`
