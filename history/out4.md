## Iteration 4 Results

### Summary

This iteration started by testing two larger changes:

- n-step returns on top of the custom dueling / double / prioritized `DQN`
- extra efficiency-focused observation and reward shaping

Those smoke runs did not generalize well enough on the benchmark, so they were not carried into the final model. The winning change was more conservative:

- keep the iteration-3 environment and 1-step `DQN`
- separate checkpoint-evaluation seeds from final benchmark seeds
- use a larger evaluation set during checkpoint selection

That improved the full `1000`-game benchmark over iteration 3.

### What Changed In The Final Main Run

1. Checkpoint selection no longer reused the benchmark seed prefix.
   - training-time evals used seeds starting at `10000`
   - the final benchmark still used seeds starting at `0`

2. The final run used a slightly shorter training horizon with stronger eval filtering.
   - this reduced how much the model could overfit to later high-step survival regimes before checkpoint selection

3. The trainer stayed inside the DQN-only constraint.
   - custom `DQN`
   - dueling network
   - double-DQN target selection
   - prioritized replay
   - `n_step = 1` in the final winning run

### Smoke Validation

The early iteration-4 smokes (`smoke1` through `smoke4`) showed that n-step returns and stronger efficiency shaping looked promising on small eval slices but underperformed on the benchmark.

The final validation run was `artifacts/iteration4/smoke5/summary.json`, which kept the iteration-3 environment but used disjoint eval seeds:

- training timesteps: `60000`
- eval episodes per checkpoint: `50`
- benchmark episodes: `200`

Best smoke checkpoint:

- best eval timestep: `55000`
- best eval score: `4.546543795`

Smoke benchmark:

- avg snake length: `29.585`
- avg steps: `204.335`
- score: `4.283515917`
- max snake length observed: `47`

That was better than iteration 3's smoke benchmark (`4.232655671`), so the same selection strategy was promoted to the main run.

### Main Run Configuration

- algorithm: custom `DQN`
- seed: `7`
- training timesteps: `70000`
- eval frequency: every `5000` steps
- eval episodes per checkpoint: `100`
- eval seed base: `10000`
- final benchmark seed base: `0`
- final benchmark episodes: `1000`
- network: dueling MLP `[256, 256, 128]`
- learning rate: `3e-4`
- replay buffer size: `150000`
- batch size: `256`
- learning starts: `5000`
- train frequency: every `4` env steps
- target update interval: `2500`
- exploration fraction: `0.35`
- exploration epsilon: `1.0 -> 0.03`
- prioritized replay alpha: `0.6`
- prioritized replay beta schedule: `0.4 -> 1.0`
- n-step returns: `1`

### Best Model During Training

- best eval timestep: `60000`
- best eval score: `4.279636271`
- best eval avg snake length: `30.43`
- best eval avg steps: `216.37`

### Final Benchmark

Benchmark of the saved best model over `1000` games:

- avg snake length: `30.700`
- avg steps: `220.912`
- score: `4.266359455`
- max snake length observed: `58`

### Comparison To Iteration 3

- iteration 3 score: `4.073235667`
- iteration 4 score: `4.266359455`
- absolute gain: `0.193123788`
- multiplicative gain: about `1.047x`

The model is still not as step-efficient as the repository's original documented DQN benchmark, but it is a real improvement over the previous LTIL iteration.

### Evaluation Curve Notes

Selected checkpoints from the main run:

- `30000` steps: score `4.053686102`
- `40000` steps: score `4.118734090`
- `50000` steps: score `4.144598888`
- `55000` steps: score `4.146311179`
- `60000` steps: score `4.279636271` (best)
- `65000` steps: score `4.265461538`
- `70000` steps: score `3.783257404`

The same late-training failure mode remained, but the better checkpoint-selection procedure found a stronger mid-to-late checkpoint than iteration 3 did.

### Tests

Executed:

```bash
XDG_CACHE_HOME=.cache MPLCONFIGDIR=.cache/matplotlib ./.venv/bin/python -m pytest snake/tests/base snake/tests/solver snake/tests/util
```

Result:

- `18 passed, 1 skipped`

### Saved Artifacts

- best model copied to `best/iteration4.zip`
- main training summary written to `artifacts/iteration4/summary.json`
- final validation summary written to `artifacts/iteration4/smoke5/summary.json`
