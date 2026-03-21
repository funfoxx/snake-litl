## Iteration 3 Results

### Summary

This iteration replaced the vanilla SB3 DQN trainer with a custom PyTorch DQN loop while staying within the DQN-only constraint. The new trainer uses:

- dueling value / advantage heads
- double-DQN target selection
- prioritized replay
- score-based checkpointing

The environment also kept the relative 3-action successor-state observation from iteration 2, but added extra successor-space features and switched reward shaping from mostly Manhattan-progress feedback to stronger shortest-path and anti-wandering pressure.

The final result improved the project benchmark over iteration 2.

### Smoke Run

A validation run was executed before the main run:

- artifact: `artifacts/iteration3/smoke3/summary.json`
- training timesteps: `60000`
- eval episodes per checkpoint: `20`
- benchmark episodes: `200`

Best smoke checkpoint:

- best eval timestep: `45000`
- best eval score: `4.316928105`
- best eval avg snake length: `25.70`
- best eval avg steps: `153.00`

Smoke benchmark:

- avg snake length: `26.505`
- avg steps: `165.975`
- score: `4.232655671`
- max snake length observed: `47`

That was materially better than iteration 2's smoke benchmark (`4.066835319`), so the same trainer/config was kept for the main run.

### Main Run Configuration

- algorithm: custom `DQN`
- seed: `7`
- training timesteps: `150000`
- eval frequency: every `5000` steps
- eval episodes per checkpoint: `50`
- final benchmark episodes: `1000`
- network: dueling MLP `[256, 256, 128]`
- learning rate: `3e-4`
- replay buffer size: `150000`
- batch size: `256`
- learning starts: `5000`
- train frequency: every `4` env steps
- target update interval: `2500`
- exploration fraction: `0.35`
- prioritized replay alpha: `0.6`
- prioritized replay beta schedule: `0.4 -> 1.0`

### Best Model During Training

- best eval timestep: `55000`
- best eval score: `4.179222154`
- best eval avg snake length: `26.06`
- best eval avg steps: `162.50`

### Final Benchmark

Benchmark of the saved best model over `1000` games:

- avg snake length: `26.447`
- avg steps: `171.717`
- score: `4.073235667`
- max snake length observed: `50`

### Comparison To Iteration 2

- iteration 2 score: `3.992963098`
- iteration 3 score: `4.073235667`
- absolute gain: `0.080272569`
- multiplicative gain: about `1.020x`

This is a modest gain, but it is a real improvement on the full benchmark.

### Evaluation Curve Notes

Selected checkpoints from the main run:

- `15000` steps: score `3.950025189`
- `30000` steps: score `3.999092833`
- `55000` steps: score `4.179222154` (best)
- `90000` steps: score `4.055162157`
- `120000` steps: score `3.280072907`
- `150000` steps: score `2.891225755`

The same late-training failure mode from iteration 2 remained: the agent kept learning to survive and grow, but started taking far too many extra steps per unit of growth. The custom DQN trainer did improve the best score-efficient regime, but checkpoint selection was still essential.

### Tests

Executed:

```bash
XDG_CACHE_HOME=.cache MPLCONFIGDIR=.cache/matplotlib ./.venv/bin/python -m pytest snake/tests/base snake/tests/solver snake/tests/util
```

Result:

- `18 passed, 1 skipped`

### Saved Artifacts

- best model copied to `best/iteration3.zip`
- main training summary written to `artifacts/iteration3/summary.json`
- smoke validation summary written to `artifacts/iteration3/smoke3/summary.json`
