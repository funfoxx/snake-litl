## Iteration 1 Results

### Summary

This iteration replaced the raw board observation with a compact egocentric feature vector, switched to relative left/straight/right controls to remove the reverse-direction stall action, added reward shaping plus a starvation cutoff, and trained a longer SB3 DQN run with score-based checkpoint selection.

### Main Run Configuration

- algorithm: `stable_baselines3.DQN`
- seed: `7`
- training timesteps: `200000`
- eval frequency: every `5000` steps
- eval episodes per checkpoint: `50`
- final benchmark episodes: `1000`
- policy: `MlpPolicy`
- network: `[256, 256]`

### Best Model During Training

- best eval timestep: `145000`
- best eval score: `3.872327106`
- best eval avg snake length: `25.02`
- best eval avg steps: `161.66`

### Final Benchmark

Benchmark of the saved best model over `1000` games:

- avg snake length: `23.317`
- avg steps: `150.331`
- score: `3.616569364`
- max snake length observed: `38`

### Comparison To Iteration 0

- iteration 0 score: `0.000441791399`
- iteration 1 score: `3.616569364`
- absolute gain: `3.616127572605`
- multiplicative gain: about `8186x`

The new setup eliminated the previous failure mode where episodes drifted near the `10000`-step truncation limit while barely growing.

### Evaluation Curve Notes

Selected checkpoints from the main run:

- `5000` steps: score `1.682648074`
- `30000` steps: score `3.435288527`
- `90000` steps: score `3.598051540`
- `125000` steps: score `3.808538992`
- `145000` steps: score `3.872327106` (best)
- `200000` steps: score `3.758833917`

The policy improved quickly, then oscillated in a fairly narrow band after roughly `80k` steps. Checkpointing by the project score was useful because the final weights were not the best weights.

### Tests

Executed:

```bash
XDG_CACHE_HOME=.cache MPLCONFIGDIR=.cache/matplotlib ./.venv/bin/python -m pytest snake/tests/base snake/tests/solver snake/tests/util
```

Result:

- `18 passed, 1 skipped`

### Saved Artifacts

- best model copied to `best/iteration1.zip`
- training summary written to `artifacts/iteration1/summary.json`
