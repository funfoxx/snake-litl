## Iteration 5 Results

### Summary

This iteration started by testing a larger hybrid observation and CNN-based DQN, reintroducing a relative global board encoding. That path was not competitive:

- it trained much more slowly
- its early smoke checkpoints underperformed the existing iteration-4 baseline
- it also exposed a checkpoint-overwrite issue, which was fixed by switching to temp-file saves plus atomic replace

The winning change for the final model was more conservative and more reliable:

- keep the iteration-4 custom `DQN` trainer and environment
- keep dueling heads, double-DQN targets, prioritized replay, and `n_step = 1`
- improve checkpoint selection by saving the top `3` checkpoints from the main run by primary eval score
- run a larger secondary validation on a disjoint seed block after training
- export only the checkpoint that wins the secondary validation

That produced a small but real improvement on the full `1000`-game benchmark over iteration 4.

### Smoke Validation

The first hybrid-observation smoke was abandoned after it lagged the established trainer and made the run far slower. The useful smoke result for the final approach was:

- artifact: `artifacts/iteration5/smoke_select/summary.json`
- training timesteps: `60000`
- primary eval episodes: `50`
- secondary eval episodes: `200`
- benchmark episodes: `200`

Top smoke checkpoints by primary eval were:

- `55000` steps: primary eval score `4.546543795`
- `45000` steps: primary eval score `4.155577937`
- `40000` steps: primary eval score `4.095746119`

Secondary validation then re-ranked those candidates on seeds starting at `20000`:

- `55000` steps: secondary score `4.283515917`
- `45000` steps: secondary score `4.126251946`
- `40000` steps: secondary score `3.999543596`

The smoke benchmark of the selected checkpoint over `200` games was:

- avg snake length: `29.385`
- avg steps: `201.375`
- score: `4.287911732`
- max snake length observed: `47`

That was slightly better than iteration 4's final validation smoke (`4.283515917`), so the selection strategy was promoted to the main run.

### Main Run Configuration

- algorithm: custom `DQN`
- seed: `7`
- training timesteps: `70000`
- eval frequency: every `5000` steps
- primary eval episodes per checkpoint: `100`
- secondary eval episodes for top checkpoints: `300`
- primary eval seed base: `10000`
- secondary eval seed base: `20000`
- final benchmark seed base: `0`
- final benchmark episodes: `1000`
- top checkpoints retained: `3`
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

### Best Model Selection

Best primary eval checkpoint during training:

- timestep: `60000`
- primary eval score: `4.279636271`
- primary eval avg snake length: `30.43`
- primary eval avg steps: `216.37`

Top `3` candidate checkpoints sent to secondary validation:

- `60000` steps: primary `4.279636271`
- `65000` steps: primary `4.265461538`
- `55000` steps: primary `4.146311179`

Secondary validation over `300` episodes selected:

- selected timestep: `60000`
- secondary score: `4.233630930`
- secondary avg snake length: `30.783333333`
- secondary avg steps: `223.830000000`

So in the main run, secondary selection did not change the winning checkpoint, but it did verify that the `60000`-step model was slightly more robust than the nearby `55000` and `65000` candidates.

### Final Benchmark

Benchmark of the exported selected model over `1000` games:

- avg snake length: `29.988`
- avg steps: `210.119`
- score: `4.279861145`
- max snake length observed: `57`

### Comparison To Iteration 4

- iteration 4 score: `4.266359455`
- iteration 5 score: `4.279861145`
- absolute gain: `0.013501690`
- multiplicative gain: about `1.003x`

This is a small improvement, but it is a genuine improvement on the repository benchmark with the required DQN-only constraint intact.

### Evaluation Curve Notes

Selected primary checkpoint evaluations from the main run:

- `30000` steps: score `4.053686102`
- `40000` steps: score `4.118734090`
- `50000` steps: score `4.144598888`
- `55000` steps: score `4.146311179`
- `60000` steps: score `4.279636271` (best primary eval and final selected checkpoint)
- `65000` steps: score `4.265461538`
- `70000` steps: score `3.783257404`

The same late-training degradation remained. Iteration 5 did not solve the underlying tendency to trade too many extra steps for survival, but it improved experiment reliability and extracted a slightly stronger checkpoint from the same training regime.

### Tests

Executed:

```bash
XDG_CACHE_HOME=.cache MPLCONFIGDIR=.cache/matplotlib TMPDIR=/Users/jude/Desktop/school/snake/snake-litl/.cache TORCHINDUCTOR_CACHE_DIR=/Users/jude/Desktop/school/snake/snake-litl/.cache ./.venv/bin/python -m pytest snake/tests/base snake/tests/solver snake/tests/util
```

Result:

- `18 passed, 1 skipped`

### Saved Artifacts

- best model copied to `best/iteration5.zip`
- main training summary written to `artifacts/iteration5/summary.json`
- final validation smoke summary written to `artifacts/iteration5/smoke_select/summary.json`
