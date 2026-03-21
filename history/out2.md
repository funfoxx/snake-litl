## Iteration 2 results

### Summary

- Replaced reward-based checkpoint selection with score-based checkpoint selection using the true benchmark metric `(avg length)^2 / avg steps`.
- Kept the compact egocentric observation and PPO training setup from iteration 1.
- Ran a small seed search and kept the strongest checkpoint from the completed runs.

### Main training run

- Command: `PYTHONPATH=. MPLCONFIGDIR=.cache/matplotlib XDG_CACHE_HOME=.cache .venv/bin/python train_test.py --timesteps 150000 --eval-freq 5000 --eval-episodes 50 --test-episodes 1000 --seed 13`
- Best checkpoint saved at timestep `125000`
- Best score during training-time evaluation: `3.464693069458008`
- Best eval average snake length / steps: `21.2` / `129.72`
- Best model copied to `best/iteration2.zip`

### 1000-game benchmark

- Average snake length: `19.828`
- Average steps: `121.216`
- Score `(avg length)^2 / avg steps`: `3.2433802798310456`
- Average per-episode score `length^2 / steps`: `3.3133123163314004`
- Max snake length observed: `42`
- Min / max steps observed: `17` / `338`

### Comparison vs iteration 1

- Iteration 1 score: `3.3172447799390135`
- Iteration 2 score: `3.2433802798310456`
- Absolute change: about `-0.0739`

### Notes

- Score-based checkpointing did find checkpoints that looked stronger than the final iteration 1 benchmark on small in-training eval sets, but those gains did not hold up on the full 1000-game benchmark.
- Across the completed runs, the best realized 1000-game score came from the seed `13` run above, so that checkpoint is the iteration 2 artifact.

### Verification

- `PYTHONPATH=. MPLCONFIGDIR=.cache/matplotlib XDG_CACHE_HOME=.cache .venv/bin/pytest --ignore=snake/tests/gui`
- Result: `18 passed, 1 skipped`
