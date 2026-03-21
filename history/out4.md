## Iteration 4 results

### Summary

- Restored `SnakeEnv` to the original game's fixed 4-body start by default, while keeping an explicit `--random-start` option for future runs.
- Kept the greedy-hinted observation design from iteration 3, but replaced approximate behavior-cloning warm starts with an exact PPO initialization that copies the 3-way greedy hint channel.
- On this setup, PPO fine-tuning did not beat the warm start; the best checkpoint was again the timestep `0` model.

### Main training run

- Command: `MPLCONFIGDIR=.cache/matplotlib XDG_CACHE_HOME=.cache PYTHONPATH=. .venv/bin/python train_test.py --run-name iteration4 --timesteps 5000 --eval-freq 5000 --eval-episodes 200 --test-episodes 1000 --n-envs 8 --seed 13 --exact-hint-init`
- Best checkpoint saved at timestep `0`
- Best score during training-time evaluation: `5.738245964050293`
- Best eval average snake length / steps: `60.709999084472656` / `642.3049926757812`
- Best eval average reward: `81.93351745605469`
- Best model copied to `best/iteration4.zip`

### 1000-game benchmark

- Average snake length: `59.894`
- Average steps: `633.483`
- Score `(avg length)^2 / avg steps`: `5.662805846407876`
- Average per-episode score `length^2 / steps`: `5.7944725039362694`
- Max snake length observed: `64`
- Min / max steps observed: `73` / `768`

### Comparison vs prior iterations

- Iteration 3 score: `5.580994307775686`
- Iteration 4 score: `5.662805846407876`
- Absolute improvement vs iteration 3: about `+0.0818`
- Multiplicative improvement vs iteration 3: about `1.0147x`

### Notes

- A probe run with the fixed-start environment showed that the main remaining loss versus the greedy teacher came from rare imitation mistakes over long episodes. Because the observation already exposes the greedy action as a one-hot hint, an exact policy initialization inside the PPO network removed that mismatch cleanly.
- The best checkpoint staying at timestep `0` means the improvement came from the environment reset change plus the exact greedy-hint warm start, not from additional PPO updates.
- This iteration is not directly apples-to-apples with iterations 1-3 on reset distribution: the benchmark still uses the same score formula, but the environment now starts from the original game's fixed 4-body configuration instead of a random 2-body snake.

### Verification

- `MPLCONFIGDIR=.cache/matplotlib XDG_CACHE_HOME=.cache PYTHONPATH=. .venv/bin/pytest --ignore=snake/tests/gui`
- Result: `18 passed, 1 skipped`
