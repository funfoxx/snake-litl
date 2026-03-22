## Iteration 5 results

### Summary

- Added a configurable teacher wrapper and replaced the hard-coded greedy hint with a teacher-selected 3-way hint channel.
- Ran iteration 5 with an `aggressive_food` teacher that follows pure shortest-path-to-food actions while the snake length is below `12`, then falls back to the built-in greedy solver.
- The idea looked promising on small direct probes, but it did not beat iteration 4 on the full 1000-episode benchmark. The best checkpoint was again the timestep `0` exact-hint model.

### Main training run

- Command: `MPLCONFIGDIR=.cache/matplotlib XDG_CACHE_HOME=.cache PYTHONPATH=. .venv/bin/python train_test.py --run-name iteration5 --timesteps 5000 --eval-freq 5000 --eval-episodes 200 --test-episodes 1000 --n-envs 8 --seed 13 --exact-hint-init --teacher-mode aggressive_food --teacher-aggressive-food-len 12`
- Best checkpoint saved at timestep `0`
- Best score during training-time evaluation: `5.631655216217041`
- Best eval average snake length / steps: `59.125` / `620.7349853515625`
- Best eval average reward: `79.4969482421875`
- Best model copied to `best/iteration5.zip`

### 1000-game benchmark

- Average snake length: `58.547`
- Average steps: `608.543`
- Score `(avg length)^2 / avg steps`: `5.632718162890707`
- Average per-episode score `length^2 / steps`: `5.811816071253574`
- Max snake length observed: `64`
- Min / max steps observed: `9` / `768`

### Comparison vs prior iterations

- Iteration 4 score: `5.662805846407876`
- Iteration 5 score: `5.632718162890707`
- Absolute change vs iteration 4: about `-0.0301`
- Multiplicative change vs iteration 4: about `0.9947x`

### Notes

- A direct check before the run showed that the exact-hint initialization still reproduces the hint channel exactly after save/load, so the teacher policy itself was the only meaningful change in this iteration.
- The `aggressive_food` opening reduced average steps, but it also reduced average snake length enough to lose overall benchmark score on the larger evaluation.
- The very low minimum step count (`9`) indicates that the more aggressive opening occasionally created early failures that the pure greedy teacher avoided.
- This iteration is a negative result, but the new configurable teacher path is useful infrastructure for future experiments because it allows teacher-policy changes without touching `snake/` or reworking the PPO architecture again.

### Verification

- `MPLCONFIGDIR=.cache/matplotlib XDG_CACHE_HOME=.cache PYTHONPATH=. .venv/bin/pytest --ignore=snake/tests/gui`
- Result: `18 passed, 1 skipped`
