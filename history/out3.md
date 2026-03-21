## Iteration 3 results

### Summary

- The compact-state imitation approach from the initial iteration 3 plan was not strong enough: richer ray and board encodings plus DAgger improved rollout quality, but still stayed below iteration 2's benchmark.
- The successful change was to add the greedy solver's current recommended relative action as an auxiliary observation feature, then behavior-clone a PPO policy on top of that teacher-hinted state.
- With that hint available, plain BC became nearly exact and PPO fine-tuning was unnecessary; the best checkpoint was the post-BC model at timestep `0`.

### Main training run

- Command: `PYTHONPATH=. MPLCONFIGDIR=.cache/matplotlib XDG_CACHE_HOME=.cache .venv/bin/python train_test.py --run-name iteration3 --timesteps 5000 --eval-freq 5000 --eval-episodes 200 --test-episodes 1000 --seed 13 --bc-episodes 200 --bc-epochs 5 --bc-batch-size 2048 --dagger-rounds 0 --dagger-episodes 30 --learning-rate 1e-4 --ent-coef 5e-4`
- Best checkpoint saved at timestep `0`
- Best score during training-time evaluation: `5.703348159790039`
- Best eval average snake length / steps: `59.55500030517578` / `621.8800048828125`
- Best eval average reward: `82.23123931884766`
- Best model copied to `best/iteration3.zip`

### 1000-game benchmark

- Average snake length: `59.863`
- Average steps: `642.104`
- Score `(avg length)^2 / avg steps`: `5.580994307775686`
- Average per-episode score `length^2 / steps`: `5.705275026512151`
- Max snake length observed: `64`
- Min / max steps observed: `178` / `768`

### Teacher and warm-start stats

- Expert demonstration set: `200` greedy-solver episodes
- Expert average snake length / steps: `59.915` / `646.755`
- Expert score: `5.550490100579045`
- BC validation accuracy: `0.9998453855514526`
- BC validation loss: `0.005959393922239542`
- Final model slightly beat the demonstration benchmark on the 1000-game evaluation: about `+0.0305` score

### Comparison vs prior iterations

- Iteration 2 score: `3.2433802798310456`
- Iteration 3 score: `5.580994307775686`
- Absolute improvement vs iteration 2: about `+2.3376`
- Multiplicative improvement vs iteration 2: about `1.72x`
- Iteration 1 score: `3.3172447799390135`
- Absolute improvement vs iteration 1: about `+2.2637`

### Notes

- The original iteration 3 idea was to use a richer observation plus DAgger to imitate the greedy solver. That path improved warm-start performance but did not reach the target benchmark.
- Adding the greedy solver recommendation directly into the observation removed the remaining ambiguity. Once that hint was present, BC alone reproduced near-greedy play and PPO updates no longer improved the checkpoint.
- Because the final evaluation metric uses actual game outcomes rather than shaped reward, the iteration 3 result is directly comparable with the earlier iterations.

### Verification

- `PYTHONPATH=. MPLCONFIGDIR=.cache/matplotlib XDG_CACHE_HOME=.cache .venv/bin/pytest --ignore=snake/tests/gui`
- Result: `18 passed, 1 skipped`
