## Iteration 1 results

### Summary

- Switched from sparse-reward DQN on a raw grid to PPO on a compact egocentric state.
- Replaced absolute actions with relative `{left, straight, right}` actions.
- Added food-distance shaping, a per-step penalty, and starvation / max-step truncation.

### Main training run

- Command: `train_test.py --timesteps 200000 --eval-freq 5000 --eval-episodes 20 --test-episodes 1000`
- Best checkpoint saved at timestep `150000`
- Best eval mean reward during training: `22.97811615`
- Best model copied to `best/iteration1.zip`

### 1000-game benchmark

- Average snake length: `19.981`
- Average steps: `120.353`
- Score `(avg length)^2 / avg steps`: `3.3172447799390135`
- Average per-episode score `length^2 / steps`: `3.397642117442929`
- Max snake length observed: `37`
- Min / max steps observed: `12` / `294`

### Comparison vs iteration 0

- Iteration 0 score: `0.000441791399`
- Iteration 1 score: `3.3172447799390135`
- Absolute improvement: about `+3.3168`
- Multiplicative improvement: about `7508x`

### Verification

- `PYTHONPATH=. XDG_CACHE_HOME=.cache MPLCONFIGDIR=.cache/matplotlib .venv/bin/pytest --ignore=snake/tests/gui`
- Result: `18 passed, 1 skipped`
- Full `pytest` collection still fails on `snake/tests/gui/test_gui.py` because this Python environment does not have `_tkinter` available.

### Notes

- The policy learned quickly and peaked around `150k` steps; later checkpoints were competitive but not better than the best saved model.
- The new setup dramatically reduced episode length while learning to collect many more food items, which is why the benchmark score increased so sharply.
