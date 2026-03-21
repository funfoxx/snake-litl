## Iteration 5 results

Iteration 5 kept the iteration-4 environment and reward shaping fixed, and focused on stabilizing the DQN training schedule so the best efficiency checkpoint could be captured more reliably.

What changed in code and training:

- `train_test.py` now supports:
  - linearly decaying learning rate
  - configurable terminal exploration epsilon
  - configurable target-network update interval
- reward coefficients were kept at the iteration-4 main values
- checkpoint evaluation frequency was tightened from `5000` to `2500` timesteps
- probes were used to decide whether the main run should stay short because iteration 4 had already shown early peaking

## Probe summary

All probes used the iteration-4 main environment coefficients:

- step penalty: `-0.027`
- closer reward: `+0.25`
- farther penalty: `-0.38`
- space-loss scale: `0.20`
- food-unreachable penalty: `-0.20`
- safe-progress bonus: `+0.15`
- safe-progress margin: `0.10`
- revisit penalty: `-0.22`
- starvation limit: `32 + 6 * snake_len`
- starvation penalty: `-3.0`
- no-progress threshold: `6`
- no-progress penalty: `-0.12`

Probe setup:

- timesteps: `12000`
- eval frequency: `2500`
- eval episodes: `20`
- selection episodes: `200`
- benchmark episodes: `200`

Probe results:

- candidate A:
  - schedule: learning rate `1e-4 -> 3e-5`, exploration fraction `0.20`, final epsilon `0.01`, target update interval `2000`
  - selected checkpoint: `7500`
  - benchmark: avg length `23.605`, avg steps `144.960`, score `0.162838`

- candidate B:
  - schedule: learning rate `1e-4 -> 2e-5`, exploration fraction `0.18`, final epsilon `0.005`, target update interval `1500`
  - selected checkpoint: `7500`
  - benchmark: avg length `24.055`, avg steps `147.110`, score `0.163517`

- candidate C:
  - schedule: learning rate `8e-5 -> 2e-5`, exploration fraction `0.20`, final epsilon `0.005`, target update interval `1500`
  - selected checkpoint: `7500`
  - benchmark: avg length `20.805`, avg steps `121.195`, score `0.171666`

Candidate C clearly separated itself on score by cutting steps much harder than A and B, so it was promoted to the main run.

## Main run configuration

- algorithm: `stable_baselines3.DQN`
- policy: `MlpPolicy`
- observation: `46`-dim egocentric feature vector
- actions: relative `left / straight / right`
- timesteps: `10000`
- eval frequency: `2500`
- eval episodes per checkpoint: `20`
- checkpoint-selection episodes: `400`
- final benchmark episodes: `1000`
- learning rate: `8e-5 -> 2e-5`
- exploration fraction: `0.20`
- exploration final epsilon: `0.005`
- target update interval: `1500`
- network: `[256, 256, 256, 128]`
- seed: `7`

Environment coefficients used in the main run:

- step penalty: `-0.027`
- closer reward: `+0.25`
- farther penalty: `-0.38`
- space-loss scale: `0.20`
- food-unreachable penalty: `-0.20`
- safe-progress bonus: `+0.15`
- safe-progress margin: `0.10`
- revisit penalty: `-0.22`
- starvation limit: `32 + 6 * snake_len`
- starvation penalty: `-3.0`
- no-progress threshold: `6`
- no-progress penalty: `-0.12`

Periodic checkpoint evaluations during training:

- `2500` timesteps: avg length `2.000`, avg steps `44.000`, score `0.045455`
- `5000` timesteps: avg length `2.000`, avg steps `44.000`, score `0.045455`
- `7500` timesteps: avg length `19.950`, avg steps `116.450`, score `0.171318`
- `10000` timesteps: avg length `22.750`, avg steps `138.850`, score `0.163846`

## Best model selection

Selection used the existing score-first rule over `400` deterministic episodes.

- selected checkpoint: `runs/iteration5/checkpoints/step_7500.zip`
- selected checkpoint metrics:
  - avg length: `20.083`
  - avg steps: `116.893`
  - score: `0.171803`
  - avg return: `241.677`
- training time: `14.38s`
- copied to `best/iteration5.zip`

The denser checkpointing mattered here: the `7500` checkpoint had a clearly better score than the `10000` checkpoint, and the final benchmark confirmed that the earlier checkpoint was the right model to keep.

## Final benchmark

Benchmark over `1000` games for the selected model:

- avg snake length: `20.025`
- avg steps: `116.350`
- avg return: `241.097`
- median snake length: `20.0`
- median steps: `115.0`
- max snake length: `35`
- min steps: `12`
- score: `0.172110`

## Comparison

Versus iteration 4:

- avg snake length changed from `22.390` to `20.025` (`-2.365`, about `10.56%` lower)
- avg steps changed from `135.930` to `116.350` (`-19.580`, about `14.40%` lower)
- score changed from `0.164717` to `0.172110` (`+0.007393`, about `4.49%` higher)

Versus iteration 3:

- avg snake length changed from `21.521` to `20.025` (`-1.496`, about `6.95%` lower)
- avg steps changed from `129.438` to `116.350` (`-13.088`, about `10.11%` lower)
- score changed from `0.166265` to `0.172110` (`+0.005845`, about `3.52%` higher)

Versus iteration 1:

- avg snake length changed from `20.233` to `20.025` (`-0.208`, about `1.03%` lower)
- avg steps changed from `120.153` to `116.350` (`-3.803`, about `3.17%` lower)
- score changed from `0.168394` to `0.172110` (`+0.003716`, about `2.21%` higher)

## Takeaways

- the iteration-4 asymmetric trap shaping was good enough to keep; the larger issue was late-training drift
- the lower starting learning rate plus decay and lower terminal epsilon produced a much sharper efficiency checkpoint around `7500` steps
- denser checkpoint spacing was necessary, because the best model would have been missed by the old `5000`-step cadence
- this iteration set a new best score by prioritizing step efficiency, even though it gave back some of the longer-snake behavior from iterations 2 and 4

## Validation notes

- smoke training succeeded with the new schedule-aware CLI
- non-GUI tests passed with `./.venv/bin/python -m pytest -q snake/tests/base snake/tests/solver snake/tests/util`: `18 passed, 1 skipped`
- full `pytest` collection was not run because the GUI test imports `tkinter`, and prior iterations already noted that the local Python build is missing `_tkinter`
