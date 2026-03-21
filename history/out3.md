## Iteration 3 results

Iteration 3 kept the iteration-2 trap-aware `46`-feature observation, but changed the DQN setup to prioritize efficiency more directly:

- reward and starvation settings became configurable from `train_test.py`
- reward shaping was rebalanced toward faster food collection:
  - stronger move-away penalty
  - larger step pressure
  - smaller reachable-space bonus
  - revisit penalty
  - no-progress penalty after repeated non-improving moves
  - tighter starvation cutoff
- best-checkpoint selection changed from length-first to score-first:
  - primary: higher `avg_length / avg_steps`
  - secondary: higher average length
  - tertiary: lower average steps

## Probe summary

Short probes were run for `10000` timesteps with `200`-episode benchmarks:

- candidate A:
  - config: step `-0.03`, closer `+0.25`, farther `-0.35`, space scale `0.10`, revisit `-0.25`, starvation `30 + 6 * len`, no-progress `6 / -0.15`, exploration `0.20`
  - benchmark: avg length `23.440`, avg steps `151.015`, score `0.155216`

- candidate B:
  - config: step `-0.025`, closer `+0.25`, farther `-0.35`, space scale `0.15`, revisit `-0.25`, starvation `32 + 6 * len`, no-progress `8 / -0.12`, exploration `0.20`
  - benchmark: avg length `22.240`, avg steps `145.150`, score `0.153221`

- candidate C:
  - config: step `-0.03`, closer `+0.20`, farther `-0.40`, space scale `0.10`, revisit `-0.30`, starvation `28 + 6 * len`, no-progress `6 / -0.18`, exploration `0.18`
  - benchmark: avg length `22.555`, avg steps `144.295`, score `0.156312`

- candidate D:
  - config: step `-0.03`, closer `+0.25`, farther `-0.40`, space scale `0.10`, revisit `-0.25`, starvation `30 + 6 * len`, no-progress `6 / -0.15`, exploration `0.18`
  - benchmark: avg length `22.635`, avg steps `142.315`, score `0.159049`

Candidate D had the best score among the probes while keeping length above candidates B and C, so it was promoted to the main run.

## Main run configuration

- algorithm: `stable_baselines3.DQN`
- policy: `MlpPolicy`
- observation: `46`-dim egocentric feature vector
- actions: relative `left / straight / right`
- timesteps: `20000`
- eval frequency: `5000`
- eval episodes per checkpoint: `20`
- checkpoint-selection episodes: `200`
- final benchmark episodes: `1000`
- learning rate: `1e-4`
- exploration fraction: `0.18`
- network: `[256, 256, 256, 128]`
- seed: `7`

Environment coefficients used in the main run:

- step penalty: `-0.03`
- closer reward: `+0.25`
- farther penalty: `-0.40`
- reachable-space delta scale: `0.10`
- revisit penalty: `-0.25`
- starvation limit: `30 + 6 * snake_len`
- starvation penalty: `-3.0`
- no-progress threshold: `6`
- no-progress penalty: `-0.15`

Periodic checkpoint evaluations during training:

- `5000` timesteps: avg length `2.000`, avg steps `42.000`, score `0.047619`
- `10000` timesteps: avg length `23.150`, avg steps `153.850`, score `0.150471`
- `15000` timesteps: avg length `21.350`, avg steps `132.750`, score `0.160829`
- `20000` timesteps: avg length `20.400`, avg steps `123.300`, score `0.165450`

## Best model selection

Selection used the new score-first rule over `200` deterministic episodes.

- selected checkpoint: `runs/iteration3/checkpoints/step_20000.zip`
- selected checkpoint metrics:
  - avg length: `21.980`
  - avg steps: `132.520`
  - score: `0.165862`
  - avg return: `258.225`
- training time: `25.28s`
- copied to `best/iteration3.zip`

The score-first selector did what iteration 2 failed to do: it preferred the later, more efficient checkpoint instead of the earlier longer-but-slower one.

## Final benchmark

Benchmark over `1000` games for the selected model:

- avg snake length: `21.521`
- avg steps: `129.438`
- avg return: `251.070`
- median snake length: `21.0`
- median steps: `127.0`
- max snake length: `41`
- min steps: `12`
- score: `0.166265`

## Comparison

Versus iteration 2:

- avg snake length changed from `23.575` to `21.521` (`-2.054`, about `8.71%` lower)
- avg steps changed from `154.932` to `129.438` (`-25.494`, about `16.45%` lower)
- score changed from `0.152164` to `0.166265` (`+0.014101`, about `9.27%` higher)

Versus iteration 1:

- avg snake length changed from `20.233` to `21.521` (`+1.288`, about `6.37%` higher)
- avg steps changed from `120.153` to `129.438` (`+9.285`, about `7.73%` higher)
- score changed from `0.168394` to `0.166265` (`-0.002129`, about `1.26%` lower)

## Takeaways

- the efficiency-focused shaping successfully reversed most of iteration 2's step inflation
- the stronger efficiency pressure also pulled length down from the iteration-2 peak
- score-first checkpoint selection was an improvement and should be kept; in this run, later checkpoints were clearly better aligned with the project metric
- the remaining problem is closing the gap to iteration 1's best score while preserving more of iteration 2's longer-snake behavior

## Validation notes

- smoke training succeeded with the updated pipeline
- non-GUI tests passed with `./.venv/bin/python -m pytest -q snake/tests/base snake/tests/solver snake/tests/util`: `18 passed, 1 skipped`
- full `pytest` collection could not run in this environment because the GUI test imports `tkinter`, and the local Python build is missing `_tkinter`
