## Iteration 4 results

Iteration 4 kept the iteration-3 trap-aware `46`-feature observation, but changed the reward shaping so reachable space behaved more like a safety constraint than a standing objective:

- positive reward for increasing reachable space was removed
- only space loss was penalized
- a new penalty was added when the resulting state made food unreachable
- a small safe-progress bonus was added for moves that reduced food distance while preserving reachable space close to the best safe alternative

The training script was also extended so those new shaping coefficients could be tuned from the command line.

## Probe summary

Smoke validation:

- `4000`-step smoke run completed successfully

Short probes were run for `10000` timesteps with `200`-episode benchmarks:

- candidate A:
  - config: step `-0.028`, closer `+0.25`, farther `-0.40`, space-loss `0.18`, food-unreachable `-0.25`, safe-progress `+0.18`, margin `0.08`, revisit `-0.22`, starvation `30 + 6 * len`, no-progress `6 / -0.12`, exploration `0.22`
  - benchmark: avg length `23.465`, avg steps `147.500`, score `0.159085`

- candidate B:
  - config: step `-0.030`, closer `+0.28`, farther `-0.42`, space-loss `0.16`, food-unreachable `-0.30`, safe-progress `+0.22`, margin `0.08`, revisit `-0.20`, starvation `28 + 6 * len`, no-progress `5 / -0.14`, exploration `0.20`
  - benchmark: avg length `22.670`, avg steps `142.820`, score `0.158731`

- candidate C:
  - config: step `-0.027`, closer `+0.25`, farther `-0.38`, space-loss `0.20`, food-unreachable `-0.20`, safe-progress `+0.15`, margin `0.10`, revisit `-0.22`, starvation `32 + 6 * len`, no-progress `6 / -0.12`, exploration `0.22`
  - benchmark: avg length `22.500`, avg steps `138.655`, score `0.162273`

Candidate C had the best probe score, so it was promoted to the main run.

## Main run configuration

- algorithm: `stable_baselines3.DQN`
- policy: `MlpPolicy`
- observation: `46`-dim egocentric feature vector
- actions: relative `left / straight / right`
- timesteps: `25000`
- eval frequency: `5000`
- eval episodes per checkpoint: `20`
- checkpoint-selection episodes: `200`
- final benchmark episodes: `1000`
- learning rate: `1e-4`
- exploration fraction: `0.22`
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

- `5000` timesteps: avg length `2.000`, avg steps `44.000`, score `0.045455`
- `10000` timesteps: avg length `23.650`, avg steps `150.300`, score `0.157352`
- `15000` timesteps: avg length `20.300`, avg steps `123.550`, score `0.164306`
- `20000` timesteps: avg length `21.950`, avg steps `131.900`, score `0.166414`
- `25000` timesteps: avg length `22.600`, avg steps `141.100`, score `0.160170`

## Best model selection

Selection used the existing score-first rule over `200` deterministic episodes.

- selected checkpoint: `runs/iteration4/checkpoints/step_20000.zip`
- selected checkpoint metrics:
  - avg length: `23.460`
  - avg steps: `143.275`
  - score: `0.163741`
  - avg return: `294.122`
- training time: `39.78s`
- copied to `best/iteration4.zip`

## Final benchmark

Benchmark over `1000` games for the selected model:

- avg snake length: `22.390`
- avg steps: `135.930`
- avg return: `277.680`
- median snake length: `22.0`
- median steps: `133.0`
- max snake length: `45`
- min steps: `23`
- score: `0.164717`

## Comparison

Versus iteration 3:

- avg snake length changed from `21.521` to `22.390` (`+0.869`, about `4.04%` higher)
- avg steps changed from `129.438` to `135.930` (`+6.492`, about `5.02%` higher)
- score changed from `0.166265` to `0.164717` (`-0.001548`, about `0.93%` lower)

Versus iteration 2:

- avg snake length changed from `23.575` to `22.390` (`-1.185`, about `5.03%` lower)
- avg steps changed from `154.932` to `135.930` (`-19.002`, about `12.26%` lower)
- score changed from `0.152164` to `0.164717` (`+0.012554`, about `8.25%` higher)

Versus iteration 1:

- avg snake length changed from `20.233` to `22.390` (`+2.157`, about `10.66%` higher)
- avg steps changed from `120.153` to `135.930` (`+15.777`, about `13.13%` higher)
- score changed from `0.168394` to `0.164717` (`-0.003677`, about `2.18%` lower)

## Takeaways

- removing the positive reachable-space reward did stop the agent from chasing open space for its own sake
- the new asymmetric shaping preserved more snake length than iteration 3, but not enough efficiency to beat iteration 3 on score
- in the main run, the best checkpoint again appeared before the end of training, which suggests the later policy drifted back toward slower play
- the next iteration should probably keep the asymmetric trap shaping idea, but tune exploration or checkpoint horizon rather than simply training longer with the same coefficients

## Validation notes

- smoke training succeeded with the updated pipeline
- non-GUI tests passed with `./.venv/bin/python -m pytest -q snake/tests/base snake/tests/solver snake/tests/util`: `18 passed, 1 skipped`
- full `pytest` collection was not run because the GUI test imports `tkinter`, and the local Python build in prior iterations was missing `_tkinter`
