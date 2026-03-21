## Iteration 1 results

Main run used the new relative-action DQN environment and the rewritten training/evaluation pipeline.

Final run configuration:

- algorithm: `stable_baselines3.DQN`
- policy: `MlpPolicy`
- observation: 30-dim egocentric feature vector
- actions: relative `left / straight / right`
- timesteps: `20000`
- eval frequency: `5000`
- eval episodes: `20`
- final benchmark episodes: `1000`
- seed: `7`

Why `20000` timesteps for the main run:

- exploratory runs showed a strong jump by `20000` timesteps
- a larger `50000` timestep probe only increased benchmark length slightly while increasing average steps, so `20000` had the better length/efficiency tradeoff

Evaluation checkpoints during training:

- `5000` timesteps: mean eval reward `-14.046`, mean eval episode length `53.8`
- `10000` timesteps: mean eval reward `256.057`, mean eval episode length `138.9`
- `15000` timesteps: mean eval reward `237.868`, mean eval episode length `120.85`
- `20000` timesteps: mean eval reward `275.074`, mean eval episode length `138.3`

Best model:

- best checkpoint was found at `20000` timesteps
- training time: `11.50s`
- copied to `best/iteration1.zip`

Benchmark over `1000` games:

- avg snake length: `20.233`
- avg steps: `120.153`
- avg return: `238.583`
- median snake length: `20`
- median steps: `116`
- max snake length: `37`
- min steps: `20`
- score: `0.168394`

Comparison vs iteration 0:

- avg snake length improved from `2.083` to `20.233` (`+18.150`, about `9.71x`)
- avg steps dropped from `9821.126` to `120.153` (`-9700.973`, about `98.78%` lower)
- score improved from `0.000441791399` to `0.168394` (about `381x`)

What changed that mattered:

- switching to relative actions removed unnecessary symmetry-breaking from the control problem
- egocentric state features made food direction and nearby hazards much easier for the MLP DQN to learn
- reward shaping plus the starvation limit heavily reduced useless wandering, which is why steps fell so sharply
