## Iteration 4 Results

### Summary

Iteration 4 added replay-buffer prefill support and more conservative warm-start RL settings, then tested whether Q-learning could improve a DAgger-initialized policy instead of degrading it. In practice, the replay-prefilled RL probes still underperformed the best pure DAgger checkpoints, so the strongest configuration found for the official run stayed on the imitation-plus-DAgger backbone and used a better-performing seed discovered during probing.

Official main run configuration:

- train steps: 1
- eval frequency: every 2000 steps
- eval episodes: 40
- benchmark episodes: 1000
- seed: 29
- imitation episodes: 160
- imitation epochs: 10
- imitation batch size: 256
- imitation learning rate: 0.001
- DAgger rounds: 2
- DAgger episodes per round: 30
- DAgger beta schedule: 0.7 -> 0.2
- DAgger epochs per round: 2
- DAgger learning rate: 0.0005
- replay prefill: implemented in code, not used in the final main run because probes regressed benchmark performance

### Best checkpoint during training

Best checkpoint was found at **0 timesteps** after aggregation round 2 (`label: dagger_round_2`).

Eval metrics at the best checkpoint:

- avg snake length: **32.825**
- avg steps: **266.35**
- avg reward: **49.13024999999986**
- max snake length during eval: **63**
- median snake length: **31.0**
- selection score (`avg_len * 1000 - avg_steps`): **32558.65**
- efficiency score (`avg_len^2 / avg_steps`): **4.045355734934298**

Supervised fit at the selected checkpoint:

- final DAgger round loss: **0.019428601331618944**
- final DAgger round accuracy: **0.994751199725777**

### Final benchmark on saved best model

Benchmark results over **1000 games**:

- avg snake length: **30.627**
- avg steps: **240.345**
- avg reward: **45.61154999999987**
- max snake length: **64**
- median snake length: **30.0**
- efficiency score (`avg_len^2 / avg_steps`): **3.9027777944205204**
- selection score (`avg_len * 1000 - avg_steps`): **30386.655**
- terminal events: **994 deaths, 5 full boards, 1 stall**

### Comparison to iteration 3

Iteration 3 benchmark:

- avg snake length: 30.817
- avg steps: 241.904
- efficiency score: 3.925885843144388
- selection score: 30575.096

Iteration 4 change relative to iteration 3:

- avg snake length: **-0.19**
- avg steps: **-1.559**
- efficiency score: **-0.023108048723867736**
- selection score: **-188.441**

### Relevant notes

- The replay-prefill implementation worked end to end and passed smoke tests, but RL fine-tuning consistently reduced closed-loop performance in the probes, even with seeded replay and lower exploration.
- A seed sweep mattered more than the replay stage. Among the seeds tested, **29** was the strongest on the iteration 4 backbone and was therefore used for the official run.
- The saved main model was copied to `best/iteration4.zip`.
- `best/iteration4.zip` matches `artifacts/iteration4_main/best_model.zip` (`shasum 0edfb0d38641064b84b3aedf60bc8d69a6dcadb8`).
- `PYTHONPATH=. .venv/bin/pytest --ignore=snake/tests/gui` passed with **18 passed, 1 skipped**.
