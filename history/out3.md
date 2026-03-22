## Iteration 3 Results

### Summary

Iteration 3 replaced one-shot imitation with a DAgger-style aggregation pipeline. The policy was first pretrained on `GreedySolver` demonstrations, then improved through two relabeling rounds where the current policy controlled the snake on mixed teacher/student rollouts while `GreedySolver` labeled every visited state.

Main run configuration:

- train steps: 1
- eval frequency: every 2000 steps
- eval episodes: 40
- benchmark episodes: 1000
- seed: 7
- imitation episodes: 160
- imitation epochs: 10
- imitation batch size: 256
- imitation learning rate: 0.001
- DAgger rounds: 2
- DAgger episodes per round: 30
- DAgger beta schedule: 0.7 -> 0.2
- DAgger epochs per round: 2
- DAgger learning rate: 0.0005

Initial teacher dataset summary:

- transitions: 117112
- avg teacher snake length: 59.5625
- avg teacher steps: 731.95
- max teacher length: 64
- teacher terminal events: 56 full, 88 deaths, 16 stalls

Aggregation round summaries:

- round 1: 13747 relabeled transitions, avg rollout length 46.56666666666667, avg rollout steps 458.23333333333335, teacher control rate 0.7025532843529497
- round 2: 8670 relabeled transitions, avg rollout length 35.233333333333334, avg rollout steps 289.0, teacher control rate 0.20265282583621683
- combined dataset after round 2: 139529 transitions

### Best checkpoint during training

Best checkpoint was found at **0 timesteps** after aggregation round 2 (`label: dagger_round_2`).

Eval metrics at the best checkpoint:

- avg snake length: **31.975**
- avg steps: **262.5**
- avg reward: **47.69874999999985**
- max snake length during eval: **50**
- median snake length: **32.0**
- selection score (`avg_len * 1000 - avg_steps`): **31712.5**
- efficiency score (`avg_len^2 / avg_steps`): **3.894859523809524**

Supervised fit at the selected checkpoint:

- final DAgger round loss: **0.019433571626995126**
- final DAgger round accuracy: **0.9945094691309737**

### Final benchmark on saved best model

Benchmark results over **1000 games**:

- avg snake length: **30.817**
- avg steps: **241.904**
- avg reward: **45.96875999999986**
- max snake length: **64**
- median snake length: **30.0**
- efficiency score (`avg_len^2 / avg_steps`): **3.925885843144388**
- selection score (`avg_len * 1000 - avg_steps`): **30575.096**
- terminal events: **995 deaths, 5 full boards**

### Comparison to iteration 2

Iteration 2 benchmark:

- avg snake length: 28.152
- avg steps: 209.006
- efficiency score: 3.7919251313359426
- selection score: 27942.994

Iteration 3 change relative to iteration 2:

- avg snake length: **+2.665** (**1.095x** higher)
- avg steps: **+32.898** (**1.157x** higher)
- efficiency score: **+0.1339607118084454** (**1.035x** higher)
- selection score: **+2632.102** (**1.094x** higher)

### Relevant notes

- The best checkpoint came from the second aggregation round, not from the initial expert-only model. That supports the iteration hypothesis that covariate shift, not network capacity, was the main bottleneck.
- A three-round probe was tested before the main run and benchmarked slightly worse than the two-round setup, so the official run used two aggregation rounds.
- The saved main model was copied to `best/iteration3.zip`.
- `best/iteration3.zip` matches `artifacts/iteration3_main/best_model.zip` (`shasum fcb63625799880055a8c34e9a7da9314fcf5a72d`).
- `PYTHONPATH=. .venv/bin/pytest --ignore=snake/tests/gui` passed with **18 passed, 1 skipped**.
