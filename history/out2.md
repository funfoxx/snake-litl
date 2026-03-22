## Iteration 2 Results

### Summary

Iteration 2 added a demonstration-learning stage before RL fine-tuning. The policy was pretrained on `GreedySolver` rollouts in the existing egocentric relative-action environment, then the best checkpoint was selected from evaluation history before the short RL phase could degrade it.

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

Teacher dataset summary:

- transitions: 117112
- avg teacher snake length: 59.5625
- avg teacher steps: 731.95
- max teacher length: 64
- teacher terminal events: 56 full, 88 deaths, 16 stalls

### Best checkpoint during training

Best checkpoint was found at **0 timesteps** immediately after imitation pretraining (`label: post_imitation`).

Eval metrics at the best checkpoint:

- avg snake length: **28.45**
- avg steps: **200.75**
- avg reward: **42.3325**
- max snake length during eval: **41**
- median snake length: **28.0**
- selection score (`avg_len * 1000 - avg_steps`): **28249.25**
- efficiency score (`avg_len^2 / avg_steps`): **4.031892901618929**

Imitation fit at the selected checkpoint:

- final supervised loss: **0.022690612047007837**
- final supervised accuracy: **0.9910156250520564**

### Final benchmark on saved best model

Benchmark results over **1000 games**:

- avg snake length: **28.152**
- avg steps: **209.006**
- avg reward: **41.77824**
- max snake length: **62**
- median snake length: **28.0**
- efficiency score (`avg_len^2 / avg_steps`): **3.7919251313359426**
- selection score (`avg_len * 1000 - avg_steps`): **27942.994**
- terminal events: **1000 deaths, 0 stalls**

### Comparison to iteration 1

Iteration 1 benchmark:

- avg snake length: 18.574
- avg steps: 118.26
- efficiency score: 2.917245695924235
- selection score: 18455.74

Iteration 2 change relative to iteration 1:

- avg snake length: **+9.578** (**1.516x** higher)
- avg steps: **+90.746** (**1.767x** higher)
- efficiency score: **+0.8746794354117076** (**1.300x** higher)
- selection score: **+9487.254**

### Relevant notes

- A smaller imitation budget already improved over iteration 1, but short RL fine-tuning runs consistently reduced deterministic performance. Because of that, the main run intentionally kept RL updates effectively negligible and selected the post-imitation checkpoint as the best model.
- The main model was copied to `best/iteration2.zip`.
- `best/iteration2.zip` matches `artifacts/iteration2_main/best_model.zip` (`shasum 70734c32864a9130fe54a3855966ba6c5dc662ee`).
- `PYTHONPATH=. .venv/bin/pytest --ignore=snake/tests/gui` passed with **18 passed, 1 skipped**.
