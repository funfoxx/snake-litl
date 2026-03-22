## Iteration 1 Results

### Summary

Iteration 1 replaced the sparse absolute-action wrapper with a relative-action egocentric environment, added shaped rewards and a hunger-based truncation rule, and trained SB3 DQN with a custom multi-input feature extractor.

Main run configuration:

- train steps: 20000
- eval frequency: every 2000 steps
- eval episodes: 40
- benchmark episodes: 1000
- seed: 7

### Best checkpoint during training

Best checkpoint was found at **18000 timesteps**.

Eval metrics at the best checkpoint:

- avg snake length: **19.175**
- avg steps: **126.05**
- avg reward: **27.38825**
- max snake length during eval: **31**
- median snake length: **18.0**
- selection score (`avg_len * 1000 - avg_steps`): **19048.95**
- efficiency score (`avg_len^2 / avg_steps`): **2.916942681475605**

### Final benchmark on saved best model

Benchmark results over **1000 games**:

- avg snake length: **18.574**
- avg steps: **118.26**
- avg reward: **26.4247**
- max snake length: **36**
- median snake length: **18.0**
- efficiency score (`avg_len^2 / avg_steps`): **2.917245695924235**
- terminal events: **1000 deaths, 0 stalls**

### Comparison to iteration 0

Iteration 0 benchmark:

- avg snake length: 2.083
- avg steps: 9821.126
- efficiency score: 0.000441791399

Iteration 1 change relative to iteration 0:

- avg snake length: **+16.491** (**8.917x** higher)
- avg steps: **-9702.866** (**0.012041x** of the baseline, about 98.8% lower)
- efficiency score: **6603.22x** higher

### Relevant notes

- The training curve improved quickly once learning began, and the best checkpoint appeared late in training rather than at the end.
- The saved best checkpoint was copied to `best/iteration1.zip`.
- `pytest` initially failed during collection because `snake` was not on `PYTHONPATH`.
- `PYTHONPATH=. .venv/bin/pytest --ignore=snake/tests/gui` passed with **18 passed, 1 skipped**.
- The GUI test module could not run in this environment because the Python install lacks `tkinter` (`_tkinter` import error).
