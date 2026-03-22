## Iteration 5 Results

### Summary

Iteration 5 tested whether the remaining performance gap was caused by how the imitation dataset was weighted and by checkpoint-selection noise rather than by model capacity.

The code changes added:

- per-transition supervised sample weights, with separate weights for expert data, teacher-controlled DAgger data, and student-controlled DAgger data
- optional evaluation after every imitation / DAgger epoch
- deterministic evaluation and benchmark seed schedules for lower-variance probe comparisons
- a negative-seed override so official runs can still use the historical random-start protocol for direct comparison with earlier iterations

In practice, the new weighting and per-epoch-selection ideas did **not** improve the historical random-start benchmark. The strongest fixed-seed probe this iteration came from the unchanged two-round DAgger backbone with seed 19, but when rerun under the official 1000-game random benchmark protocol it still underperformed iteration 4. The official saved model for iteration 5 is therefore the best full run achieved this round, not an improvement over the previous best.

Official selected run configuration:

- train steps: 1
- eval frequency: every 2000 steps
- eval episodes: 40
- benchmark episodes: 1000
- seed: 19
- eval seed: random protocol restored with `--eval-seed -1`
- benchmark seed: random protocol restored with `--benchmark-seed -1`
- imitation episodes: 160
- imitation epochs: 10
- imitation batch size: 256
- imitation learning rate: 0.001
- DAgger rounds: 2
- DAgger episodes per round: 30
- DAgger beta schedule: 0.7 -> 0.2
- DAgger epochs per round: 2
- DAgger learning rate: 0.0005
- expert sample weight: 1.0
- DAgger teacher sample weight: 1.0
- DAgger student sample weight: 1.0

### Probe outcomes

Notable probes run during this iteration:

- learner-state weighting (`1.0 / 1.5 / 4.0` for expert / DAgger-teacher / DAgger-student) with per-epoch eval: **29.425** avg length, **226.27** avg steps over 200 benchmark games
- per-epoch checkpoint selection only: **29.19** avg length, **222.4** avg steps over 200 benchmark games
- fixed-seed eval plus per-epoch checkpoint selection: **28.595** avg length, **215.65** avg steps over 200 benchmark games
- unchanged backbone, seed 7, fixed shared benchmark seed: **29.98** avg length, **228.675** avg steps over 200 benchmark games
- unchanged backbone, seed 19, fixed shared benchmark seed: **30.8** avg length, **238.92** avg steps over 200 benchmark games

Takeaways:

- heavy learner-state weighting consistently hurt closed-loop performance
- per-epoch checkpoint selection overfit evaluation noise or held-out seed idiosyncrasies instead of helping
- deterministic held-out evaluation was useful for probe comparison, but it did not identify a configuration that beat the previous historical best on the official random benchmark
- run-to-run variance remains high even on the same backbone, so the benchmark leader did not remain the same across probe and main-run protocols

### Best checkpoint during training

Best checkpoint for the official selected run was found at **0 timesteps** after aggregation round 2 (`label: dagger_round_2`).

Eval metrics at the best checkpoint:

- avg snake length: **29.275**
- avg steps: **228.725**
- avg reward: **43.67624999999988**
- max snake length during eval: **64**
- median snake length: **29.0**
- selection score (`avg_len * 1000 - avg_steps`): **29046.275**
- efficiency score (`avg_len^2 / avg_steps`): **3.746970092922746**

Supervised fit at the selected checkpoint:

- final DAgger round loss: **0.014997121637215264**
- final DAgger round accuracy: **0.995027927797999**

### Final benchmark on saved best model

Benchmark results over **1000 games**:

- avg snake length: **29.682**
- avg steps: **226.655**
- avg reward: **44.15619999999986**
- max snake length: **64**
- median snake length: **29.0**
- efficiency score (`avg_len^2 / avg_steps`): **3.887057969160177**
- selection score (`avg_len * 1000 - avg_steps`): **29455.345**
- terminal events: **998 deaths, 2 full boards**

### Comparison to iteration 4

Iteration 4 benchmark:

- avg snake length: 30.627
- avg steps: 240.345
- efficiency score: 3.9027777944205204
- selection score: 30386.655

Iteration 5 change relative to iteration 4:

- avg snake length: **-0.945**
- avg steps: **-13.69**
- efficiency score: **-0.015719825260343434**
- selection score: **-931.3099999999977**

### Relevant notes

- The iteration hypothesis was falsified: learner-state reweighting and finer-grained checkpoint selection did not produce a better official model.
- The strongest probe under a shared fixed benchmark seed was `artifacts/iteration5_probe_seed19_backbone`, but that advantage did not survive the official 1000-game random-start rerun.
- The saved model was copied to `best/iteration5.zip`.
- `best/iteration5.zip` matches `artifacts/iteration5_main_seed19/best_model.zip` (`shasum 54e3e22415da8bbcb1916559b629e1832e5bbda4`).
- `PYTHONPATH=. .venv/bin/pytest --ignore=snake/tests/gui` passed with **18 passed, 1 skipped**.
