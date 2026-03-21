## Iteration 4 plan

### Goal

Improve the benchmark score beyond iteration 3's `5.580994307775686` by restoring the environment to the original game's fixed 4-body start state and retraining the greedy-hinted PPO policy on that distribution.

### Diagnosis from prior iterations

- Iteration 3's main gain came from exposing the teacher action directly and behavior-cloning a policy that could nearly reproduce the built-in greedy solver.
- The current wrapper still differs from the original game in one important way: `SnakeEnv.reset()` randomizes the initial snake and starts from length `2`.
- A direct spot check under a fixed 4-body start matching the original game configuration produced a stronger greedy benchmark than the current random-start setup, which suggests the random reset is now the main performance bottleneck rather than policy capacity.
- The built-in Hamilton solver is not robust to the random-start wrapper, so aligning the reset state with the original game also keeps future teacher options open without touching `snake/`.

### Planned changes

1. Update `SnakeEnv` so the default reset state matches the original game:
   - fixed initial direction `RIGHT`
   - fixed initial body positions `[Pos(1, 4), Pos(1, 3), Pos(1, 2), Pos(1, 1)]`
   - random food placement preserved
   - keep the option to request the old random start explicitly for future experiments
2. Keep the successful iteration 3 observation design, including the greedy action hint and full-board channels.
3. Retrain the PPO policy with behavior cloning on the fixed-start environment and keep score-based checkpoint selection.
4. Evaluate the best checkpoint over 1000 episodes, copy it to `best/iteration4.zip`, and record the results in `history/out4.md`.

### Expected effect

Because the fixed 4-body start removes difficult low-length random initial states while staying aligned with the original game setup, the same teacher-hinted policy should achieve higher average length at comparable or lower step counts. That should raise `(avg length)^2 / avg steps` without requiring a riskier architecture change.
