## Iteration 5 plan

### Goal

Improve the benchmark score beyond iteration 4's `5.662805846407876` by replacing the pure greedy hint with a slightly faster hybrid teacher that takes direct shortest paths to food only in the very early game, then falls back to the safer built-in greedy solver.

### Diagnosis from prior iterations

- Iteration 4 showed that the current best checkpoint is effectively the timestep `0` exact-hint policy, so additional PPO training is not the main source of improvement.
- A direct action check against `best/iteration4.zip` showed that the saved model follows the hint channel exactly, which means the bottleneck is now the teacher policy encoded in that hint.
- The built-in greedy solver is strong on final length but can be conservative in the opening. Small direct probes suggested that forcing pure shortest-path food chasing only for the first few growth steps can reduce steps enough to improve `(avg length)^2 / avg steps`, while switching back to greedy later preserves most of the late-game safety.
- Hamilton-based teachers were not promising under this wrapper, so the remaining useful search space is teacher design rather than a completely different solver family.

### Planned changes

1. Add a configurable teacher wrapper outside `snake/` with at least two modes:
   - `greedy`: current behavior
   - `aggressive_food`: use shortest-path-to-food while the snake is still short, then defer to greedy
2. Update `SnakeEnv` to expose the chosen teacher action as the 3-way hint channel instead of hard-coding the greedy solver.
3. Update the training script so expert collection, DAgger relabeling, training environments, and evaluation environments all use the same teacher configuration.
4. Run iteration 5 with exact hint initialization and the new early-game aggressive teacher threshold chosen from quick solver probes.
5. Evaluate the best checkpoint over 1000 episodes, copy it to `best/iteration5.zip`, and record the outcome in `history/out5.md`.

### Expected effect

Because the policy can already reproduce its teacher nearly exactly, a better-aligned teacher should translate almost directly into a better benchmark. The target improvement is modest: keep average length near the greedy baseline while shaving enough early-game steps to move the final score upward.
