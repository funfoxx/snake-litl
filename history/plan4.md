## Iteration 4 Plan

### Baseline

Iteration 3 moved the project to a custom PyTorch `DQN` with double-DQN targets, prioritized replay, and a dueling head. That raised the final `1000`-game benchmark to:

- avg snake length: `26.447`
- avg steps: `171.717`
- score: `4.073235667`

The main remaining gap is efficiency. The policy now grows well, but it still spends too many extra steps per food relative to the repository's documented DQN benchmark (`24.44` length, `131.69` steps, score about `4.53`).

### Hypothesis

The current agent has two weak points:

1. Reward credit still travels mostly one step at a time. Even with shaping, the trainer may under-value short efficient food-collection sequences versus long safe wandering sequences.
2. The observation and reward do not explicitly encode how efficient the current chase has been relative to the food's initial difficulty.

The next gain should therefore come from combining a stronger DQN return target with more direct efficiency signal:

- add n-step returns to the custom DQN trainer
- add pursuit-efficiency features to the observation
- shape rewards around food-collection efficiency, not just survival and eventual progress

### Changes

1. Upgrade the custom trainer while remaining a `DQN`.
   - keep double-DQN target selection, prioritized replay, and the dueling network
   - add configurable n-step returns in the replay path
   - keep score-based checkpointing, but make checkpoint evaluation a bit more robust during the main run

2. Add efficiency-aware environment state features.
   - expose the current shortest-path distance to food
   - expose how long the current food chase has taken relative to the shortest path when that food appeared
   - add per-action detour information so the network can distinguish direct food progress from safe but wasteful movement

3. Align reward shaping more directly with the project metric.
   - preserve strong food reward, death penalty, and starvation truncation
   - penalize inefficient food collection when the snake takes far more steps than the food initially required
   - keep shortest-path progress shaping, but bias it slightly more against detours and repeated wandering

4. Run iteration 4 in two stages.
   - smoke run first under `artifacts/iteration4/smoke1/`
   - if the smoke score is at least competitive with iteration 3, run the main training job and export only its best checkpoint to `best/iteration4.zip`

### Expected Outcome

If n-step targets and efficiency-aware shaping work together, the agent should retain the stronger growth from iteration 3 while reducing average steps enough to push the final score meaningfully above `4.07`.

### Main Risks

1. Efficiency pressure could become too strong and make the agent die early instead of taking safe setup moves.
2. N-step returns plus prioritized replay can destabilize training if the effective target scale becomes too large.
3. Added pursuit features may help only early in the episode and not at larger snake lengths, where trap avoidance still dominates.
