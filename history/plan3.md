## Iteration 3 Plan

### Baseline

Iteration 2 improved the benchmark score to `3.992963098` with:

- relative left/straight/right actions
- action-conditioned successor-state features
- shaped rewards for food, death, anti-stall behavior, and food progress
- a vanilla `stable_baselines3.DQN` training loop with score-based checkpointing

The strongest checkpoint during iteration 2 reached an eval score of `4.216517024` at `115000` steps, but the final benchmark of the saved best model still underperformed the repository's original documented DQN efficiency. The remaining gap is likely not caused by missing state signal alone.

### Hypothesis

The current setup is still using a fairly plain DQN optimizer. The repository documentation explicitly reports that prioritized replay and a dueling value head improve Snake DQN performance, and those ideas have not been used in the LTIL iterations yet.

The next gain should therefore come from upgrading the training algorithm while staying inside the "DQN only" constraint:

1. keep the discrete DQN action-value setup
2. add prioritized replay so rare but important failures/successes are revisited more often
3. add double-DQN targets and a dueling head to reduce value overestimation and improve action discrimination
4. make reward shaping slightly more score-aligned so long detours are less acceptable

### Changes

1. Replace the SB3 training loop with a custom PyTorch DQN trainer outside `snake/`.
   - dueling network
   - double-DQN target computation
   - prioritized replay
   - periodic hard target sync
   - score-based checkpoint selection

2. Keep the relative-action observation design, but add a bit more efficiency signal.
   - preserve action-conditioned safety / reachability features
   - add successor-tail distance and safe-space margin features
   - keep the observation compact enough for an MLP DQN

3. Adjust rewards to penalize inefficient food chasing more directly.
   - replace Manhattan-only progress shaping with shortest-path progress when available
   - slightly increase time pressure as `steps_since_food` grows
   - preserve strong food reward, death penalty, and starvation truncation

4. Preserve experiment discipline required by `AGENTS.md`.
   - write iteration-3 artifacts under `artifacts/iteration3/`
   - run a smoke experiment before the main run
   - export only the best main-run model to `best/iteration3.zip`
   - record all results in `history/out3.md`

### Expected Outcome

If the custom DQN trainer stabilizes learning better than vanilla SB3 DQN, the agent should reach a better score-efficient regime: similar or better average length than iteration 2, but with fewer wasted steps per food. A realistic target is to move the final `1000`-episode benchmark above iteration 2 and closer to or beyond the repository's documented DQN efficiency.

### Main Risks

1. A fresh custom DQN implementation may train less stably than SB3 if the replay or target updates are mis-tuned.
2. Stronger anti-wandering reward pressure could shorten episodes too aggressively and reduce average length.
3. The richer trainer may need slightly different exploration timing than the SB3 runs; the smoke run is intended to catch that before the main experiment.
