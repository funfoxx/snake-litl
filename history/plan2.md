## Iteration 2 Plan

### Baseline

Iteration 1 improved the score from `0.000441791399` to `3.616569364` by switching to:

- a compact egocentric observation
- relative left/straight/right controls
- reward shaping for food progress and anti-stall behavior
- a longer Stable-Baselines3 `DQN` run with score-based checkpoint selection

The best checkpoint during that run reached:

- best eval score: `3.872327106`
- best eval avg snake length: `25.02`
- best eval avg steps: `161.66`

The final `1000`-episode benchmark for the saved best model was:

- avg snake length: `23.317`
- avg steps: `150.331`
- score: `3.616569364`

### Hypothesis

The current agent is no longer failing because of sparse learning. It is failing because the observation still does not tell the DQN enough about which local move preserves future space versus which move walks into a trap a few steps later.

The next improvement should therefore come from giving the DQN stronger action-aware state features rather than changing algorithms or only increasing training time.

### Changes

1. Replace the mostly local occupancy observation with a more structured action-conditioned feature vector.
   - For each relative action (`left`, `straight`, `right`), expose:
   - immediate safety
   - whether the food remains reachable
   - normalized shortest-path distance to food after the move
   - reachable free-space ratio after the move
   - local branching / escape information from the successor state
   - whether the successor state can still reach the tail
   - whether the move reduces Manhattan distance to food
   - if the move is fatal, use a strong negative feature pattern so the network can separate bad actions easily

2. Keep the model a `DQN`, but retune training around the richer features.
   - keep `stable_baselines3.DQN`
   - make iteration/config values explicit in the training script
   - support short smoke runs and the main run without editing past iteration artifacts
   - consider a slightly larger MLP and longer training horizon if smoke results justify it

3. Keep reward shaping conservative.
   - preserve food reward, death penalty, anti-stall truncation, and food-distance shaping
   - only adjust shaping if the new observation clearly needs it

4. Improve experiment logging.
   - save smoke artifacts under `artifacts/iteration2/`
   - save only the best checkpoint from the main run to `best/iteration2.zip`
   - record benchmark/test outputs in `history/out2.md`

### Expected Outcome

If the action-aware features expose future survivability well enough, the DQN should avoid self-closing loops more often, reach longer snakes before dying, and improve the project score beyond iteration 1 without requiring a different algorithm.

### Main Risk

The hand-engineered successor-state features could become too noisy or too expensive relative to their value. Smoke runs should catch that quickly; if they do not outperform the previous observation at small scale, the main run should not be extended blindly.
