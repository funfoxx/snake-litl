## Iteration 5 Plan

### Baseline

Iteration 4 improved the final benchmark to:

- avg snake length: `30.700`
- avg steps: `220.912`
- score: `4.266359455`

The custom trainer is already using a defensible DQN setup:

- dueling network
- double-DQN target selection
- prioritized replay
- relative 3-action control
- score-based checkpoint selection with disjoint eval and benchmark seeds

The remaining problem is efficiency. The agent grows longer than the repository's documented DQN benchmark, but it still uses too many steps per unit of growth.

### Hypothesis

The current LTIL state representation is likely over-compressed for efficient route choice. It exposes strong local/action-conditioned heuristics, but it no longer gives the network an explicit global board layout like the original project DQN did.

That missing global context is a plausible reason for the current behavior:

- the agent can identify many safe actions
- it can continue surviving and growing
- but it may not distinguish the shortest clean route from a longer safe loop until too late

The next improvement should therefore come from combining:

1. the current engineered successor-state features that already helped training stability
2. a relative global board encoding that exposes the actual geometry of food, body, and open space
3. a DQN network that can process the board spatially instead of forcing all information through one MLP

### Changes

1. Extend `snake_env.py` with a hybrid observation.
   - add a relative `8x8x4` one-hot board encoding
   - add immediate danger bits for the three relative actions
   - keep the existing compact global progress and successor-state action features

2. Upgrade `train_test.py` to a hybrid dueling DQN architecture.
   - process the board slice with a small CNN
   - process the auxiliary features with an MLP
   - fuse both streams before the value/advantage heads
   - keep the algorithm a `DQN`

3. Keep training and reward shaping conservative at first.
   - preserve iteration-4 reward shaping and checkpoint selection
   - keep double-DQN targets, prioritized replay, and dueling heads
   - only tune hyperparameters if the smoke run suggests the hybrid state changes the optimal regime

4. Preserve experiment discipline required by `AGENTS.md`.
   - write smoke artifacts under `artifacts/iteration5/`
   - export only the best main-run model to `best/iteration5.zip`
   - record smoke/main/benchmark/test results in `history/out5.md`

### Expected Outcome

If the hybrid observation restores enough board-level context, the agent should make fewer unnecessary detours while keeping the stronger survival behavior from iterations 3 and 4. The target is to improve the final `1000`-episode benchmark beyond `4.266`.

### Main Risks

1. The larger state and CNN could slow learning enough that the existing short training horizon becomes suboptimal.
2. The engineered action features and raw board input could partially duplicate information and make optimization noisier.
3. The new architecture may improve long-horizon survival more than step efficiency, which would increase average length without improving the project score.
