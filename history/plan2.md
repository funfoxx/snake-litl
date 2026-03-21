## Iteration 2 plan

Iteration 1 already solved the biggest formulation issues and reached a strong benchmark (`avg snake length 20.233`, `avg steps 120.153`). The next gain is likely not from another large reward rewrite, but from helping the DQN reason about traps and from selecting checkpoints using the actual project metrics instead of shaped return.

This iteration will keep the model family as DQN and focus on three changes:

1. Expand the egocentric observation with short-horizon survival features.
   - Keep the relative 3-action control and the existing danger / food / ray features.
   - Add current free-space information from the head position.
   - Add tail-relative features.
   - Add one-step lookahead features for each relative action (`left`, `straight`, `right`):
     - whether the next square is blocked
     - normalized Manhattan distance to food after that move
     - reachable-space ratio after that move using flood fill
     - whether food is reachable from that next state

2. Keep reward shaping mostly stable, but add a mild structural signal.
   - Preserve the strong food reward, death penalty, distance shaping, step cost, and starvation cutoff from iteration 1.
   - Add a small penalty when the chosen move sharply reduces reachable free space, so the agent gets feedback before it fully traps itself.
   - Avoid aggressive new shaping that could destroy the efficient policy already learned in iteration 1.

3. Replace reward-based best-model selection with metric-based checkpoint selection.
   - Save periodic checkpoints during the main run.
   - Evaluate checkpoints with deterministic episodes and record:
     - average snake length
     - average steps
     - score (`avg_length / avg_steps`)
     - average return
   - Choose the best checkpoint from the main run using project-aligned metrics:
     - primary: higher average snake length
     - secondary: lower average steps
     - tertiary: higher score

Training plan for the main run:

- algorithm: `stable_baselines3.DQN`
- policy: `MlpPolicy`
- total timesteps: start with short probes, then one main run at `30000` timesteps if the probes look stable
- learning rate: reduce to `1e-4`
- buffer size: `100000`
- learning starts: `5000`
- batch size: `256`
- gamma: `0.99`
- train frequency: `4`
- gradient steps: `1`
- target update interval: `2000`
- exploration fraction: `0.25`
- exploration final epsilon: `0.02`
- network architecture: widen slightly to `[256, 256, 256, 128]`
- periodic checkpoint eval frequency: `5000`
- probe benchmark episodes: `200`
- final benchmark episodes: `1000`

Execution steps:

1. Implement the observation, reward, and checkpoint-selection changes outside `snake/`.
2. Run a smoke test to confirm the new pipeline works.
3. Run one or more short probe trainings to compare the new setup against the old behavior.
4. Run the main iteration-2 training once with the chosen settings.
5. Copy the best checkpoint from that main run to `best/iteration2.zip`.
6. Write `history/out2.md` with the exact run settings and results.

Success criterion:

- Improve on iteration 1's average snake length while keeping average steps at or below a similar range, ideally improving both at once.
