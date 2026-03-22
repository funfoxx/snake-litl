## Iteration 2 Plan

### Motivation

Iteration 1 made the environment and policy representation much better and reached a strong benchmark for a short run:

- avg snake length: 18.574
- avg steps: 118.26
- efficiency score: 2.917245695924235

The clearest remaining bottleneck is **sample efficiency**. The current agent learns from scratch, while the repository already contains search-based teachers that solve the game far better than the learned policy. A quick benchmark of the built-in `GreedySolver` inside the current egocentric relative-action environment showed:

- avg snake length: about 58.6 over 100 seeded episodes
- avg steps: about 684.36

That makes demonstration-guided initialization the highest-upside next move under the repo constraints.

### Changes

1. Add an imitation-learning stage before RL fine-tuning.
   - Roll out `GreedySolver` in the current `SnakeEnv`.
   - Convert teacher directions into the relative action space already used by the policy.
   - Collect demonstration observations and actions from many seeded episodes.
   - Pretrain the DQN policy network with supervised cross-entropy on the expert action labels.

2. Keep the environment and DQN backbone mostly stable.
   - Reuse the egocentric observation and reward shaping from iteration 1.
   - Reuse the same SB3 DQN implementation and feature extractor unless implementation details force a small adjustment.
   - Synchronize the target network after imitation pretraining so RL starts from the distilled policy.

3. Improve experiment bookkeeping.
   - Evaluate once at timestep 0 after imitation, so a strong warm start can be selected as the best checkpoint if RL later regresses.
   - Save imitation statistics, evaluation history, and benchmark results in `results.json`.
   - Keep the smoke-test and benchmark flow stable for future iterations.

### Expected effect

- Much higher average snake length by importing the search policy's competence instead of discovering it from sparse interaction alone.
- Competitive or improved efficiency after RL fine-tuning, because the shaped rewards still penalize drift and unsafe play.
- Better stability across short training budgets, which is important for iterative experimentation.
