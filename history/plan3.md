## Iteration 3 Plan

### Motivation

Iteration 2 showed a large mismatch between supervised fit and closed-loop control quality:

- teacher benchmark in the current environment: about 59 average length
- imitation training accuracy: about 99.1%
- learned policy benchmark: 28.152 average length

That gap strongly suggests **covariate shift**. The policy predicts expert actions well on expert-visited states, but once it makes a small mistake it enters states that were rare or absent in the demonstration set and degrades quickly. Iteration 2 also showed that ordinary RL fine-tuning was destructive even at short budgets, so the next experiment should improve the imitation dataset rather than push harder on Q-learning.

### Changes

1. Replace one-shot behavior cloning with dataset aggregation.
   - Keep the existing `GreedySolver` as the labeling oracle because it is stable in this environment.
   - Collect an initial teacher dataset as before.
   - Pretrain the policy on that dataset.
   - Run one or more **DAgger-style relabeling rounds** where the current policy controls the snake while the teacher labels every visited state with the expert action.
   - Mix teacher-forced and student-forced actions during aggregation so rollouts stay near the task while still exposing the model to its own off-policy states.

2. Keep RL updates negligible unless they clearly help.
   - Continue selecting the best checkpoint from evaluation history.
   - Evaluate after the initial imitation stage and after each aggregation round.
   - Preserve the existing benchmark flow and results artifact format, but extend it with aggregation summaries.

3. Improve experiment bookkeeping.
   - Record per-round aggregation statistics such as transitions, teacher-forced rate, and action distribution.
   - Make the training script configurable from the CLI for aggregation rounds, episodes, beta schedule, and fine-tuning epochs.

### Intended main-run parameters

- train steps: 1
- eval frequency: 2000
- eval episodes: 40
- benchmark episodes: 1000
- seed: 7
- initial expert episodes: 120 to 160
- initial imitation epochs: around 8 to 10
- aggregation rounds: 2 to 4
- aggregation episodes per round: around 20 to 60
- teacher-action mixture (`beta`) decaying across rounds

### Expected effect

- Higher average snake length by training on states induced by the learner rather than only clean expert trajectories.
- Better stability late in the episode, which is where single-mistake compounding currently collapses performance.
- Similar or slightly higher steps than iteration 2 are acceptable if the length gain is material; the preferred result is higher length with only a modest step increase.
