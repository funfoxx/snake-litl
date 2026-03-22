## Iteration 4 Plan

### Motivation

Iteration 3 improved behavior cloning with DAgger and reached:

- avg snake length: 30.817
- avg steps: 241.904
- efficiency score: 3.925885843144388

The remaining gap is still large relative to the search teacher:

- greedy teacher in the current environment: about 59 average length
- learned DAgger policy: about 31 average length

Previous iterations also showed that naive RL fine-tuning was destructive. The likely reason is that DQN was being asked to resume learning from an almost empty replay buffer, so early updates were dominated by a small amount of noisy online experience rather than the much stronger demonstration state distribution already available.

### Changes

1. Extend dataset collection to keep full control transitions, not only imitation labels.
   - Record `obs`, `next_obs`, executed action, reward, and done for expert rollouts.
   - Record the same transition data for DAgger rollouts using the actual control action, while still keeping teacher labels for supervised updates.

2. Warm-start Q-learning from the aggregated demonstration replay.
   - Add a replay-buffer prefill stage after imitation / DAgger.
   - Seed the SB3 replay buffer with the collected control transitions before online RL begins.
   - Tighten warm-start exploration and learning-rate settings so RL fine-tuning is conservative instead of destructive.

3. Keep checkpoint selection evaluation-driven.
   - Continue evaluating after imitation and after each DAgger round.
   - Evaluate during RL and keep whichever checkpoint actually maximizes the selection score.

### Intended probe and main-run direction

- Start from the iteration 3 backbone: expert imitation plus two DAgger rounds.
- Add replay prefill and a nontrivial RL budget such as a few thousand timesteps.
- Use low exploration during warm-start RL because the model should exploit the seeded replay instead of re-discovering basic behavior.

### Expected effect

- Higher average snake length than iteration 3 by letting value updates refine the cloned policy without first collapsing under low-quality early replay.
- Similar or lower average steps would be ideal, but a modest step increase is acceptable if the length gain is material.
