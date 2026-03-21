## Iteration 3 plan

### Goal

Improve the benchmark score beyond iteration 2's `3.2433802798310456` by making the policy capable of imitating the much stronger built-in greedy solver and then fine-tuning from that initialization with PPO.

### Diagnosis from prior iterations

- Iteration 1's compact egocentric PPO setup was a major improvement over the raw-grid baseline, but iteration 2 showed that better checkpoint selection alone did not produce a better 1000-game benchmark.
- Under the current environment wrapper, the greedy solver is still much stronger than the learned PPO policy. A direct measurement over 200 games gave:
  - average length `59.125`
  - average steps `637.35`
  - score `5.484844473209383`
- The codebase already contains behavior-cloning support, so the obvious next lever is to imitate that greedy policy.
- A scratch behavior-cloning test on the current 16-feature observation failed badly:
  - 100 demonstration episodes, 5 BC epochs
  - validation accuracy about `0.816`
  - rollout score only `0.061929967426710086`
- That mismatch means the current observation is too aliased: many greedy decisions look similar in the compact local state, so the policy can match labels often enough in-distribution without learning a rollout-stable control policy.

### Planned changes

1. Replace the observation with a richer feature vector that exposes more of the board structure while keeping the problem small enough for an MLP policy:
   - immediate danger and local clearance features
   - relative food / tail direction features
   - heading one-hot
   - 8-direction ray features for obstacle distance plus food/body visibility
   - progress features such as normalized length and starvation counter
2. Keep the relative `{left, straight, right}` action space and the existing reward / truncation setup unless debugging shows a regression there.
3. Improve the trainer so iteration 3 can be run cleanly with its own artifact paths and configurable hyperparameters.
4. Run greedy behavior cloning with the richer observation to produce a usable starting policy, then fine-tune with PPO using lower-entropy settings so RL improves rather than immediately destroying the imitation prior.
5. Evaluate the best checkpoint over 1000 games, copy that model to `best/iteration3.zip`, and write the full results in `history/out3.md`.

### Expected effect

If the richer observation removes the worst state aliasing, behavior cloning should start near greedy-like behavior instead of near-random control. That should give PPO a much stronger initialization and a realistic path to beating the current learned-policy score, even if it does not fully match the greedy solver.
