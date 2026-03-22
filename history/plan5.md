## Iteration 5 Plan

### Motivation

Iteration 4 confirmed that short RL fine-tuning is still not the path forward:

- replay-prefilled RL probes regressed relative to pure DAgger
- the official result improved only by choosing a stronger seed, not by changing the learning procedure
- the final aggregated imitation dataset remained dominated by clean expert states, while only a much smaller fraction came from learner-induced states

That makes the current bottleneck look like **dataset weighting**, not architecture or RL stability. DAgger helped in iteration 3, but the refit still trains mostly on expert-distribution states because the initial demonstration set is so large. If covariate shift remains the main failure mode, the policy should benefit from assigning more importance to learner-visited states, especially the transitions where the student actually controlled the rollout.

### Changes

1. Add per-transition imitation weights.
   - Keep expert transitions at a configurable base weight.
   - Assign separate configurable weights to DAgger transitions collected under teacher control and under student control.
   - Use those weights directly in the supervised cross-entropy objective.

2. Improve checkpoint selection granularity.
   - Add optional evaluation after every imitation / DAgger epoch.
   - Keep using the evaluation selection score so the saved checkpoint can come from the middle of a refit round instead of only its endpoint.

3. Probe a learner-state-heavy configuration before the official run.
   - Start from the iteration 4 backbone and seed 29, since that seed was strongest there.
   - Increase the weight on student-controlled DAgger transitions materially above expert transitions.
   - Keep RL effectively disabled unless a probe unexpectedly shows a genuine gain.

### Intended probe direction

- seed: 29
- imitation episodes: 160
- DAgger rounds: 2
- DAgger episodes: 30
- beta schedule: 0.7 -> 0.2
- expert sample weight: 1.0
- DAgger teacher-controlled weight: modestly above expert
- DAgger student-controlled weight: clearly above expert
- evaluate every epoch to capture intra-round peaks

### Expected effect

- Higher average snake length than iteration 4 by making the optimizer spend more capacity on off-distribution recovery states rather than further polishing already-mastered expert states.
- Similar steps would be ideal; a moderate step increase is acceptable if the length gain is real and survives the 1000-game benchmark.
