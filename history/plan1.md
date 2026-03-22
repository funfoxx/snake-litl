## Iteration 1 Plan

### Motivation

Iteration 0 used a nearly default SB3 DQN on a sparse-reward, absolute-action grid observation and produced an agent that barely grows (`avg snake length: 2.083`) while consuming almost the full truncation budget (`avg steps: 9821.126`).

The main bottlenecks are:

- Sparse rewards make it easy to learn survival loops instead of food-seeking.
- Absolute actions waste symmetry and force the policy to relearn equivalent situations for each heading.
- The current observation does not explicitly expose immediate danger or food direction.
- The train/test harness is minimal and fragile.

### Changes

1. Replace the environment wrapper with a richer RL interface while keeping the underlying game logic unchanged.
   - Use **relative actions**: `turn left`, `go forward`, `turn right`.
   - Use a **rotated egocentric observation** so the snake always faces "up" in policy space.
   - Provide a **dict observation** combining a compact multi-channel board view with local features such as immediate danger and food direction.
   - Add **reward shaping** for food, death, distance-to-food progress, and a small step cost.
   - Add a **stall limit** based on time since last food instead of a flat 10000-step truncation.

2. Replace the single-script baseline with a more robust experiment harness.
   - Train with SB3 DQN using `MultiInputPolicy`.
   - Evaluate periodically and save the best checkpoint.
   - Run a deterministic benchmark over many episodes after training.
   - Log enough information to write a meaningful result summary.

3. Validate with a short smoke run before the main run.

### Expected effect

- Higher average snake length from denser learning signal and action symmetry.
- Lower average steps because the agent is penalized for drifting and stalled episodes terminate earlier.
- More reliable training/evaluation artifacts for future iterations.
