## Iteration 1 plan

### Goal

Improve the benchmark score `(avg length)^2 / avg steps` by fixing the current RL setup's two biggest problems:

1. The agent gets almost no learning signal except food/death.
2. Episodes can stall for thousands of steps, which destroys the score even when the snake survives.

### Diagnosis from iteration 0

- Baseline result: average length `2.083`, average steps `9821.126`, score `0.000441791399`.
- The existing environment exposes a full grid but leaves action semantics awkward: choosing the opposite direction becomes a wasted step.
- The reward is extremely sparse and the episode timeout is so large that the learned policy can wander almost indefinitely.

### Planned changes

1. Replace the observation with a compact egocentric feature vector:
   - immediate danger flags relative to the current heading
   - food direction relative to the head
   - current heading one-hot
   - normalized length / progress features
   - local occupancy / wall distance signals
2. Replace absolute actions with relative actions `{turn left, go straight, turn right}` to remove invalid reverse moves.
3. Add reward shaping:
   - positive reward for eating food
   - strong negative reward for death
   - small step penalty
   - shaped reward for moving closer to food
4. Add an idle limit based on time since last food so episodes terminate before reaching uselessly large step counts.
5. Switch the trainer to PPO for this dense, low-dimensional control setup.
6. Save the best PPO checkpoint during training, then evaluate it over 1000 games and copy that checkpoint to `best/iteration1.zip`.

### Expected effect

Even if the learned policy does not reach very large snake lengths yet, cutting average steps down by orders of magnitude while getting the snake to consistently eat a few pieces of food should materially improve the benchmark score over iteration 0.
