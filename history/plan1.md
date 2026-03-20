## Iteration 1 Plan

### Baseline

Iteration 0 used Stable-Baselines3 DQN with a mostly default MLP setup, 10,000 training steps, sparse rewards (`+1` food, `-1` death, `0` otherwise), and a raw `10x10` board observation. Reported benchmark:

- avg snake length: `2.083`
- avg steps: `9821.126`
- score: `0.000441791399`

This is effectively a stalling policy. The average episode length is near the environment truncation limit, so the immediate priority is reducing useless wandering while keeping food-seeking behavior learnable.

### Hypothesis

The current setup is weak for three reasons:

1. The observation is large and poorly structured for the simple MLP DQN baseline.
2. Sparse rewards provide almost no signal before the first food.
3. Training is too short to give the agent a chance to stabilize.

### Changes

1. Replace the raw board observation with a compact engineered feature vector for DQN.
   - Immediate collision risk for neighboring moves.
   - Current heading one-hot.
   - Relative food direction.
   - Normalized distances from head to food and walls.
   - Small local occupancy scan around the head.

2. Reshape rewards to discourage stalling and make progress toward food visible.
   - Positive reward for food.
   - Strong negative reward for death.
   - Small step penalty.
   - Reward/penalty based on whether the move decreases or increases Manhattan distance to food.
   - Mild revisit penalty using per-episode visit counts.

3. Improve training and evaluation code.
   - Increase total training timesteps substantially.
   - Use a tuned DQN configuration for this small discrete environment.
   - Add periodic evaluation with the project score formula.
   - Save the best model from the main run only.
   - Fix the current evaluation script issues.

### Expected Outcome

If the shaping is reasonable, the agent should stop timing out nearly every episode and should raise both average length and score materially above iteration 0. The main risk is over-shaping toward greedy food chasing that causes more deaths; evaluation over many episodes should reveal that quickly.
