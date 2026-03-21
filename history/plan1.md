## Iteration 1 plan

Baseline iteration 0 used an almost default SB3 DQN on the raw absolute board with sparse rewards and produced near-random behavior (`avg snake length 2.083`, `avg steps 9821.126`).

This iteration will keep the model family as DQN but change the problem formulation so the agent can actually learn useful control:

1. Replace the absolute 4-action interface with a relative 3-action interface (`turn left`, `go straight`, `turn right`).
2. Replace the raw categorical board observation with a compact egocentric feature vector:
   - immediate danger in front/left/right
   - food direction in snake-relative coordinates
   - normalized distances to wall/body/food along 8 rays
3. Add reward shaping that still prioritizes food, but also:
   - rewards moving closer to food
   - penalizes moving away from food
   - mildly penalizes each step
   - penalizes repeated states / stalling
   - gives a stronger terminal penalty on death
4. Reduce episode truncation from the old `10000` step cap to a dynamic budget tied to snake length so training pressure favors efficient food collection.
5. Replace the broken one-off script with a real training/evaluation pipeline that:
   - trains SB3 `DQN` with tuned hyperparameters
   - evaluates periodically and saves the best checkpoint
   - runs a fixed benchmark pass after training
   - writes machine-readable results for the final report

Planned hyperparameters for the main run:

- algorithm: `stable_baselines3.DQN`
- policy: `MlpPolicy`
- total timesteps: `250000`
- learning rate: `3e-4`
- buffer size: `100000`
- learning starts: `5000`
- batch size: `256`
- gamma: `0.99`
- train frequency: `4`
- gradient steps: `1`
- target update interval: `2000`
- exploration fraction: `0.35`
- exploration final epsilon: `0.02`
- network architecture: `[256, 256, 128]`
- eval frequency: `5000`
- eval episodes per checkpoint: `20`
- final benchmark episodes: `1000`

Success criterion for this iteration: materially beat iteration 0 on average snake length while also cutting average steps by a large margin.
