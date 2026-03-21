## Iteration 4 plan

Iteration 3 recovered much of iteration 2's step inflation, but it did so by pushing the policy too far toward short-term efficiency. The trap-aware observation is still useful, so the next iteration should keep that state representation and change the shaping logic so safety acts more like a constraint than a standing reward.

Hypothesis:

- iteration 2 became too slow partly because increasing reachable space was directly rewarded, which can pay the agent for cautious detours
- iteration 3 reduced that pressure, but it still uses a symmetric space-delta term, so the agent is learning from a signal that mixes "avoid traps" with "wander into larger open regions"
- a better balance is:
  - penalize moves that shrink future reachable space
  - do not reward moves just because they expand space
  - add a small bonus for moving closer to food when the chosen move remains nearly as safe as the best safe alternative

This iteration will keep the model family as DQN and focus on four changes:

1. Replace symmetric reachable-space shaping with asymmetric trap shaping.
   - Remove the positive reward for increasing reachable space.
   - Keep only a penalty when the chosen move reduces reachable space.
   - Add a small extra penalty if the resulting state makes food unreachable.

2. Add a safe-progress bonus.
   - Before each move, compare the chosen relative action to the available safe actions.
   - If the chosen move decreases food distance while preserving reachable space close to the best safe option, give a small bonus.
   - This should encourage direct food collection when it is not obviously reckless.

3. Make the new shaping tunable from `train_test.py`.
   - Expose:
     - space-loss penalty scale
     - food-unreachable penalty
     - safe-progress bonus
     - safe-progress tolerance margin
   - Keep the rest of the iteration-3 pipeline intact so probes stay comparable.

4. Probe around the new shaping before one main run.
   - Run short probes to compare balanced variants of the new asymmetric shaping.
   - Choose the main-run configuration using score first, then average length.

Planned probe settings:

- common DQN settings:
  - algorithm: `stable_baselines3.DQN`
  - policy: `MlpPolicy`
  - network: `[256, 256, 256, 128]`
  - learning rate: `1e-4`
  - buffer size: `100000`
  - learning starts: `5000`
  - batch size: `256`
  - gamma: `0.99`
  - train frequency: `4`
  - gradient steps: `1`
  - target update interval: `2000`
  - exploration final epsilon: `0.02`
  - checkpoint selection: score first, then average length, then lower steps

- candidate A:
  - timesteps: `12000`
  - step penalty: `-0.028`
  - closer reward: `+0.25`
  - farther penalty: `-0.40`
  - space-loss scale: `0.18`
  - food-unreachable penalty: `-0.25`
  - safe-progress bonus: `+0.18`
  - safe-progress tolerance: `0.08`
  - revisit penalty: `-0.22`
  - starvation limit: `30 + 6 * snake_len`
  - no-progress: `6 / -0.12`
  - exploration fraction: `0.22`

- candidate B:
  - timesteps: `12000`
  - step penalty: `-0.030`
  - closer reward: `+0.28`
  - farther penalty: `-0.42`
  - space-loss scale: `0.16`
  - food-unreachable penalty: `-0.30`
  - safe-progress bonus: `+0.22`
  - safe-progress tolerance: `0.08`
  - revisit penalty: `-0.20`
  - starvation limit: `28 + 6 * snake_len`
  - no-progress: `5 / -0.14`
  - exploration fraction: `0.20`

- candidate C:
  - timesteps: `12000`
  - step penalty: `-0.027`
  - closer reward: `+0.25`
  - farther penalty: `-0.38`
  - space-loss scale: `0.20`
  - food-unreachable penalty: `-0.20`
  - safe-progress bonus: `+0.15`
  - safe-progress tolerance: `0.10`
  - revisit penalty: `-0.22`
  - starvation limit: `32 + 6 * snake_len`
  - no-progress: `6 / -0.12`
  - exploration fraction: `0.22`

Main-run target:

- promote the best probe by score, then average length
- run one main training at `25000` timesteps unless the probes show a clearly earlier peak
- evaluate checkpoints every `5000` timesteps
- select the best checkpoint using the existing score-first rule
- benchmark the selected checkpoint over `1000` games

Success criterion:

- improve the score above iteration 3 (`0.166265`)
- keep average snake length clearly above iteration 1 (`20.233`)
- reduce or at least hold average steps near the iteration-3 range while preserving more of iteration-2's length
