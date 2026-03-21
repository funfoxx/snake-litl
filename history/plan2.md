## Iteration 2 plan

### Goal

Improve the benchmark score beyond iteration 1's `3.3172447799390135` by keeping the strong compact egocentric PPO setup from iteration 1 while replacing reward-based checkpoint selection with direct score-based checkpoint selection and re-running the training with a small seed search.

### Diagnosis from prior iterations

- Iteration 1 fixed the sparse-reward and runaway-episode issues and produced a large jump to an average length of `19.981` in `120.353` steps.
- The main remaining issue is model selection: the training loop was saving checkpoints by shaped reward, while the actual benchmark is `(avg length)^2 / avg steps`.
- Because the metric is sensitive to both survival and efficiency, the best-reward checkpoint is not guaranteed to be the best-score checkpoint.
- PPO performance is also seed-sensitive, so a small seed sweep is a cheap way to search for a better solution once the environment is already stable.

### Planned changes

1. Keep the compact 16-feature egocentric observation and relative action space from iteration 1 as the main training setup.
2. Replace `EvalCallback` with a custom evaluation callback that:
   - runs deterministic benchmark episodes during training
   - computes `avg length`, `avg steps`, and the true score `(avg length)^2 / avg steps`
   - saves the best checkpoint according to score instead of shaped reward
3. Update the training script so iteration 2 has its own `eval/iteration2` and `best/iteration2.zip` artifacts.
4. Run a small seed sweep with the score-based callback and keep only the strongest resulting checkpoint.
5. Evaluate the best saved checkpoint over 1000 episodes and record the results in `history/out2.md`.

### Expected effect

If reward and score are misaligned, selecting checkpoints directly by the benchmark should improve final performance without destabilizing the already-good PPO setup. The seed sweep is intended to capture a better local optimum at low implementation cost.
