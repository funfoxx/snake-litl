# Snake RL LTIL 

## Goal

Use LLMs to iteratively improve the performance of a reinforcement learning model based on previous attempts and modifications. The LLM should try its best to improve the performance of the model.

## Constraints on code changes

Do not modify anything in the `snake/` folder or this `AGENTS.md` file.

You can add new entries to `history/` and `best/`, but you can not delete or modify any previous entries. Only add one model to `best/` per iteration; only add the best model found during the main run.

Model architecture, rewards, hyperparameters, training code, evaluation code, etc. can be added, modified, or removed freely; anything outside the constraints above can be changed.

## History

Read the Markdown files in `history/` to see what previous iterations did and their results. Use that context to make changes that increase the score: avg length / avg steps.

Each iteration produce two files in `history/`:

`plan<n>.md`, the plan for iteration n: steps to be taken, low and high level reasoning, parameters and code to be changed, etc.

`out<n>.md`, the results of iteration n: model performance, relevant logs, best model score, when the best model was found during training, test results, etc.

## Steps per iteration

Read the codebase and `history/` to understand what previous iterations did.
Write `plan<n>.md` in `history/`.
Implement the plan.
Train, run, and test the model. Fix any errors that prevent this.
Copy the best model found during training to `best/iteration<n>.zip`.
Write the results in `out<n>.md` in `history/`.
Once you are satisfied and have written your final report in `out<n>.md`, you cannot go back and redo the experiment. Your iteration is finished; leave future changes to a future iteration.
