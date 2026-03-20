import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from snake_env import SnakeEnv

eval_env = SnakeEnv()
env = SnakeEnv()

# default params for evalcallback from docs, changes to eval_freq
eval_callback = EvalCallback(eval_env,
                             callback_on_new_best=None, 
                             callback_after_eval=None, 
                             n_eval_episodes=5, 
                             eval_freq=1000, 
                             log_path=None, 
                             best_model_save_path="eval", 
                             deterministic=True, 
                             render=False, 
                             verbose=1,
                             warn=True
                             )

# default params for dqn from docs, chnages to target_update_interval and verbose
model = DQN("MlpPolicy", 
            env, 
            learning_rate=0.0001, 
            buffer_size=1000000, 
            learning_starts=100, 
            batch_size=32, 
            tau=1.0, 
            gamma=0.99, 
            train_freq=4, 
            gradient_steps=1, 
            replay_buffer_class=None, 
            replay_buffer_kwargs=None, 
            optimize_memory_usage=False, 
            n_steps=1, 
            target_update_interval=1000, 
            exploration_fraction=0.1, 
            exploration_initial_eps=1.0, 
            exploration_final_eps=0.05, 
            max_grad_norm=10, 
            stats_window_size=100, 
            tensorboard_log=None, 
            policy_kwargs=None, 
            verbose=1, 
            seed=None, 
            device='auto', 
            _init_setup_model=True
            )

print("train start...")
print()
model.learn(10000, callback=eval_callback)

eval_env.close()
env.close()

best_model = DQN.load("eval/best_model")
test_env = SnakeEnv()

# benchmark uses avg scores over 1000 games
snake_len = []
steps = []
print("test start...")
print()
for i in range(1000):
    print(f"game {i}")
    obs, info = test_env.reset()
    done = False
    while not done:
        action, _states = best_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated
    snake_len.append(info["snake_len"])
    steps.append(info["steps"])
    print(f"snake_len: {info["snake_len"]}")
    print(f"steps: {info["steps"]}")

print()
print("test results over 1000 games:")
print(f"avg snake length: {np.mean(snake_len)}")
print(f"avg steps: {np.mean(steps)}")

test_env.close()