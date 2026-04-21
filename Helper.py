import gymnasium as gym
import pygame
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

def evaluate(model, env):
    eval_env = Monitor(gym.make(env, render_mode='rgb_array'))
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

def render_callback(model, env):
    env = gym.make(env, render_mode="human")
    obs, _ = env.reset()

    running = True
    for _ in range(1000):
        if not running:
            break

        # Handle window close
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            obs, _ = env.reset()

    env.close()
    pygame.quit()