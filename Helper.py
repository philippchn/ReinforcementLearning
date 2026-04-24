import gymnasium as gym
import pygame
from matplotlib import pyplot as plt
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

def plot_reward(rewards):
    window = 50
    smoothed = rewards.rolling(window).mean()

    plt.figure()
    plt.title("Rewards History (Smoothed)")
    plt.plot(rewards, alpha=0.3, label="Raw")
    plt.plot(smoothed, label=f"{window}-episode avg")
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()