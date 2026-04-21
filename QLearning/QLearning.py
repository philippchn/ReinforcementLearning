import random
import gymnasium as gym
import numpy as np
import pygame
from tqdm import tqdm


def initialize_q_table(state_space, action_space):
  return np.zeros((state_space, action_space))

def greedy_policy(QTable, state):
    return np.argmax(QTable[state][:])

def epsilon_greedy_policy(qtable, state, epsilon, env):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return greedy_policy(qtable, state)

def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, learning_rate, gamma, qtable):
  for episode in tqdm(range(n_training_episodes)):
    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
    state, info = env.reset()
    step = 0
    terminated = False
    truncated = False

    for step in range(max_steps):
      action = epsilon_greedy_policy(qtable, state, epsilon, env)

      new_state, reward, terminated, truncated, info = env.step(action)

      # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
      qtable[state][action] = qtable[state][action] + learning_rate * (reward + gamma * np.max(qtable[new_state]) - qtable[state][action])

      if terminated or truncated:
        break

      state = new_state
  return qtable

def evaluate_agent(env, max_steps, n_eval_episodes, Q, seed):
  """
  Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
  :param env: The evaluation environment
  :param max_steps: Maximum number of steps per episode
  :param n_eval_episodes: Number of episode to evaluate the agent
  :param Q: The Q-table
  :param seed: The evaluation seed array (for taxi-v3)
  """
  episode_rewards = []
  for episode in tqdm(range(n_eval_episodes)):
    if seed:
      state, info = env.reset(seed=seed[episode])
    else:
      state, info = env.reset()
    step = 0
    truncated = False
    terminated = False
    total_rewards_ep = 0

    for step in range(max_steps):
      # Take the action (index) that have the maximum expected future reward given that state
      action = greedy_policy(Q, state)
      new_state, reward, terminated, truncated, info = env.step(action)
      total_rewards_ep += reward

      if terminated or truncated:
        break
      state = new_state
    episode_rewards.append(total_rewards_ep)
  mean_reward = np.mean(episode_rewards)
  std_reward = np.std(episode_rewards)

  return mean_reward, std_reward

def render_callback(qtable, env):
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

        action = np.argmax(qtable[obs])
        obs, reward, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            obs, _ = env.reset()

    env.close()
    pygame.quit()