import gymnasium as gym
import pygame
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

def evaluate_atari(model, env_id, n_stack=4):
    eval_env = DummyVecEnv([make_env_atari(env_id)])
    eval_env = VecFrameStack(eval_env, n_stack)

    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)

    eval_env.close()
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

def make_env_atari(env_id, render_mode=None):
    def _init():
        env = gym.make(env_id, render_mode=render_mode)
        env = AtariWrapper(env)
        return env
    return _init

def render_callback_atari(model, env, n_stack=4):
    env = DummyVecEnv([make_env_atari(env, render_mode="human")])
    env = VecFrameStack(env, n_stack)
    obs = env.reset()

    running = True

    for _ in range(1000):
        if not running:
            break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = env.step(action)

        if dones[0]:
            obs = env.reset()

    env.close()
    pygame.quit()