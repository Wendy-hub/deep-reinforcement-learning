import gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
import os

environment_name = "Breakout-v0"
env = gym.make(environment_name, render_mode='human')

episodes = 5
for episode in range(1, episodes + 1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)  # old api
        score += reward
        print(info)
    print('Episode:{} Score:{}'.format(episode, score))

# env.close()
