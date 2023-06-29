import gymnasium as gym
import matplotlib.pyplot as plt
from os import environ


# pravimo okruženje
env = gym.make("CarRacing-v2", domain_randomize=True, render_mode="human")
# env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()