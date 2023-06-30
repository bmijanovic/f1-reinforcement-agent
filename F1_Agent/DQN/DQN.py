import cv2
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import transforms

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(input_shape), 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def feature_size(self, input_shape):
        return self.features(torch.zeros(1, *input_shape)).view(1, -1).size(1)

    def save(self, path):
        torch.save(self, path)

    @classmethod
    def load(cls, path):
        return torch.load(path)


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            nesto = policy_net(state)
            max_value, max_index = torch.max(nesto.view(5), dim=0)
            print(max_index, eps_threshold)
            return torch.tensor([[max_index]], dtype=torch.long)
    else:
        return torch.tensor([[env.action_space.sample()]], dtype=torch.long)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]

    expected_state_action_values = (next_state_values.detach() * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def process_state_image(state):
    transform = transforms.Grayscale()
    monochrome_tensor = transform(state.permute(2, 0, 1)).squeeze()
    # monochrome_tensor /= 255
    return monochrome_tensor


BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 7000
TAU = 0.005
LR = 1e-4

env = gym.make("CarRacing-v2", domain_randomize=False, render_mode="human", continuous=False)
# env = gym.make("CartPole-v1")

plt.ion()

n_actions = env.action_space.n
state, info = env.reset()
state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).view(96, 96, 3)
state = state.permute(2, 0, 1)

try:
    policy_net = DQN(state.shape, n_actions)
    target_net = DQN(state.shape, n_actions)
    policy_net.load("policynet.pt")
    policy_net.load("targetnet.pt")
except:
    policy_net = DQN(state.shape, n_actions)
    target_net = DQN(state.shape, n_actions)

target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

num_episodes = 600
episode_durations = []

try:
    for i_episode in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        state = state.permute(2, 0, 1).unsqueeze(0)
        negative_reward_counter = 0
        time_frame_counter = 1
        for t in count():
            action = select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward])
            done = terminated or truncated

            negative_reward_counter = negative_reward_counter + 1 if time_frame_counter > 100 and reward < 0 else 0

            if action.item() == 3:
                reward += 0.5

            if action.item() == 0:
                reward -= 0.5

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32)
                next_state = next_state.permute(2, 0, 1).unsqueeze(0)

            memory.push(state, action, next_state, reward)

            state = next_state

            optimize_model()
            time_frame_counter += 1
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)
            if done:
                episode_durations.append(t + 1)
                break
except:
    target_net.save("targetnet.pt")
    policy_net.save("policynet.pt")

plt.show()
target_net.save("targetnet.pt")
policy_net.save("policynet.pt")
