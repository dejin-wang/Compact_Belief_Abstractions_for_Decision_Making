# -*- coding: utf-8 -*-
import math
import csv
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


steps_done = 0


def select_action_index(state):
    return policy_net(state).max(1).indices.view(1, 1)  # 1*1


def indicator_function(condition):
    return 1 if condition else 0


def get_reward(next_state, action_index_before, action_index):
    possible_actions = [-3, -2, -1, 0, 1, 2, 3]
    action = possible_actions[action_index.item()]
    action_before = possible_actions[action_index_before.item()]
    d_safe = 10
    d_large = 15
    thr_smooth = 1
    v_max = 30
    v_min = 25
    k=-1

    next_d, next_v, next_v_lead, next_a_lead= next_state.numpy()[0]

    R_danger = -100 * indicator_function(next_d < d_safe)
    #R_large = -50 * indicator_function(next_d > d_large)
    R_smooth = -1 * indicator_function(abs(action - action_before) > thr_smooth)
    R_efficiency = -1*(indicator_function(next_v > v_max)* abs(next_v-v_max)+indicator_function(next_v < v_min)* abs(next_v-v_min))


    return R_danger + R_smooth + R_efficiency


def get_next_state(state , action_index):
    possible_actions = [-3, -2, -1, 0, 1, 2, 3]
    action = possible_actions[action_index.item()]
    DT = 1
    # noise
    nd = np.random.normal(0, 0.5)
    nv = np.random.normal(0, 0.2)
    nv_lead = np.random.normal(0, 0.2)
    na_lead = np.random.normal(0, 0.1)
    v_desired = 25

    d, v, v_lead, a_lead = state.numpy()[0]
    # update state

    next_v = v + action * DT + nv
    next_d = d - 0.5 * (v-v_lead) * DT * DT - (v-v_lead) * DT + nd
    next_v_lead = v_lead + a_lead * DT + nv_lead
    next_a_lead = 0.5*(v_desired-(v_lead + a_lead * DT))+na_lead

    next_state = [next_d, next_v, next_v_lead, next_a_lead]
    return torch.tensor([next_state], device=device, dtype=torch.float32)


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


# experience replay buffer
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


# The structure of DQN
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        num_neros= 512
        self.layer1 = nn.Linear(n_observations, num_neros)
        self.layer2 = nn.Linear(num_neros, num_neros)
        self.layer3 = nn.Linear(num_neros, num_neros)
        self.layer4 = nn.Linear(num_neros, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)



if __name__ == '__main__':
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 128
    GAMMA = 0.9
    TAU = 8e-6
    LR = 5e-6
    # mean_reward_dict
    mean_reward_dict = {}
    # store eposide reward
    episode_rewards = []
    # possible actions
    possible_actions = [-3, -2, -1, 0, 1, 2, 3]
    action_indexes = [0, 1, 2, 3, 4, 5, 6]
    n_actions = len(possible_actions)
    # initial a
    action_index_before = random.choice(action_indexes)
    action_index_before = torch.tensor([[action_index_before]], device=device, dtype=torch.long)

    # initial state
    v0= random.uniform(20, 30)
    state = [random.uniform(10, 15),v0 , random.uniform(v0, 30), random.uniform(-3, 3) ]
    n_observations = len(state)
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    # define the Q-net
    policy_net = DQN(n_observations, n_actions).to(device)
    policy_net.load_state_dict(torch.load('dqn_model.pth', map_location=device))
    policy_net.eval()


    if torch.cuda.is_available():
        num_episodes =100
    else:
        num_episodes =100
    reward_set = []
    for i_episode in range(num_episodes):

        # initial reward
        episode_reward = 0
        episode_reward_set = []
        # Initialize the environment and get its state

        # initial state
        random.seed(i_episode)
        v0 = random.uniform(20, 30)
        state = [random.uniform(10, 15), v0, random.uniform(v0, 30), random.uniform(-3, 3)]
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        #print(state)

        # initial a = 0
        action_index_before = 3
        action_index_before = torch.tensor([[action_index_before]], device=device, dtype=torch.long)

        for t in range(20):
            episode_reward_set.append(episode_reward)
            action_index = select_action_index(state)
            #print(action_index)
            next_state = get_next_state(state, action_index)
            reward = torch.tensor([get_reward(next_state, action_index_before, action_index)], device=device)
            # Move to the next state, store the action_before
            state = next_state
            action_index_before = action_index
            #print(reward)
            #print(state)
            episode_reward += reward.item()

        print(f"Episode {i_episode + 1}/{num_episodes}, Reward: {episode_reward}")
        reward_set.append(episode_reward_set)

longest_array = max(reward_set, key=len)

for i in range(num_episodes):
    if len(reward_set[i])< max(len(longest_array),30):
        l= max(len(longest_array),30)-len(reward_set[i])
        for j in range (l):
            reward_set[i].append(reward_set[i][-1])

print(reward_set[0])
import csv
with open('DQN_reward_test.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for episode_index, episode in enumerate(reward_set):
        # 为每个 episode 写入一个标题行，标识 episode 的索引
        writer.writerow(['Episode ' + str(episode_index)])
        for reward in episode:
            # 将 NumPy 数组转换为列表，并写入 CSV 文件
            writer.writerow([reward])










