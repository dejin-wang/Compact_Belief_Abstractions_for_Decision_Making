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
    # obtain possible actions
    global steps_done
    EPS_START = 0.05
    EPS_END = 0.05
    EPS_DECAY = 1000
    action_indexes = [0, 1, 2, 3, 4, 5, 6]
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)  # 1*1
    else:
        return torch.tensor([[random.choice(action_indexes)]], device=device, dtype=torch.long)

def indicator_function(condition):
    return 1 if condition else 0


def get_reward(next_state, action_index_before, action_index):
    possible_actions = [-3, -2, -1, 0, 1, 2, 3]
    action = possible_actions[action_index.item()]
    action_before = possible_actions[action_index_before.item()]
    d_safe = 10
    d_large =15
    thr_smooth = 1
    v_max = 30
    v_min = 25
    k=-1

    next_d, next_v, next_v_lead, next_a_lead= next_state.numpy()[0]

    R_danger = -100 * indicator_function(next_d < d_safe)
    #R_large = -50 * indicator_function(next_d > d_large)
    R_smooth = -1 * indicator_function(abs(action - action_before) > thr_smooth)
    R_efficiency = -1*(indicator_function(next_v > v_max)* abs(next_v-v_max)+indicator_function(next_v < v_min)* abs(next_v-v_min))


    return R_danger + R_smooth  + R_efficiency

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
    next_d = d - 0.5 * (action-a_lead) * DT * DT - (v-v_lead) * DT + nd
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


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

if __name__ == '__main__':
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 128
    GAMMA = 0.9
    TAU = 8e-5
    LR = 5e-5
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
    state = [random.uniform(10, 15), random.uniform(20, 30), random.uniform(20, 30), random.uniform(-3, 3) ]
    n_observations = len(state)
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    # define the Q-net
    policy_net = DQN(n_observations, n_actions).to(device)
    policy_net.load_state_dict(torch.load('dqn_model.pth', map_location=device))
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)

    if torch.cuda.is_available():
        num_episodes =500
    else:
        num_episodes =5000

    for i_episode in range(num_episodes):

        # initial reward
        episode_reward = 0
        # Initialize the environment and get its state

        # initial state
        state = [random.uniform(10, 15), random.uniform(20, 30), random.uniform(20, 30), random.uniform(-3, 3) ]
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        # initial a = 0
        action_index_before = 3
        action_index_before = torch.tensor([[action_index_before]], device=device, dtype=torch.long)

        for t in range(30):
            action_index = select_action_index(state)

            next_state = get_next_state(state, action_index)
            reward = torch.tensor([get_reward(next_state, action_index_before, action_index)], device=device)
            #print(reward)

            # Store the transition in memory
            memory.push(state, action_index, next_state, reward)

            # Move to the next state, store the action_before
            state = next_state
            action_index_before = action_index

            # Perform one step of the optimization (on the policy network)
            optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)
            episode_reward += reward.item()



        if i_episode % 10 == 0:
            torch.save(policy_net.state_dict(), 'dqn_model1.pth')
        print(f"Episode {i_episode + 1}/{num_episodes}, Reward: {episode_reward}")


