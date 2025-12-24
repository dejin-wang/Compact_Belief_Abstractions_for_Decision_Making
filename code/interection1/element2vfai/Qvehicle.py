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
    EPS_START = 0.9
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


def get_next_state(state , action_index):
    possible_actions = [-3, -2, -1, 0, 1, 2, 3]
    action = possible_actions[action_index.item()]
    Fai = [0, 1, 2]  # G Y R
    TFai = [10, 5, 10]  # G Y R
    DT = 1
    # noise
    nd = np.random.normal(0, 0.4)
    nv = np.random.normal(0, 0.1)
    nfai = np.random.normal(0, 0.01)
    d, v, fai, tfai = state.numpy()[0]
    # update state

    next_v = v + action * DT + nv
    if next_v <0:
        next_v=0
        next_d = d - 0.5 * (-v) * DT * DT - v * DT + nd
    else:
        next_d = d - 0.5 * action * DT * DT - v * DT + nd
    if tfai + DT <= TFai[int(fai)] + nfai:
        next_fai = fai
        next_tfai = tfai + DT
    else:
        next_fai = (fai + 1) % len(Fai)
        next_tfai = np.random.uniform(0, 1)
    next_state = [next_d, next_v, next_fai, next_tfai]
    return torch.tensor([next_state], device=device, dtype=torch.float32)


def indicator_function(condition):
    return 1 if condition else 0




def get_reward(next_state, action_index_before, action_index):
    possible_actions = [-3, -2, -1, 0, 1, 2, 3]
    action = possible_actions[action_index.item()]
    action_before = possible_actions[action_index_before.item()]
    d_critical = 10
    thr_smooth = 1
    v_max=15
    v_min=5

    next_d, next_v, next_fai, next_tfai = next_state.numpy()[0]

    R_term = 100 * indicator_function((next_d <= 0) and ((next_fai == 0) or (next_fai == 1)))
    R_RRl = -200 * indicator_function((next_d <= 0) and (next_fai == 2))
    R_vel= -100 * indicator_function((next_v > v_max) or ((next_v < v_min) and (next_d > d_critical)))
    R_smooth = -10 * indicator_function(abs(action-action_before) > thr_smooth)

    R_act = -2

    return R_term + R_RRl + R_vel + R_smooth + R_act


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
        loss = 0
        return loss
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

    return loss



def get_mean_reward():
    sum_rewards_100 = 0
    for i in range(100):
        # initial a = 0
        action_index_before = 3
        action_index_before = torch.tensor([[action_index_before]], device=device, dtype=torch.long)
        # initial state
        fai_0 = random.choices(Fai, weights=probabilities, k=1)[0]
        state = [random.uniform(80, 120), random.uniform(5, 15), fai_0, random.uniform(0, TFai[fai_0])]
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        policy_net.load_state_dict(torch.load('dqn_model.pth', map_location=device))
        policy_net.eval()
        # initial reward
        episode_reward = 0

        for t in count():
            action_index = policy_net(state).max(1).indices.view(1, 1)
            done = (state.cpu().numpy()[0][0] < 0) or (t > 50)
            if state.cpu().numpy()[0][0] < 0:
                next_state = None
                reward = torch.tensor([0], device=device)
            else:
                next_state = get_next_state(state, action_index)
                reward = torch.tensor([get_reward(next_state, action_index_before, action_index)], device=device)

            state = next_state
            action_index_before = action_index
            episode_reward += reward.item()
            if done:
                break
        sum_rewards_100 = sum_rewards_100 + episode_reward
    return sum_rewards_100/100


if __name__ == '__main__':
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 128
    GAMMA = 0.9
    TAU = 8e-4
    LR = 5e-4
    # mean_reward_dict
    mean_reward_dict = {}
    mean_loss_dict = {}
    # store eposide reward
    episode_rewards = []
    # possible actions
    possible_actions = [-3, -2, -1, 0, 1, 2, 3]
    action_indexes = [0, 1, 2, 3, 4, 5, 6]
    # denote green red yellow light
    Fai = [0, 1, 2]  # G Y R
    TFai = [10, 5, 10]  # G Y R
    # the probabilities of different lights
    probabilities = [0.4, 0.2, 0.4]
    n_actions = len(possible_actions)
    # initial a
    action_index_before = random.choice(action_indexes)
    action_index_before = torch.tensor([[action_index_before]], device=device, dtype=torch.long)

    # initial state
    fai_0 = random.choices(Fai, weights=probabilities, k=1)[0]
    state = [random.uniform(80, 120), random.uniform(5, 15), fai_0, random.uniform(0, TFai[fai_0])]
    n_observations = len(state)
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    # define the Q-net
    policy_net = DQN(n_observations, n_actions).to(device)
    #policy_net.load_state_dict(torch.load('dqn_model.pth', map_location=device))
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)

    if torch.cuda.is_available():
        num_episodes =3000
    else:
        num_episodes =3000

    for i_episode in range(num_episodes):
        i_loss = 0
        loss_total = 0

        # initial reward
        episode_reward = 0
        # Initialize the environment and get its state

        # initial state
        fai_0 = random.choices(Fai, weights=probabilities, k=1)[0]
        state = [random.uniform(80, 120), random.uniform(5, 15), fai_0, random.uniform(0, TFai[fai_0])]
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        # initial a = 0
        action_index_before = 3
        action_index_before = torch.tensor([[action_index_before]], device=device, dtype=torch.long)

        for t in count():
            action_index = select_action_index(state)
            done = (state.cpu().numpy()[0][0] < 0) or (t > 1000)
            if state.cpu().numpy()[0][0] < 0:
                next_state = None
                reward = torch.tensor([0], device=device)
            else:
                next_state = get_next_state(state, action_index)
                reward = torch.tensor([get_reward(next_state, action_index_before, action_index)], device=device)

            # Store the transition in memory
            memory.push(state, action_index, next_state, reward)

            # Move to the next state, store the action_before
            state = next_state
            action_index_before = action_index

            # Perform one step of the optimization (on the policy network)
            loss = optimize_model()

            if loss != 0:
                loss_total = loss_total + loss
                i_loss = i_loss + 1



            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                break
            episode_reward += reward.item()

        if i_episode % 50 == 0:
            torch.save(policy_net.state_dict(), 'dqn_model.pth')
            #mean_reward_dict[i_episode] = {get_mean_reward()}
            if i_loss != 0:
                mean_loss_dict[i_episode] = {loss_total / i_loss}
        print(f"Episode {i_episode + 1}/{num_episodes}, Reward: {episode_reward}")




    # # write CSV
    # with open('mean_rewards_data.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(['Episode Number', 'Mean Reward'])
    #     for key, value in mean_reward_dict.items():
    #         writer.writerow([key, list(value)[0]])
    #
    # # write CSV
    # with open('mean_loss_data.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(['Episode Number', 'Mean Loss'])
    #     for key, value in mean_loss_dict.items():
    #         writer.writerow([key, list(value)[0]])
