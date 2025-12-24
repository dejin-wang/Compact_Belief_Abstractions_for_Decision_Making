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
        self.layer4 = nn.Linear(num_neros, num_neros)
        self.layer5 = nn.Linear(num_neros, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        return self.layer5(x)


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



def measurement_model(state):
    Fai = [0, 1, 2]  # G Y R
    d, v, fai, tfai = state.numpy()[0]
    nd = np.random.normal(0, 1)
    nv = np.random.normal(0, 0.1)
    nt= np.random.normal(0, 0.1)
    sample = np.random.random()
    if sample <0.94:
        mfai = fai
    else:
        mfai = random.choice(Fai)
    return [d+nd, v+nv, mfai,tfai+nt]

def particle_init(num_particles, measurement):
    d_m, v_m, fai_m, tfai_m = measurement
    Fai = [0, 1, 2]  # G Y R
    TFai = [10, 5, 10]  # G Y R
    probabilities = [0.4, 0.2, 0.4]
    particles = []

    for i in range(num_particles):
        fai_0 = random.choices(Fai, weights=probabilities, k=1)[0]
        state = [random.uniform(110, 120), random.uniform(5, 15), fai_0, random.uniform(0, TFai[fai_0])]
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        particles.append(state)
    weights = np.full(num_particles, 1 / num_particles)
    return weights, particles


def particle_filter(num_particles, measurement, particles, weights, action_index_est):
    l = len(particles)
    for i in range(l):
        particles[i] = get_next_state(particles[i], action_index_est)

        d_m, v_m, fai_m, tfai_m = measurement
        d, v, fai, tfai = particles[i].numpy()[0]

        likelihood1 = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * 1 * (d_m - d) ** 2)
        likelihood2 = (10 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * 100 * (v_m - v) ** 2)
        if fai == fai_m:
            likelihood3 = 0.96
        else:
            likelihood3 = 0.04
        likelihood4 = (10 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * 100 * (tfai_m - tfai) ** 2)

        likelihood = likelihood1 * likelihood2 * likelihood3 * likelihood4

        weights[i] = weights[i] * likelihood

    weights += 1e-300  # division by zero

    weights /= np.sum(weights)
    indices = np.random.choice(np.arange(num_particles), size=num_particles, replace=True, p=weights)
    re_particles_t = []
    for i in range(l):
        re_particles_t.append(particles[indices[i]])
    re_weights_t = np.full(num_particles, 1 / num_particles)

    return re_particles_t, re_weights_t


def get_estimated_state(weights, particles):
    l = len(particles)
    # estimation
    d_est = 0
    v_est = 0
    fai_0 = 0
    fai_1 = 0
    fai_2 = 0
    # d_est, v_est
    for i in range(l):
        d, v, fai, tfai = particles[i].numpy()[0]
        d_est = d_est + d * weights[i]
        v_est = v_est + v * weights[i]
        if fai == 0:
            fai_0 = fai_0 + weights[i]
        elif fai == 1:
            fai_1 = fai_1 + weights[i]
        else:
            fai_2 = fai_2 + weights[i]

    # fai_est
    array = np.array([fai_0, fai_1, fai_2])
    fai_est = np.argmax(array)

    # tfai_est
    tfai_est = 0
    for i in range(l):
        if fai_est == particles[i].numpy()[0][2]:
            tfai_est = tfai_est + particles[i].numpy()[0][3] * weights[i]

    estimated_state = [d_est, v_est, fai_est, tfai_est]
    estimated_state = torch.tensor(estimated_state, dtype=torch.float32, device=device).unsqueeze(0)

    return estimated_state

def compute_moments(particles, weights):
    """
    计算粒子的高阶矩
    :param particles: 粒子列表，每个粒子是一个张量
    :param weights: 权重数组，形状为 (N,)
    :return: 一阶矩（均值），协方差矩阵，三阶矩（偏度）
    """
    weights = np.array(weights)  # 确保权重是 numpy 数组
    N = len(particles)
    D = particles[0].shape[1]  # 每个粒子的维度

    # 将所有粒子转换为 numpy 数组
    particles_array = np.array([particle.numpy().flatten() for particle in particles])


    # 计算一阶矩（均值）
    mean = np.sum(particles_array * weights[:, np.newaxis], axis=0)
    mean1 = get_estimated_state(weights, particles)

    # 计算协方差矩阵
    cov = np.zeros((D, D))
    for i in range(N):
        diff = particles_array[i] - mean
        cov += weights[i] * np.outer(diff, diff)

    # 计算三阶矩（偏度）
    skew = np.zeros(D)
    for i in range(N):
        diff = particles_array[i] - mean
        skew += weights[i] * diff ** 3

    return mean, cov, skew


def get_constructed_state(mean, cov, skew):

    # 获取协方差矩阵右上角的所有元素（包括对角线）
    cov_upper_right = cov[np.triu_indices(4)]
    # 组合成一个向量
    constructed_state = np.concatenate((mean, cov_upper_right, skew))
    return torch.tensor([constructed_state], device=device, dtype=torch.float32)


if __name__ == '__main__':
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 128
    GAMMA = 0.99
    TAU = 8e-6
    LR = 5e-6
    num_particles = 3000
    # mean_reward_dict
    mean_reward_dict = {}
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
    n_observations = 18


    # define the Q-net
    policy_net = DQN(n_observations, n_actions).to(device)
    policy_net.load_state_dict(torch.load('dqn_model_m4.pth', map_location=device))
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)

    if torch.cuda.is_available():
        num_episodes = 6500
    else:
        num_episodes =10000

    for i_episode in range(num_episodes):

        # initial reward
        episode_reward = 0
        # Initialize the environment and get its state

        # initial state
        fai_0 = random.choices(Fai, weights=probabilities, k=1)[0]
        state = [random.uniform(110, 120), random.uniform(5, 15), fai_0, random.uniform(0, TFai[fai_0])]
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        measurement = measurement_model(state)
        re_weights, re_particles = particle_init(num_particles, measurement)
        estimated_state = get_estimated_state(re_weights, re_particles)
        mean, var, skew = compute_moments(re_particles, re_weights)
        constructed_state = get_constructed_state(mean, var, skew)

        # initial a = 0
        action_index_before = 3
        action_index_before = torch.tensor([[action_index_before]], device=device, dtype=torch.long)

        for t in count():
            action_index = select_action_index(constructed_state)
            done = (state.numpy()[0][0] < 0) or (t > 1000)
            if state.numpy()[0][0] < 0:
                next_state = None
                reward = torch.tensor([0], device=device)
                constructed_next_state = None
            else:
                next_state = get_next_state(state, action_index)
                reward = torch.tensor([get_reward(next_state, action_index_before, action_index)], device=device)
                action_index_est = action_index
                measurement = measurement_model(next_state)
                re_particles, re_weights = particle_filter(num_particles, measurement, re_particles, re_weights, action_index_est)
                mean, var, skew = compute_moments(re_particles, re_weights)
                constructed_next_state = get_constructed_state(mean, var, skew)
                #print(next_state.numpy()[0])
                #print(constructed_next_state.numpy()[0][0:4])

            # Store the transition in memory
            memory.push(constructed_state, action_index,  constructed_next_state, reward)

            # Move to the next state, store the action_before
            state = next_state
            action_index_before = action_index
            constructed_state = constructed_next_state


            # print(measurement)
            # Perform one step of the optimization (on the policy network)
            optimize_model()

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
        if i_episode % 10 == 0:
            torch.save(policy_net.state_dict(), 'dqn_model_m4.pth')
        print(f"Episode {i_episode + 1}/{num_episodes}, Reward: {episode_reward}")

