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


def select_action_index(state):
    # obtain possible actions
    action_indexes = [0, 1, 2, 3, 4, 5, 6]
    return policy_net(state).max(1).indices.view(1, 1)  # 1*1

def indicator_function(condition):
    return 1 if condition else 0


def get_reward(next_state, action_index_before, action_index):
    possible_actions = [-3, -2, -1, 0, 1, 2, 3]
    action = possible_actions[action_index.item()]
    action_before = possible_actions[action_index_before.item()]
    d_safe = 10
    thr_smooth = 1
    v_max = 30
    v_min = 25
    k=-1

    next_d, next_v, next_v_lead, next_a_lead= next_state.numpy()[0]

    R_danger = -100 * indicator_function(next_d < d_safe)
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

def measurement_model(state):
    d, v, v_lead, a_lead = state.numpy()[0]
    nd = np.random.normal(0, 0.2)
    nv = np.random.normal(0, 0.2)
    nv_lead= np.random.normal(0, 0.2)
    na_lead = np.random.normal(0, 0.2)
    return [d+nd, v+nv, v_lead+nv_lead, a_lead+na_lead]

# 初始化粒子和权重
def particle_init(num_particles):
    particles = []
    for i in range(num_particles):
        state = [random.uniform(10, 15), random.uniform(20, 30), random.uniform(20, 30), random.uniform(-3, 3)]
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        particles.append(state)
    weights = np.full(num_particles, 1/num_particles)
    return weights, particles


def systematic_resample(weights):
    N = len(weights)
    positions = (np.arange(N) + np.random.uniform(0, 1)) / N
    indexes = np.zeros(N, dtype=np.int32)

    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes

# 定义粒子滤波函数
def is_multiple(x, y):
    if x == 0:
        return False  # 避免除以0的错误
    return y % x == 0


# 定义粒子滤波函数

def particle_filter(num_particles, measurement, particles, weights, action_index_est,t):
    l = len(particles)
    if is_multiple(ts, t):
        for i in range(l):
            particles[i] = get_next_state(particles[i], action_index_est)
            d_m, v_m, v_lead_m, a_lead_m = measurement
            d, v, v_lead, a_lead = particles[i].numpy()[0]
            likelihood1 = (5 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * 25 * (d_m - d) ** 2)
            likelihood2 = (5 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * 25 * (v_m - v) ** 2)
            likelihood3 = (5 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * 25 * (v_lead_m - v_lead) ** 2)
            likelihood4 = (5 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * 25 * (a_lead_m - a_lead) ** 2)
            likelihood = likelihood1 * likelihood2 * likelihood3 * likelihood4
            weights[i] = weights[i] * likelihood
        weights = np.array(weights)  # Ensure weights are a numpy array
        weights += 1.e-300  # Avoid division by zero
        weights /= np.sum(weights)  # Normalize
        # 系统重采样
        indices = systematic_resample(weights)
        particles = [particles[i] for i in indices]
    else:
        for i in range(l):
            particles[i] = get_next_state(particles[i], action_index_est)
        weights = np.ones(num_particles) / num_particles  # 重采样后的权重初始化为均匀分布
    return particles, weights

# 计算估计的矩
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
    GAMMA = 0.9
    TAU = 8e-5
    LR = 5e-5
    ts=3
    num_particles = 1000
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
    n_observations = 18


    # define the Q-net
    policy_net = DQN(n_observations, n_actions).to(device)
    policy_net.load_state_dict(torch.load('dqn_model_m3.pth', map_location=device))


    if torch.cuda.is_available():
        num_episodes =50
    else:
        num_episodes =100
    reward_set = []

    for i_episode in range(num_episodes):
        episode_reward = 0
        episode_reward_set = []

        # Initialize the environment and get its state
        # initial a = 0
        action_index_before = 3
        action_index_before = torch.tensor([[action_index_before]], device=device, dtype=torch.long)

        # initial state
        random.seed(i_episode)
        v0 = random.uniform(20, 30)
        state = [random.uniform(10, 15), v0, random.uniform(v0, 30), random.uniform(-3, 3)]
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        #print(state)
        measurement = measurement_model(state)
        weights, particles = particle_init(num_particles)
        particles, weights = particle_filter(num_particles, measurement, particles, weights, action_index_before,ts)
        mean, cov, skew =  compute_moments(particles, weights)
        constructed_state = get_constructed_state(mean, cov, skew)


        for t in range(20):
            episode_reward_set.append(episode_reward)
            action_index = select_action_index(constructed_state)

            next_state = get_next_state(state, action_index)
            reward = torch.tensor([get_reward(next_state, action_index_before, action_index)], device=device)

            # particle information
            measurement = measurement_model(next_state)
            particles, weights = particle_filter(num_particles, measurement, particles, weights, action_index,t)
            mean, cov, skew = compute_moments(particles, weights)
            constructed_next_state = get_constructed_state(mean, cov, skew)


            # Move to the next state, store the action_before
            state = next_state
            action_index_before = action_index
            constructed_state =  constructed_next_state
            episode_reward += reward.item()

            # print(state)
            # print(constructed_next_state.numpy()[0][0:4])
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
with open('DQN_M_reward_testt34.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for episode_index, episode in enumerate(reward_set):
        # 为每个 episode 写入一个标题行，标识 episode 的索引
        writer.writerow(['Episode ' + str(episode_index)])
        for reward in episode:
            # 将 NumPy 数组转换为列表，并写入 CSV 文件
            writer.writerow([reward])




