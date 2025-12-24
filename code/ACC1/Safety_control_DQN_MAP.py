# -*- coding: utf-8 -*-
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义系统重采样函数
def residual_resample(weights):
    N = len(weights)
    indexes = np.zeros(N, dtype=int)

    # Step 1: Determine the number of copies for each particle
    num_copies = (N * weights).astype(int)
    k = 0
    for i in range(N):
        for _ in range(num_copies[i]):
            indexes[k] = i
            k += 1

    # Step 2: Calculate the residual weights
    residual = weights - num_copies / N
    residual /= residual.sum()

    # Step 3: Perform systematic resampling on the residuals
    cumulative_sum = np.cumsum(residual)
    cumulative_sum[-1] = 1.0  # Ensure the last value is exactly 1.0
    positions = (np.arange(N - k) + np.random.uniform(0, 1)) / (N - k)

    i, j = 0, 0
    while i < len(positions):
        if positions[i] < cumulative_sum[j]:
            indexes[k] = j
            k += 1
            i += 1
        else:
            j += 1

    return indexes


def effective_sample_size(weights):
    return 1.0 / np.sum(np.square(weights))


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
def particle_filter(num_particles, measurement, particles, weights, action_index_est, threshold=0.5):
    l = len(particles)
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

    # 计算有效粒子数
    ess = effective_sample_size(weights)

    # 自适应重采样
    if ess < threshold * num_particles:
        indices = residual_resample(weights)
        particles = [particles[i] for i in indices]
        weights = np.ones(num_particles) / num_particles  # 重采样后的权重初始化为均匀分布

    return particles, weights

# 计算估计的状态
def get_estimated_state(weights, particles):
    l = len(particles)
    d_est, v_est, v_lead_est, a_lead_est = 0, 0, 0, 0
    for i in range(l):
        d, v, v_lead, a_lead = particles[i].numpy()[0]
        v_lead_est += v_lead * weights[i]
        a_lead_est += a_lead * weights[i]
        d_est += d * weights[i]
        v_est += v * weights[i]
    estimated_state = [d_est, v_est, v_lead_est, a_lead_est]
    estimated_state = torch.tensor(estimated_state, dtype=torch.float32, device=device).unsqueeze(0)
    return estimated_state

# 定义状态转移模型
def get_next_state(state, action_index):
    possible_actions = [-3, -2, -1, 0, 1, 2, 3]
    action = possible_actions[action_index.item()]
    DT = 1
    nd = np.random.normal(0, 0.5)
    nv = np.random.normal(0, 0.2)
    nv_lead = np.random.normal(0, 0.2)
    na_lead = np.random.normal(0, 0.1)
    v_desired = 25

    d, v, v_lead, a_lead = state.numpy()[0]
    next_v = v + action * DT + nv
    next_d = d - 0.5 * (action - a_lead) * DT * DT - (v - v_lead) * DT + nd
    next_v_lead = v_lead + a_lead * DT + nv_lead
    next_a_lead = 0.5 * (v_desired - (v_lead + a_lead * DT)) + na_lead

    next_state = [next_d, next_v, next_v_lead, next_a_lead]
    return torch.tensor([next_state], device=device, dtype=torch.float32)

# 定义测量模型
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

# 定义DQN网络
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        num_neros= 512
        self.layer1 = nn.Linear(n_observations, num_neros)
        self.layer2 = nn.Linear(num_neros, num_neros)
        self.layer3 = nn.Linear(num_neros, num_neros)
        self.layer4 = nn.Linear(num_neros, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)

# 选择动作
def select_action_index(particles):
    value_list = [0 for _ in range(7)]
    for particle in particles:
        action_index = policy_net(particle).max(1).indices.item()
        value_list[action_index] += 1

    # 返回 value_list 中最大值的索引
    max_index = value_list.index(max(value_list))
    return torch.tensor([[max_index]], device=device, dtype=torch.long)
# def select_action_index(particles):
#     Qvalue_list = [0 for _ in range(7)]
#     for i in range(len(particles)):
#         value_list = policy_net(particles[i]).tolist()[0]
#         for i_action in range(7):
#             Qvalue_list[i_action] = Qvalue_list[i_action] + value_list[i_action]
#     # 返回 value_list 中最大值的索引
#     max_index = Qvalue_list.index(max(Qvalue_list))
#     return torch.tensor([[max_index]], device=device, dtype=torch.long)

# def select_action_index(state):
#     return policy_net(state).max(1).indices.view(1, 1)  # 1*1

# 定义奖励函数
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

    next_d, next_v, next_v_lead, next_a_lead= next_state.numpy()[0]

    R_danger = -100 * indicator_function(next_d < d_safe)
    R_smooth = -1 * indicator_function(abs(action - action_before) > thr_smooth)
    R_efficiency = -1*(indicator_function(next_v > v_max)* abs(next_v-v_max)+indicator_function(next_v < v_min)* abs(next_v-v_min))

    return R_danger + R_smooth + R_efficiency

# 主程序
if __name__ == '__main__':
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    num_particles = 1000
    mean_reward_dict = {}
    episode_rewards = []
    possible_actions = [-3, -2, -1, 0, 1, 2, 3]
    action_indexes = [0, 1, 2, 3, 4, 5, 6]
    n_actions = len(possible_actions)
    action_index_before = torch.tensor([[random.choice(action_indexes)]], device=device, dtype=torch.long)
    state = torch.tensor([random.uniform(10, 15), random.uniform(20, 30), random.uniform(20, 30), random.uniform(-3, 3)], dtype=torch.float32, device=device).unsqueeze(0)
    policy_net = DQN(state.size(1), n_actions).to(device)
    policy_net.load_state_dict(torch.load('dqn_model.pth', map_location=device))
    policy_net.eval()

    num_episodes = 100
    reward_set = []

    for i_episode in range(num_episodes):
        episode_reward = 0
        episode_reward_set = []

        random.seed(i_episode)
        v0 = random.uniform(20, 30)
        state = [random.uniform(10, 15), v0, random.uniform(v0, 30), random.uniform(-3, 3)]
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        measurement = measurement_model(state)
        action_index_before = torch.tensor([[3]], device=device, dtype=torch.long)
        re_weights, re_particles = particle_init(num_particles)
        re_particles, re_weights = particle_filter(num_particles, measurement, re_particles, re_weights,
                                                   action_index_before)
        estimated_state = get_estimated_state(re_weights, re_particles)

        for t in range(20):
            episode_reward_set.append(episode_reward)
            action_index = select_action_index(re_particles)
            next_state = get_next_state(state, action_index)
            reward = torch.tensor([get_reward(next_state, action_index_before, action_index)], device=device)
            state = next_state
            action_index_before = action_index
            action_index_est = action_index
            measurement = measurement_model(state)
            re_particles, re_weights = particle_filter(num_particles, measurement, re_particles, re_weights, action_index_est)
            #estimated_state = get_estimated_state(re_weights, re_particles)
            episode_reward += reward.item()
            #print(state)
            #print(estimated_state)
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
with open('DQN_reward_MAP_test.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for episode_index, episode in enumerate(reward_set):
        # 为每个 episode 写入一个标题行，标识 episode 的索引
        writer.writerow(['Episode ' + str(episode_index)])
        for reward in episode:
            # 将 NumPy 数组转换为列表，并写入 CSV 文件
            writer.writerow([reward])




