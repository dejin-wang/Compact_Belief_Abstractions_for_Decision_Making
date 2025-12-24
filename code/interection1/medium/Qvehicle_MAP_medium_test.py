import math
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def select_action_index(particles):
    value_list = [0 for _ in range(7)]
    for particle in particles:
        action_index = policy_net(particle).max(1).indices.item()
        value_list[action_index] += 1

    # 返回 value_list 中最大值的索引
    max_index = value_list.index(max(value_list))
    return torch.tensor([[max_index]], device=device, dtype=torch.long)

def get_next_state(state, action_index):
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
    if next_v < 0:
        next_v = 0
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
    v_max = 15
    v_min = 5

    next_d, next_v, next_fai, next_tfai = next_state.numpy()[0]

    R_term = 100 * indicator_function((next_d <= 0) and ((next_fai == 0) or (next_fai == 1)))
    R_RRl = -200 * indicator_function((next_d <= 0) and (next_fai == 2))
    R_vel = -100 * indicator_function((next_v > v_max) or ((next_v < v_min) and (next_d > d_critical)))
    R_rev = -100 * indicator_function(next_v < 0)
    R_smooth = -10 * indicator_function(abs(action - action_before) > thr_smooth)

    R_act = -2

    return R_term + R_RRl + R_vel + R_rev + R_smooth + R_act

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        num_neurons = 512
        self.layer1 = nn.Linear(n_observations, num_neurons)
        self.layer2 = nn.Linear(num_neurons, num_neurons)
        self.layer3 = nn.Linear(num_neurons, num_neurons)
        self.layer4 = nn.Linear(num_neurons, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)

def measurement_model(state):
    Fai = [0, 1, 2]  # G Y R
    d, v, fai, tfai = state.numpy()[0]
    nd = np.random.normal(0, 5)
    nv = np.random.normal(0, 0.5)
    nt= np.random.normal(0, 0.1)
    sample = np.random.random()
    if sample <0.91:
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

        likelihood1 = (0.2 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * 0.2*0.2 * (d_m - d) ** 2)
        likelihood2 = (2/ np.sqrt(2 * np.pi)) * np.exp(-0.5 * 2*2 * (v_m - v) ** 2)
        if fai == fai_m:
            likelihood3 = 0.94
        else:
            likelihood3 = 0.06
        #likelihood4 = (10 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * 100 * (tfai_m - tfai) ** 2)
        likelihood4 = 1

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
    d_est = 0
    v_est = 0
    fai_0 = 0
    fai_1 = 0
    fai_2 = 0
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

    array = np.array([fai_0, fai_1, fai_2])
    fai_est = np.argmax(array)

    tfai_est = 0
    for i in range(l):
        if fai_est == particles[i].numpy()[0][2]:
            tfai_est = tfai_est + particles[i].numpy()[0][3] * weights[i]

    estimated_state = [d_est, v_est, fai_est, tfai_est]
    estimated_state = torch.tensor(estimated_state, dtype=torch.float32, device=device).unsqueeze(0)

    return estimated_state

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
possible_actions = [-3, -2, -1, 0, 1, 2, 3]
n_actions = len(possible_actions)

Fai = [0, 1, 2]  # G Y R
TFai = [10, 5, 10]  # G Y R
probabilities = [0.4, 0.2, 0.4]
num_particles = 3000

fai_0 = random.choices(Fai, weights=probabilities, k=1)[0]
state = [random.uniform(110, 120), random.uniform(5, 15), fai_0, random.uniform(0, TFai[fai_0])]
n_observations = len(state)
policy_net = DQN(n_observations, n_actions).to(device)
reward_set = []
policy_net.load_state_dict(torch.load('dqn_model.pth', map_location=device))
policy_net.eval()
numb = 100
n_rrd = 0
reward_set = []

for i in range(numb):
    episode_reward_set = []
    random.seed(i)
    action_index_before = 3
    action_index_before = torch.tensor([[action_index_before]], device=device, dtype=torch.long)
    action_index_est_before = action_index_before
    fai_0 = random.choices(Fai, weights=probabilities, k=1)[0]
    state = [random.uniform(110, 120), random.uniform(5, 15), fai_0, random.uniform(0, TFai[fai_0])]
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    #print(state)
    measurement = measurement_model(state)
    re_weights, re_particles = particle_init(num_particles, measurement)
    estimated_state = get_estimated_state(re_weights, re_particles)

    episode_rewards = []
    episode_reward = 0

    for t in count():
        episode_reward_set.append(episode_reward)
        action_index = select_action_index(re_particles)
        done = (state.numpy()[0][0] < 0) or (t > 29)
        if state.numpy()[0][0] < 0 and state.numpy()[0][2] == 2:
            n_rrd += 1
        if state.numpy()[0][0] < 0:
            next_state = None
        else:
            next_state = get_next_state(state, action_index)
        state = next_state


        if done:
            break

        action_index_est = action_index
        measurement = measurement_model(state)
        re_particles, re_weights = particle_filter(num_particles, measurement, re_particles, re_weights, action_index_est)
        # print(state)
        # print(re_particles[0])

        estimated_state = get_estimated_state(re_weights, re_particles)
        reward = torch.tensor([get_reward(state, action_index_before, action_index_est)], device=device)
        episode_reward += reward.item()
        #print(state)
        #print(estimated_state)
        # for i in range(num_particles):
        #     print(re_particles[i].numpy()[0][2])
        action_index_before = action_index
    reward_set.append(episode_reward_set)
    print(f"Episode {i + 1}/{numb}, Reward: {episode_reward}")

longest_array = max(reward_set, key=len)

for i in range(numb):
    if len(reward_set[i]) < max(len(longest_array), 30):
        l = max(len(longest_array), 30) - len(reward_set[i])
        for j in range(l):
            reward_set[i].append(reward_set[i][-1])

import csv
with open('DQN_MAP_reward_test_medium.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for episode_index, episode in enumerate(reward_set):
        writer.writerow(['Episode ' + str(episode_index)])
        for reward in episode:
            writer.writerow([reward])

print(n_rrd)
