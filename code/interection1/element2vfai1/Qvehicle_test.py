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


def select_action_index(state):
    # obtain possible actions
    action_indexes = [0, 1, 2, 3, 4, 5, 6]
    return policy_net(state).max(1).indices.view(1, 1)  # 1*1


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



device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
possible_actions = [-3, -2, -1, 0, 1, 2, 3]
n_actions = len(possible_actions)

Fai = [0, 1, 2]  # G Y R
TFai = [10, 5, 10]  # G Y R
# the probabilities of different lights
probabilities = [0.4, 0.2, 0.4]

# initial state
fai_0 = random.choices(Fai, weights=probabilities, k=1)[0]
state = [random.uniform(80, 120), random.uniform(5, 15), fai_0, random.uniform(0, TFai[fai_0])]
#state = [106.7, 10.8, 0, 4]
n_observations = len(state)
policy_net = DQN(n_observations, n_actions).to(device)

reward_set=[]
numb=100
n_rrd=0
for i in range(numb):
    random.seed(i)  # 你可以使用任何整数值作为种子
    # initial a = 0
    episode_reward_set = []
    action_index_before = 3
    action_index_before = torch.tensor([[action_index_before]], device=device, dtype=torch.long)
    # initial state
    fai_0 = random.choices(Fai, weights=probabilities, k=1)[0]
    state = [random.uniform(80, 120), random.uniform(5, 15), fai_0, random.uniform(0, TFai[fai_0])]

    #state = [108, 10.8, 0, 7]
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    print(state)

    policy_net.load_state_dict(torch.load('dqn_model.pth', map_location=device))
    policy_net.eval()

    # initial reward
    episode_reward = 0
    episode_reward_aver=0

    for t in count():
        #episode_reward_set.append(episode_reward_aver)
        episode_reward_set.append(episode_reward)
        action_index = select_action_index(state)
        done = (state.numpy()[0][0] < 0) or (t > 1000)
        if state.numpy()[0][0] < 0 and state.numpy()[0][2] == 2:
            n_rrd=n_rrd+1
        if state.numpy()[0][0] < 0:
            next_state = None
            reward = torch.tensor([0], device=device)
        else:
            next_state = get_next_state(state, action_index)
            reward = torch.tensor([get_reward(next_state, action_index_before, action_index)], device=device)

        # Move to the next state
        #print('this is the begining')
        #print(state)
        #print(possible_actions[action_index.item()])
        #print(possible_actions[action_index_before.item()])
        #print(next_state)
        #print(reward)
        #print('this is the end')
        print('action')
        print(action_index_before)
        print(action_index)
        state = next_state
        print(state)
        print(reward)
        print('action')


        action_index_before = action_index
        episode_reward += reward.item()
        episode_reward_aver=episode_reward/(t+1)
        # print(episode_reward)
        if done:
            break
    reward_set.append(episode_reward_set)
    print(f"Episode {i + 1}/{numb}, Reward: {episode_reward}")


longest_array = max(reward_set, key=len)

for i in range(numb):
    if len(reward_set[i])< max(len(longest_array),30):
        l= max(len(longest_array),30)-len(reward_set[i])
        for j in range (l):
            reward_set[i].append(reward_set[i][-1])

import csv
with open('DQN_reward_test0.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for episode_index, episode in enumerate(reward_set):
        # 为每个 episode 写入一个标题行，标识 episode 的索引
        writer.writerow(['Episode ' + str(episode_index)])
        for reward in episode:
            # 将 NumPy 数组转换为列表，并写入 CSV 文件
            writer.writerow([reward])



print(n_rrd)





