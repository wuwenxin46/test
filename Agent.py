import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
from DQNet import Qnet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Agent(object):
    def __init__(self, nb_observations, nb_actions, reward_decay, learning_rate, target_update):
        self.action_dim = nb_actions
        self.observation_dim = nb_observations
        self.q_net = Qnet(self.observation_dim, self.action_dim).to(device)
        self.target_q_net = Qnet(self.observation_dim, self.action_dim).to(device)
        self.gamma = reward_decay
        self.lr = learning_rate
        self.target_update = target_update
        self.count = 0

        self.optimizer = optim.Adam(params=self.q_net.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()  # 均方误差公式

    def take_action(self, state, epsilon):
        if np.random.uniform(0, 1) > epsilon:
            state = torch.tensor(state, dtype=torch.float).to(device)
            action = torch.argmax(self.q_net(state)).item()
        else:
            action = np.random.choice(self.action_dim)
        return action

    def update(self, transition_dict):

        states = transition_dict.state
        actions = np.expand_dims(transition_dict.action, axis=-1)  # [1, 4] → [[1] [4]]
        rewards = np.expand_dims(transition_dict.reward, axis=-1)
        next_states = transition_dict.next_state
        dones = np.expand_dims(transition_dict.done, axis=-1)

        states = torch.tensor(states, dtype=torch.float).to(device)
        actions = torch.tensor(actions, dtype=torch.int64).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(device)
        dones = torch.tensor(dones, dtype=torch.float).to(device)

        predict_q_values = self.q_net(states).gather(1, actions)
        # states:[[2,3,4,5], [3,4,5,6]], action:[[1] [4]]
        # q_net(states):[[1,9,3,4,5,6,7,8,2], [0,1,2,3,8,5,6,7,4]] .gather(1, actions) → 9, 8
        # 按dim=1, index = actions 选，[(0,0) (1,0)] → [(0,1) (1,4)]
        # 即 选出 当前动作下 （最大）的状态期望值
        with torch.no_grad():
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)  # 使得每个状态的最大Q值占一行
            q_targets = rewards + self.gamma * max_next_q_values * (1-dones)
        l = self.loss(predict_q_values, q_targets)

        self.optimizer.zero_grad()
        l.backward()
        self.optimizer.step()

        # 间隔固定次数，使得主网络（main network：q_net）的参数复制到 目标网络（target network：target_q_net）中
        # 这样做的目的是为了 稳定学习过程，避免目标Q值随着主网络的更新而频繁变化，导致训练不收敛或震荡
        if self.count % self.target_update == 0:  # target_update为目标更新频率
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1
        # print(l.cpu().detach().numpy())
        return l.cpu().detach().numpy()







