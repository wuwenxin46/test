import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
from torchsummary import summary

import DQNenv
from DQNenv import envCube, Cube
from DQNet import Qnet
from Replay_memory import ReplayMemory, Transition
from Agent import Agent
from Train import run_episode, show_print, episode_evaluate, draw_plot, test
from tqdm import tqdm

import cv2
from PIL import Image
import time
import pickle
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

if __name__ == "__main__":
    path = 'net.pth'
    env = envCube()
    observation_n = env.OBSEVATION_SPACE_VALUES
    action_n = env.ACTION_SPACE_VALUES
    epsilon = 0.4  # DQNenv.epsilon
    agent = Agent(observation_n, action_n, gamma=0.98, lr=1e-3, target_update=50)
    replayMemory = ReplayMemory(memory_size=10000)
    batch_size = 10
    num_episodes = 10000
    # show_every = DQNenv.SHOW_EVERY  # 3000
    # decay = DQNenv.EPS_DECAY  # 0.9998
    a = 10
    reward_list = []
    for i in range(a):
        if path != '':
            agent.q_net.load_state_dict(torch.load(path))
            # agent.target_q_net.load_state_dict(torch.load("net.pth"))
        with tqdm(total=int(num_episodes / a), desc="Iteration %d" % i) as pbar:
            for episode in range(int(num_episodes / a)):
                show = False
                reward_episode = run_episode(env, agent, replayMemory, batch_size, epsilon)
                reward_list.append(reward_episode)

                pbar.set_postfix({
                    'episode': '%d' % (episode + 1),
                    'return': '%.3f' % (np.mean(reward_list).item())
                })
                pbar.update(1)  # 更新进度条
        test_reward = episode_evaluate(env, agent, epsilon, False)
        # print("Episode %d, total reward: %.3f" % (episode, test_reward))
        print(f'test_eva:{test_reward}')
        torch.save(agent.q_net.state_dict(), 'net.pth')


    # for episode in range(num_episodes):
    #     if len(reward_list) == 0:
    #         show = False
    #     else:
    #         show = True
    #         test_reward = episode_evaluate(env, agent, epsilon, show)
    #         print(f'return:{test_reward}')
    #     reward_episode = run_episode(env, agent, replayMemory, batch_size, epsilon, show)
    #     reward_list.append(reward_episode)
    #     # epsilon *= decay


    # for episode in range(EPISODES):
    #     obs = env.reset()
    #     done = False
    #
    #     # 显示
    #     if episode % SHOW_EVERY == 0:
    #         print(f'episode #{episode}, epsilon:{epsilon}')
    #         print(f'mean reward:{np.mean(episode_rewards[-SHOW_EVERY:])}')
    #         show = True
    #     else:
    #         show = False
    #
    #     episode_reward = 0
    #     while not done:
    #         if np.random.random() > epsilon:
    #             action = np.argmax(q_table[obs])
    #         else:
    #             action = np.random.randint(0, env.ACTION_SPACE_VALUES)
    #
    #         new_obs, reward, done = env.step(action)  # move
    #
    #         # 更新Q_table
    #         current_q = q_table[obs][action]
    #         max_future_q = np.max(q_table[new_obs])
    #         if reward == env.FOOD_REWARD:
    #             new_q = env.FOOD_REWARD
    #         else:
    #             new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
    #
    #         q_table[obs][action] = new_q
    #         obs = new_obs
    #
    #         if show:
    #             env.render()
    #         episode_reward += reward




