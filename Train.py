import numpy as np
from DQNenv import envCube, Cube
from Replay_memory import ReplayMemory, Transition
import time
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


def run_episode(env, agent, replaymemory, batch_size, epsilon):
    state = env.reset()
    reward_total = 0
    cost_his = []
    while True:
        action = agent.take_action(state, epsilon)
        next_state, reward, done = env.step(action)
        replaymemory.push(state, action, reward, next_state, done)
        reward_total += reward
        if len(replaymemory) > batch_size:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = replaymemory.sample(batch_size)
            T_data = Transition(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
            l = agent.update(T_data)
            cost_his.append(l.item())
        state = next_state
        if done:
            break
    if len(cost_his) != 0:
        loss = np.mean(cost_his).item()
    else:
        loss = 'NotCal'
    return reward_total, loss


def show_print(episode_rewards, episode, show_every, epsilon):
    # 显示
    if episode % show_every == 0:
        print(f'episode #{episode}, epsilon:{epsilon}')
        print(f'mean reward:{np.mean(episode_rewards[-show_every:])}')
        show = True
    else:
        show = False
    return show


def episode_evaluate(env, agent, epsilon, show):
    episode_rewards = []
    for episode in range(5):
        state = env.reset()
        episode_reward = 0
        while True:
            action = agent.take_action(state, epsilon)
            next_state, reward, done = env.step(action)
            episode_reward += reward
            state = next_state
            if show:
                env.render()
            if done:
                break

        episode_rewards.append(episode_reward)
    return np.mean(episode_rewards).item()


def draw_plot(episode_rewards, show_every):
    SHOW_EVERY = show_every
    moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')
    plt.plot([i for i in range(len(moving_avg))], moving_avg)
    plt.xlabel('episode #')
    plt.ylabel(f'mean {SHOW_EVERY}reward')
    plt.show()


def test(env, agent, epsilon, episodes, delay_time, show_enable=True):
    env = envCube()
    state = env.reset()
    reward_episode = 0
    while True:
        action = agent.take_action(state, epsilon)
        next_state, reward, done = env.step(action)
        reward_episode += reward
        state = next_state
        if done:
            break
        env.render()
        time.sleep(delay_time)

    # for episode in range(episodes):
    #     state = env.reset()
    #     done = False
    #     episode_reward = 0
    #     while not done:
    #         action = agent.take_action(state)
    #         next_state, reward, done = env.step(action)
    #         episode_reward += reward
    #         state = next_state
    #
    #         if show_enable == True:
    #             env.render()
    #             time.sleep(delay_time)
    #     print(f'episode #{episode}, episode_reward:{episode_reward}')
    #     avg_reward += avg_reward
    # avg_reward /= episodes
    # print(f'avg_reward:{avg_reward}')



