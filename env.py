import numpy as np
import cv2
from PIL import Image
import time
import pickle
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


EPISODES = 30000
SHOW_EVERY = 3000  # 每隔3000局展示一下过程

epsilon = 0.6  # 对动作抽取概率 60%抽取随机动作， 40%最大期望值的动作
EPS_DECAY = 0.9998  # 越往后越使用最大期望的动作
DISCOUNT = 0.95  # 折扣回报
LEARNING_RATE = 0.1  # 学习速率
# q_table = None


class envCube:
    SIZE = 10
    OBSEVATION_SPACE_VALUES = (SIZE, SIZE, 3)
    ACTION_SPACE_VALUES = 9
    RETURN_IMAGE = False

    FOOD_REWARD = 25
    ENEMY_PENALITY = -300
    MOVE_PENALITY = -1

    d = {1: (255, 0, 0),
         2: (0, 255, 0),
         3: (0, 0, 255)}

    PLAYER_N = 1
    FOOD_N = 2
    ENEMY_N = 3

    def reset(self):
        self.player = Cube(self.SIZE)
        self.food = Cube(self.SIZE)
        while self.food == self.player:
            self.food = Cube(self.SIZE)

        self.enemy = Cube(self.SIZE)
        while self.enemy == self.player or self.enemy == self.food:
            self.enemy = Cube(self.SIZE)

        if self.RETURN_IMAGE:
            observation = np.array(self.get_image())
        else:
            observation = (self.player - self.food) + (self.player - self.enemy)

        self.episode_step = 0

        return observation

    def step(self, action):
        self.episode_step += 1
        self.player.action(action)
        self.food.move()
        self.enemy.move()

        if self.RETURN_IMAGE:
            new_observation = np.array(self.get_image())
        else:
            new_observation = (self.player - self.food) + (self.player - self.enemy)

        if self.player == self.food:
            reward = self.FOOD_REWARD
        elif self.player == self.enemy:
            reward = self.ENEMY_PENALITY
        else:
            reward = self.MOVE_PENALITY

        done = False
        if self.player == self.food or self.player == self.enemy or self.episode_step >= 200:
            done = True

        return new_observation, reward, done

    def render(self):
        img = self.get_image()
        img = img.resize((800, 800))
        cv2.imshow('Predator', np.array(img))
        cv2.waitKey(1)

    def get_image(self):
        env = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)
        env[self.food.x][self.food.y] = self.d[self.FOOD_N]
        env[self.player.x][self.player.y] = self.d[self.PLAYER_N]
        env[self.enemy.x][self.enemy.y] = self.d[self.ENEMY_N]
        img = Image.fromarray(env, 'RGB')
        return img

    def get_qtable(self, qtable_name=None):
        if qtable_name is None:
            q_table = {}
            for x1 in range(-self.SIZE + 1, self.SIZE):  # player 和 food 间的 横坐标 的 差值
                for y1 in range(-self.SIZE + 1, self.SIZE):  # player 和 food 间的 纵坐标 的 差值
                    for x2 in range(-self.SIZE + 1, self.SIZE):  # player 和 enemy 间的 横坐标 的 差值
                        for y2 in range(-self.SIZE + 1, self.SIZE):  # player 和 enemy 间的 纵坐标 的 差值
                            q_table[(x1, y1, x2, y2)] = [np.random.uniform(-5, 0) for i in range(self.ACTION_SPACE_VALUES)]
        else:
            with open(qtable_name, 'rb') as f:
                q_table = pickle.load(f)
        return q_table


class Cube:
    def __init__(self, size):
        self.size = size
        self.x = np.random.randint(0, self.size)
        self.y = np.random.randint(0, self.size)

    def __str__(self):
        return f'{self.x},{self.y}'

    def __sub__(self, other):
        return self.x-other.x, self.y-other.y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def action(self, choice):
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=1)
        elif choice == 2:
            self.move(x=1, y=-1)
        elif choice == 3:
            self.move(x=-1, y=-1)
        elif choice == 4:
            self.move(x=0, y=1)
        elif choice == 5:
            self.move(x=0, y=-1)
        elif choice == 6:
            self.move(x=1, y=0)
        elif choice == 7:
            self.move(x=-1, y=0)
        elif choice == 8:
            self.move(x=0, y=0)

    def move(self, x=False, y=False):
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        if self.x < 0:
            self.x = 0
        elif self.x >= self.size:
            self.x = self.size - 1

        if self.y < 0:
            self.y = 0
        elif self.y >= self.size:
            self.y = self.size - 1


# env = envCube()
# print(env.reset())
# new_observation, reward, done = env.step(3)
# print(new_observation)
# print(reward)
# print(done)
# env.render()
# q_table = env.get_qtable()
# print(q_table[new_observation])

# if q_table is None:
#     q_table = {}
#     for x1 in range(-SIZE+1, SIZE):  # player 和 food 间的 横坐标 的 差值
#         for y1 in range(-SIZE + 1, SIZE):  # player 和 food 间的 纵坐标 的 差值
#             for x2 in range(-SIZE + 1, SIZE):  # player 和 enemy 间的 横坐标 的 差值
#                 for y2 in range(-SIZE + 1, SIZE):  # player 和 enemy 间的 纵坐标 的 差值
#                     q_table[((x1, y1), (x2, y2))] = [np.random.uniform(-5, 0) for i in range(4)]
# else:
#     with open(q_table, 'rb') as f:
#         q_table = pickle.load(f)
#
#

q_name = 'qtable_1704641819.pickle'
env = envCube()
q_table = env.get_qtable(q_name)
episode_rewards = []

for episode in range(EPISODES):
    obs = env.reset()
    done = False

    # 显示
    if episode%SHOW_EVERY == 0:
        print(f'episode #{episode}, epsilon:{epsilon}')
        print(f'mean reward:{np.mean(episode_rewards[-SHOW_EVERY:])}')
        show = True
    else:
        show = False

    episode_reward = 0
    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0, env.ACTION_SPACE_VALUES)

        new_obs, reward, done = env.step(action)  # move

        # 更新Q_table
        current_q = q_table[obs][action]
        max_future_q = np.max(q_table[new_obs])
        if reward == env.FOOD_REWARD:
            new_q = env.FOOD_REWARD
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

        q_table[obs][action] = new_q
        obs = new_obs

        if show:
            env.render()
        episode_reward += reward

    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY

# 绘制曲线
moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')
plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.xlabel('episode #')
plt.ylabel(f'mean {SHOW_EVERY}reward')
plt.show()

# 储存Q_table
with open(f'qtable_{int(time.time())}.pickle','wb') as f:
    pickle.dump(q_table, f)


# 测试函数
def test(q_table, episodes, show_enable=True):
    env = envCube()
    avg_reward = 0
    for episode in range(episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = np.argmax(q_table[obs])
            obs, reward, done = env.step(action)
            if show_enable == True:
                env.render()
            episode_reward += reward
        print(f'episode #{episode}, episode_reward:{episode_reward}')
        avg_reward += avg_reward
    avg_reward /= episodes
    print(f'avg_reward:{avg_reward}')

# print(q_table[((1, 3), (-2, -4))])
# player = Cube()
# print(player)
# player.action(0)
# print(player)
# food = Cube()
# print(food)
# print(player - food)


