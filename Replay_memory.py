import random
from collections import deque, namedtuple
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayMemory(object):
    def __init__(self, memory_size):
        self.memory = deque([], maxlen=memory_size)  # 队列

    def sample(self, batch_size):
        batch_data = random.sample(self.memory, batch_size)  # 在经验记忆中，随机采样一个批次
        state, action, reward, next_state, done = zip(*batch_data)
        # [2,3,4,5,0],(3,4,5,6,1) → (2,3), (3,4), (4,5), (5,6), (0,1)
        return state, action, reward, next_state, done

    def push(self, *args):
        # *args: 把传进来的所有参数都打包起来生成元组形式
        # self.push(1, 2, 3, 4, 5)
        # args = (1, 2, 3, 4, 5)
        self.memory.append(Transition(*args))

    def __len__(self):
        return len(self.memory)