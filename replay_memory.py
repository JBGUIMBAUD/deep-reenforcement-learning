from collections import namedtuple
import random


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, next_state, reward, end):
        """Saves a transition."""
        Transition = namedtuple('Transition',
                                ('state', 'action', 'next_state', 'reward', 'done'))

        if len(self.memory) < self.capacity:
            self.memory.append(Transition(state, action, next_state, reward, end))
        else:
            self.memory[self.position] = Transition(state, action, next_state, reward, end)
        # print("{0} position {1}".format(self.memory[self.position], str(self.position)))
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        # print(len(self.memory))
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
