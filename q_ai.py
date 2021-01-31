from abc import ABC

from torch import Tensor

import random as r


class QNet(ABC):
    def feed_forward(self, state) -> Tensor:
        pass


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memories = []

    def add_memory(self, memory):
        self.memories.append(memory)
        if len(self.memories) > self.capacity:
            self.memories.pop(0)

    def get_random_batch(self, size):
        if len(self.memories) < size:
            raise Exception("There are not enough memories to make a batch.")
        start_index = r.randint(0, len(self.memories) - size)
        end_index = start_index + size
        return self.memories[start_index:end_index]


class Memory:
    def __init__(self, state, action, reward):
        self.state = state
        self.action = action
        self.reward = reward


class QLearning:
    def __init__(self):
        self.net = None

    def get_best_action(self, prediction):
        best_action_index = 0
        for action_index in range(len(prediction)):
            if prediction[action_index].data > prediction[best_action_index].data:
                best_action_index = action_index
        return best_action_index

    def predict(self, state):
        prediction = self.net.feed_forward(state)
        action = self.get_best_action(prediction)
        return action