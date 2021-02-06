from abc import ABC

import torch as t
from torch import Tensor
import torch.optim as optim
import torch.nn.functional as f
from torch.autograd import Variable

import random as r


class QNet(ABC):
    def predict(self, net_input: Tensor) -> Tensor:
        pass

    def get_optimizer(self) -> optim.Adam:
        pass


class Memory:
    def __init__(self, state: Tensor, action: int, reward: float):
        self.state = state
        self.action = action
        self.reward = reward


class ReplayMemory:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memories = []

    def add_memory(self, memory: Memory):
        self.memories.append(memory)
        if len(self.memories) > self.capacity:
            self.memories.pop(0)

    def get_random_batch(self, size: int):
        if len(self.memories) < size:
            raise Exception("There are not enough memories to make a batch.")
        start_index = r.randint(0, len(self.memories) - size)
        end_index = start_index + size
        return self.memories[start_index:end_index]


class QLearning:
    def __init__(self):
        self.net = None

    def get_best_action(self, prediction: Tensor):
        best_action_index = 0
        for action_index in range(len(prediction)):
            if prediction[action_index].data > prediction[best_action_index].data:
                best_action_index = action_index
        return best_action_index

    def train(self, batch: [Memory]):
        predictions = []
        optimal_predictions = []
        for i in range(len(batch)):
            predictions.append(self.net.predict(batch[i].state))

            optimal_prediction = predictions[i].clone()
            optimal_prediction[batch[i].action] = batch[i].reward
            optimal_predictions.append(optimal_prediction)

        loss = f.mse_loss(t.stack(predictions), t.stack(optimal_predictions))
        self.net.get_optimizer().zero_grad()
        loss.backward()
        self.net.get_optimizer().step()
