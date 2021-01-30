from abc import ABC
import torch
from torch import Tensor


class QNet(ABC):
    def feed_forward(self, network_input) -> Tensor:
        pass


class QLearning:
    def __init__(self):
        self.net = None
