import torch as t
import torch.nn as nn
import torch.nn.functional as f

from q_ai import *


class Net(nn.Module, QNet):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(2, 2)
        self.l2 = nn.Linear(2, 2)

    def feed_forward(self, state) -> Tensor:
        output = f.relu(self.l1(state))
        output = self.l2(output)
        return output

q = QLearning()

net = Net()
q.net = net

replay_memory = ReplayMemory(10)
print(q.predict(t.tensor([0, 1], dtype=t.float)))