import torch.nn as nn

from q_ai import *


class Net(nn.Module, QNet):
    def feed_forward(self, network_input) -> Tensor:
        return Tensor()


q = QLearning()
net = Net()
q.net = net

print(q.net.feed_forward(None))
