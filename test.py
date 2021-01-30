import torch.nn as nn

from q_ai import *


class Net(nn.Module, QNet):
    def feed_forward(self, network_input) -> Tensor:
        return Tensor()


q = QLearning()

net = Net()
q.net = net

replay_memory = ReplayMemory(10)
for i in range(10):
    replay_memory.add_memory(Memory(i, i, i))

batch = replay_memory.get_random_batch(3)
for memory in batch:
    print(memory.__dict__)