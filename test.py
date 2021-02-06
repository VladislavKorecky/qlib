import torch.nn as nn
import torch.optim as optim

from q_ai import *


class Net(nn.Module, QNet):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(2, 2)
        self.l2 = nn.Linear(2, 2)
        self.optimizer = optim.Adam(self.parameters(), lr=0.1)

    def predict(self, net_input: Tensor) -> Tensor:
        output = f.relu(self.l1(state))
        output = self.l2(output)
        return output

    def get_optimizer(self) -> optim.Adam:
        return self.optimizer


q = QLearning()

net = Net()
q.net = net

replay_memory = ReplayMemory(100)

epsilon = 1
for i in range(100):
    state = t.tensor([0, 1], dtype=t.float)

    if r.random() > epsilon:
        action = q.get_best_action(net.predict(state))
    else:
        action = r.randrange(0, 2)
    epsilon -= 0.01

    reward = 0
    if action == 1:
        reward += 1
    print(reward)
    replay_memory.add_memory(Memory(state, action, reward))

    if len(replay_memory.memories) > 10:
        q.train(replay_memory.get_random_batch(10))
