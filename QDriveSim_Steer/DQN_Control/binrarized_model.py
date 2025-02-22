import torch
import torch.nn as nn
import torch.nn.functional as F


def binarize(tensor):
    return torch.sign(tensor)


class BinaryConv2d(nn.Conv2d):
    def forward(self, input):
        binary_weight = binarize(self.weight)
        return F.conv2d(input, binary_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class BinaryLinear(nn.Linear):
    def forward(self, input):
        binary_weight = binarize(self.weight)
        if self.bias is not None:
            return F.linear(input, binary_weight, self.bias)
        else:
            return F.linear(input, binary_weight)


class BinarizedConvNet(nn.Module):
    def __init__(self, dim, in_channels, num_actions) -> None:
        super(BinarizedConvNet, self).__init__()
        self.conv1 = BinaryConv2d(in_channels, 32, 8, 4)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = BinaryConv2d(32, 64, 4, 3)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = BinaryConv2d(64, 64, 3, 1)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.fc1 = BinaryLinear(64 * 8 * 8, 256)
        self.fc1_bn = nn.BatchNorm1d(256)
        self.fc2 = BinaryLinear(256, 32)
        self.fc2_bn = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, num_actions)  # The final layer can remain as a standard linear layer

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = self.fc3(x)
        return x


import copy
import numpy as np


class DQN(object):
    def __init__(
            self,
            num_actions,
            state_dim,
            in_channels,
            device,
            discount=0.9,
            optimizer="Adam",
            optimizer_parameters=None,
            target_update_frequency=1e4,
            initial_eps=1,
            end_eps=0.05,
            eps_decay_period=25e4,
            eval_eps=0.001,
            use_binarized=False
    ) -> None:
        if optimizer_parameters is None:
            optimizer_parameters = {'lr': 0.01}
        self.current_eps = None
        self.device = device

        self.Q = BinarizedConvNet(state_dim, in_channels, num_actions).to(self.device) if use_binarized else ConvNet(
            state_dim, in_channels, num_actions).to(self.device)
        self.Q_target = copy.deepcopy(self.Q)  # copy target network
        self.Q_optimizer = getattr(torch.optim, optimizer)(self.Q.parameters(), **optimizer_parameters)

        self.discount = discount
        self.target_update_frequency = target_update_frequency

        # epsilon decay
        self.initial_eps = initial_eps
        self.end_eps = end_eps
        self.slope = (self.end_eps - self.initial_eps) / eps_decay_period

        self.state_shape = (-1,) + state_dim
        self.eval_eps = eval_eps
        self.num_actions = num_actions
        self.iterations = 0

    def select_action(self, state, eval=False):
        eps = self.eval_eps if eval else max(self.slope * self.iterations + self.initial_eps, self.end_eps)
        self.current_eps = eps

        if np.random.uniform(0, 1) > eps:
            self.Q.eval()
            with torch.no_grad():
                state = torch.FloatTensor(state).reshape(self.state_shape).to(self.device)
                return int(self.Q(state).argmax(1))
        else:
            return np.random.randint(self.num_actions)

    def train(self, replay_buffer):
        self.Q.train()
        state, action, next_state, reward, done = replay_buffer.sample()
        with torch.no_grad():
            target_Q = reward + (1 - done) * self.discount * self.Q_target(next_state).max(1, keepdim=True)[0]
        current_Q = self.Q(state).gather(1, action)
        Q_loss = F.smooth_l1_loss(current_Q, target_Q)
        self.Q_optimizer.zero_grad()
        Q_loss.backward()
        self.Q_optimizer.step()
        self.iterations += 1
        self.copy_target_update()

    def copy_target_update(self):
        if self.iterations % self.target_update_frequency == 0:
            print('Target network updated')
            print('Current epsilon:', self.current_eps)
            self.Q_target.load_state_dict(self.Q.state_dict())

    def save(self, filename):
        torch.save(self.Q.state_dict(), filename + "_Q")
        torch.save(self.Q_optimizer.state_dict(), filename + "_optimizer")

    def load(self, filename):
        self.Q.load_state_dict(torch.load(filename + "_Q"))
        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer.load_state_dict(torch.load(filename + "_optimizer"))
