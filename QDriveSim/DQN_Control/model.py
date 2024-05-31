import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self, in_channels, num_actions_steer, num_actions_throttle) -> None:
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 8, 4)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 4, 3)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc1_bn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 32)
        self.fc2_bn = nn.BatchNorm1d(32)
        self.fc3_steer = nn.Linear(32, num_actions_steer)
        self.fc3_throttle = nn.Linear(32, num_actions_throttle)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)  # aggiunge una dimensione batch se necessario
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = x.view(-1, 64 * 8 * 8)  # flatten
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.relu(self.fc2_bn(self.fc2(x)))

        steer_output = self.fc3_steer(x)
        throttle_output = self.fc3_throttle(x)

        return steer_output, throttle_output


class DQN(object):
    def __init__(
            self,
            num_actions_steer,
            num_action_throttle,
            state_dim,  # ?
            in_channels,
            device,
            discount=0.9,
            optimizer="Adam",
            optimizer_parameters=None,
            target_update_frequency=1e4,
            initial_eps=1,
            end_eps=0.05,
            eps_decay_period=25e4,
            eval_eps=0.001
    ) -> None:
        self.current_eps = None
        if optimizer_parameters is None:
            optimizer_parameters = {'lr': 0.01}
        self.device = device

        self.Q = ConvNet(in_channels, num_actions_steer, num_action_throttle).to(self.device)
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
        self.num_actions_steer = num_actions_steer
        self.num_actions_throttle = num_action_throttle

        self.iterations = 0

    def select_action(self, state, eval=False):
        eps = self.eval_eps if eval else max(self.slope * self.iterations + self.initial_eps, self.end_eps)
        self.current_eps = eps

        if np.random.uniform(0, 1) > eps:
            self.Q.eval()
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                steer, throttle = self.Q(state_tensor)
                steer_action = int(steer.argmax(1))
                throttle_action = int(throttle.argmax(1))
        else:
            steer_action = np.random.randint(self.num_actions_steer)
            throttle_action = np.random.randint(self.num_actions_throttle)
        return steer_action, throttle_action

    def train(self, replay_buffer):
        self.Q.train()
        # Sample mininbatch from replay buffer
        state, steer, throttle, next_state, reward, done = replay_buffer.sample()

        with torch.no_grad():
            next_actions_steer = self.Q(next_state)[0].argmax(1, keepdim=True)
            next_actions_throttle = self.Q(next_state)[1].argmax(1, keepdim=True)
            target_Q_steer = reward + (1 - done) * self.discount * self.Q_target(next_state)[0].gather(1,
                                                                                                       next_actions_steer)

            target_Q_throttle = reward + (1 - done) * self.discount * self.Q_target(next_state)[1].gather(1,
                                                                                                          next_actions_throttle)

        current_Q_steer, current_Q_throttle = self.Q(state)

        # Compute Q loss
        steer = steer.long().view(-1, 1)
        throttle = throttle.long().view(-1, 1)

        current_Q_steer = current_Q_steer.gather(1, steer).squeeze(1)
        current_Q_throttle = current_Q_throttle.gather(1, throttle).squeeze(1)

        Q_loss_steer = F.smooth_l1_loss(current_Q_steer, target_Q_steer.squeeze())
        Q_loss_throttle = F.smooth_l1_loss(current_Q_throttle, target_Q_throttle.squeeze())

        # Optimize the Q
        self.Q_optimizer.zero_grad()
        (Q_loss_steer + Q_loss_throttle).backward()
        self.Q_optimizer.step()

        # Update target network by full copy every X iterations.
        self.iterations += 1
        self.copy_target_update()

    def copy_target_update(self):
        if self.iterations % self.target_update_frequency == 0:
            print('target network updated')
            print('current epsilon', self.current_eps)
            self.Q_target.load_state_dict(self.Q.state_dict())

    def save(self, filename):
        torch.save(self.Q.state_dict(), filename + "_Q")
        torch.save(self.Q_optimizer.state_dict(), filename + "_optimizer")

    def load(self, filename):
        self.Q.load_state_dict(torch.load(filename + "_Q"))
        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer.load_state_dict(torch.load(filename + "_optimizer"))
