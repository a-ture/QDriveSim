import torch
import copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, dim, in_channels, num_actions_steer, num_actions_brake, num_actions_throttle) -> None:
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 8, 4)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 4, 3)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 4 * 8 * 8, 256)
        self.fc1_bn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 32)
        self.fc2_bn = nn.BatchNorm1d(32)
        self.fc3_steer = nn.Linear(32, num_actions_steer)
        self.fc3_brake = nn.Linear(32, num_actions_brake)
        self.fc3_throttle = nn.Linear(32, num_actions_throttle)

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 8 * 8)  # flatten
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.relu(self.fc2_bn(self.fc2(x)))

        steer_output = self.fc3_steer(x)
        brake_output = self.fc3_brake(x)
        throttle_output = self.fc3_throttle(x)

        return steer_output, brake_output, throttle_output

class DQN(object):
    def __init__(
            self,
            num_actions_steer,
            num_action_brake,
            num_action_throttle,
            state_dim,
            in_channels,
            device,
            discount=0.9,
            optimizer="Adam",
            optimizer_parameters={'lr': 0.01},
            target_update_frequency=1e4,
            initial_eps=1,
            end_eps=0.05,
            eps_decay_period=25e4,
            eval_eps=0.001
    ) -> None:
        self.current_eps = None
        self.device = device

        self.Q = ConvNet(state_dim, in_channels, num_actions_steer, num_action_brake, num_action_throttle).to(
            self.device)
        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer = getattr(torch.optim, optimizer)(self.Q.parameters(), **optimizer_parameters)

        self.discount = discount
        self.target_update_frequency = target_update_frequency

        self.initial_eps = initial_eps
        self.end_eps = end_eps
        self.slope = (self.end_eps - self.initial_eps) / eps_decay_period

        self.state_shape = (-1,) + state_dim
        self.eval_eps = eval_eps
        self.num_actions_steer = num_actions_steer
        self.num_actions_brake = num_action_brake
        self.num_actions_throttle = num_action_throttle

        self.iterations = 0

    def select_action(self, state, eval=False):
        eps = self.eval_eps if eval \
            else max(self.slope * self.iterations + self.initial_eps, self.end_eps)
        self.current_eps = eps

        if np.random.uniform(0, 1) > eps:
            self.Q.eval()
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                steer, brake, throttle = self.Q(state_tensor)
                steer_action = int(steer.argmax(1))
                brake_action = int(brake.argmin(1))
                throttle_action = int(throttle.argmax(1))
        else:
            steer_action = np.random.randint(self.num_actions_steer)
            brake_action = np.random.randint(self.num_actions_brake)
            throttle_action = np.random.randint(self.num_actions_throttle)
        return steer_action, brake_action, throttle_action

    def train(self, replay_buffer):
        self.Q.train()

        state, steer, brake, throttle, next_state, reward, done = replay_buffer.sample()

        with torch.no_grad():
            target_Q_steer, target_Q_brake, target_Q_throttle = self.Q_target(next_state)

            target_Q_steer_max = target_Q_steer.max(1, keepdim=True)[0]
            target_Q_brake_min = target_Q_brake.min(1, keepdim=True)[0]
            target_Q_throttle_max = target_Q_throttle.max(1, keepdim=True)[0]

            target_Q_steer = reward + (1 - done) * self.discount * target_Q_steer_max
            target_Q_brake = reward + (1 - done) * self.discount * target_Q_brake_min
            target_Q_throttle = reward + (1 - done) * self.discount * target_Q_throttle_max

        current_Q_steer, current_Q_brake, current_Q_throttle = self.Q(state)

        current_Q_steer = current_Q_steer.gather(1, steer.unsqueeze(1).long())
        current_Q_brake = current_Q_brake.gather(1, brake.unsqueeze(1).long())
        current_Q_throttle = current_Q_throttle.gather(1, throttle.unsqueeze(1).long())

        Q_loss_steer = F.smooth_l1_loss(current_Q_steer, target_Q_steer)
        Q_loss_brake = F.smooth_l1_loss(current_Q_brake, target_Q_brake)
        Q_loss_throttle = F.smooth_l1_loss(current_Q_throttle, target_Q_throttle)

        self.Q_optimizer.zero_grad()
        (Q_loss_steer + Q_loss_brake + Q_loss_throttle).backward()
        self.Q_optimizer.step()

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
