import torch
import copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DuelingConvNetLSTM(nn.Module):
    def __init__(self, in_channels, num_actions_steer, num_actions_brake, num_actions_throttle) -> None:
        super(DuelingConvNetLSTM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 8, 4)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, 2)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, 1)
        self.conv4_bn = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 128, 3, 1)
        self.conv5_bn = nn.BatchNorm2d(128)

        # Strati aggiuntivi
        self.conv6 = nn.Conv2d(128, 256, 3, 1)
        self.conv6_bn = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 256, 3, 1)
        self.conv7_bn = nn.BatchNorm2d(256)

        # Calcola la dimensione dell'output dopo i livelli convoluzionali
        self.lstm_input_size = self._get_conv_output_size(in_channels)

        self.lstm = nn.LSTM(self.lstm_input_size, 512, batch_first=True)

        self.fc1 = nn.Linear(512, 256)
        self.fc1_bn = nn.BatchNorm1d(256)

        self.value_fc = nn.Linear(256, 128)
        self.value_fc_bn = nn.BatchNorm1d(128)
        self.value = nn.Linear(128, 1)

        self.adv_fc = nn.Linear(256, 128)
        self.adv_fc_bn = nn.BatchNorm1d(128)
        self.adv_steer = nn.Linear(128, num_actions_steer)
        self.adv_brake = nn.Linear(128, num_actions_brake)
        self.adv_throttle = nn.Linear(128, num_actions_throttle)

    def _get_conv_output_size(self, in_channels):
        # Funzione per calcolare la dimensione dell'output dopo i livelli convoluzionali
        dummy_input = torch.zeros(1, in_channels, 256, 256)
        x = F.relu(self.conv1_bn(self.conv1(dummy_input)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = F.relu(self.conv5_bn(self.conv5(x)))
        x = F.relu(self.conv6_bn(self.conv6(x)))  # Strato aggiuntivo
        x = F.relu(self.conv7_bn(self.conv7(x)))  # Strato aggiuntivo
        return int(np.prod(x.size()))

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = F.relu(self.conv5_bn(self.conv5(x)))
        x = F.relu(self.conv6_bn(self.conv6(x)))  # Strato aggiuntivo
        x = F.relu(self.conv7_bn(self.conv7(x)))  # Strato aggiuntivo
        x = x.view(batch_size, -1)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x.unsqueeze(1))
        x = F.relu(self.fc1_bn(self.fc1(x[:, -1, :])))

        value = F.relu(self.value_fc_bn(self.value_fc(x)))
        value = self.value(value)

        adv_steer = F.relu(self.adv_fc_bn(self.adv_fc(x)))
        adv_steer = self.adv_steer(adv_steer)

        adv_brake = F.relu(self.adv_fc_bn(self.adv_fc(x)))
        adv_brake = self.adv_brake(adv_brake)

        adv_throttle = F.relu(self.adv_fc_bn(self.adv_fc(x)))
        adv_throttle = self.adv_throttle(adv_throttle)

        q_steer = value + adv_steer - adv_steer.mean(1, keepdim=True)
        q_brake = value + adv_brake - adv_brake.mean(1, keepdim=True)
        q_throttle = value + adv_throttle - adv_throttle.mean(1, keepdim=True)

        return q_steer, q_brake, q_throttle


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
            optimizer_parameters=None,
            target_update_frequency=1e4,
            initial_eps=1,
            end_eps=0.05,
            eps_decay_period=25e4,
            eval_eps=0.001
    ) -> None:
        if optimizer_parameters is None:
            optimizer_parameters = {'lr': 0.01}
        self.current_eps = None
        self.device = device

        self.Q = DuelingConvNetLSTM(in_channels, num_actions_steer, num_action_brake, num_action_throttle).to(
            self.device)
        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer = getattr(torch.optim, optimizer)(self.Q.parameters(), **optimizer_parameters)

        self.lr = optimizer_parameters['lr']

        self.discount = discount
        self.target_update_frequency = target_update_frequency

        self.initial_eps = initial_eps
        self.end_eps = end_eps
        self.slope = (self.end_eps - self.initial_eps) / (eps_decay_period ** 2)

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

        self.lr = max(self.lr - 0.00001, 0.00001)
        for param_group in self.Q_optimizer.param_groups:
            param_group['lr'] = self.lr

        state, steer, brake, throttle, next_state, reward, done = replay_buffer.sample()

        with torch.no_grad():
            next_actions_steer = self.Q(next_state)[0].argmax(1, keepdim=True)
            next_actions_brake = self.Q(next_state)[1].argmin(1, keepdim=True)
            next_actions_throttle = self.Q(next_state)[2].argmax(1, keepdim=True)
            target_Q_steer = reward + (1 - done) * self.discount * self.Q_target(next_state)[0].gather(1,
                                                                                                       next_actions_steer)
            target_Q_brake = reward + (1 - done) * self.discount * self.Q_target(next_state)[1].gather(1,
                                                                                                       next_actions_brake)
            target_Q_throttle = reward + (1 - done) * self.discount * self.Q_target(next_state)[2].gather(1,
                                                                                                          next_actions_throttle)

        current_Q_steer, current_Q_brake, current_Q_throttle = self.Q(state)

        steer = steer.long().view(-1, 1)
        brake = brake.long().view(-1, 1)
        throttle = throttle.long().view(-1, 1)

        current_Q_steer = current_Q_steer.gather(1, steer).squeeze(1)
        current_Q_brake = current_Q_brake.gather(1, brake).squeeze(1)
        current_Q_throttle = current_Q_throttle.gather(1, throttle).squeeze(1)

        td_error_steer = target_Q_steer - current_Q_steer
        td_error_brake = target_Q_brake - current_Q_brake
        td_error_throttle = target_Q_throttle - current_Q_throttle

        Q_loss_steer = td_error_steer.pow(2).mean()
        Q_loss_brake = td_error_brake.pow(2).mean()
        Q_loss_throttle = td_error_throttle.pow(2).mean()

        self.Q_optimizer.zero_grad()
        (Q_loss_steer + Q_loss_brake + Q_loss_throttle).backward()
        self.Q_optimizer.step()

        self.iterations += 1
        self.copy_target_update()

    def copy_target_update(self):
        # Aumenta la frequenza di aggiornamento della rete target
        if self.iterations % (self.target_update_frequency // 2) == 0:
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
