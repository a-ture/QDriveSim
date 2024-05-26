import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, state_dim, batch_size, buffer_size, device) -> None:
        self.batch_size = batch_size
        self.max_size = int(buffer_size)
        self.device = device
        self.state_dim = state_dim
        self.ptr = 0
        self.crt_size = 0

        self.state = np.zeros((self.max_size,) + self.state_dim)
        self.steer = np.zeros((self.max_size, 1))
        self.brake = np.zeros((self.max_size, 1))
        self.throttle = np.zeros((self.max_size, 1))
        self.next_state = np.array(self.state)
        self.reward = np.zeros((self.max_size, 1))
        self.done = np.zeros((self.max_size, 1))

    def add(self, state, steer, brake, throttle, next_state, reward, done):
        self.state[self.ptr] = state
        self.steer[self.ptr] = steer
        self.brake[self.ptr] = brake
        self.throttle[self.ptr] = throttle
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.crt_size = min(self.crt_size + 1, self.max_size)

    def sample(self):
        ind = np.random.randint(0, self.crt_size, size=self.batch_size)
        states = torch.FloatTensor(self.state[ind]).to(self.device)
        steers = torch.FloatTensor(self.steer[ind]).to(self.device)
        brakes = torch.FloatTensor(self.brake[ind]).to(self.device)
        throttles = torch.FloatTensor(self.throttle[ind]).to(self.device)
        next_states = torch.FloatTensor(self.next_state[ind]).to(self.device)
        rewards = torch.FloatTensor(self.reward[ind]).to(self.device)
        dones = torch.FloatTensor(self.done[ind]).to(self.device)

        return states, steers, brakes, throttles, next_states, rewards, dones