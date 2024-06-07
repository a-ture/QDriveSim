import os

import torch

from DQN_Control import model_binary
from DQN_Control.replay_buffer import ReplayBuffer

from config import action_map, env_params
from utils import *
from environment import SimEnv


# rete non binaria 5sensori
def run():
    try:
        buffer_size = 1e4
        batch_size = 32
        state_dim = (5, 128, 128)
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = "cpu"
        num_actions = len(action_map)
        in_channels = 5
        episodes = 100

        replay_buffer = ReplayBuffer(state_dim, batch_size, buffer_size, device)
        model = DQN(num_actions, state_dim, in_channels, device)

        # this only works if you have a model in your weights folder. Replace this by that file
        model.load('weights_non_binary_da_binary/model_ep_1000')

        # set to True if you want to run with pygame
        env = SimEnv(visuals=True, **env_params)

        for ep in range(episodes):
            env.create_actors()
            env.generate_episode(model, replay_buffer, ep, action_map, eval=True)
            env.reset()
    finally:
        env.reset()
        env.quit()


# rete binaria
def run_binary():
    try:
        buffer_size = 1e4
        batch_size = 32
        state_dim = (5, 128, 128)
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = "cpu"
        num_actions = len(action_map)
        in_channels = 5
        episodes = 100

        replay_buffer = ReplayBuffer(state_dim, batch_size, buffer_size, device)
        model = model_binary.DQN(num_actions, state_dim, in_channels, device)

        # this only works if you have a model in your weights folder. Replace this by that file
        model.load('weights/model_ep_1000')

        # set to True if you want to run with pygame
        env = SimEnv(visuals=True, **env_params)

        for ep in range(episodes):
            env.create_actors()
            env.generate_episode(model, replay_buffer, ep, action_map, eval=True)
            env.reset()
    finally:
        env.reset()


if __name__ == "__main__":
    run_binary()
