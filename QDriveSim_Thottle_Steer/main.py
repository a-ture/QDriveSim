import torch

from DQN_Control.replay_buffer import ReplayBuffer
from DQN_Control.model import DQN

from config import env_params, action_map_steer, action_map_throttle

from environment import SimEnv


def run():
    try:
        buffer_size = 1e4
        batch_size = 32
        state_dim = (1, 128, 128)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_actions_steer = len(action_map_steer)  # Numero di azioni disponibili
        num_actions_throttle = len(action_map_throttle)  # Numero di azioni disponibili
        in_channels = 1
        episodes = 10000
        replay_buffer = ReplayBuffer(state_dim, batch_size, buffer_size, device)

        model = DQN(num_actions_steer, num_actions_throttle, state_dim, in_channels, device)

        # this only works if you have a model in your weights folder. Replace this by that file
        model.load('weights/model_ep_1000')

        # set to True if you want to run with pygame
        env = SimEnv(visuals=True, **env_params)

        for ep in range(episodes):
            env.create_actors()
            env.generate_episode(model, replay_buffer, ep, eval=True)
            env.reset()
    finally:
        env.reset()
        env.quit()


if __name__ == "__main__":
    run()
