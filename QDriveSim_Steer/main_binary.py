import os

from DQN_Control.binrarized_model import DQN
from DQN_Control.replay_buffer import ReplayBuffer
from config import action_map, env_params

from environment import SimEnv



def binarize(tensor):
    return tensor.sign()


def binarize_model_weights(model):
    for name, param in model.named_parameters():
        if 'weight' in name:
            param.data = binarize(param.data)

#binarizzazione dei pesi
def run():
    try:
        buffer_size = 1e4
        batch_size = 32
        state_dim = (5, 128, 128)
        device = "cpu"
        num_actions = len(action_map)
        in_channels = 5
        episodes = 100

        replay_buffer = ReplayBuffer(state_dim, batch_size, buffer_size, device)
        model = DQN(num_actions, state_dim, in_channels, device, use_binarized=True)

        # Load pre-trained model weights
        model.load('weights_non_binary_da_binary/model_ep_1000')

        # Binarize the weights in the model
        binarize_model_weights(model.Q)
        binarize_model_weights(model.Q_target)

        # Set to True if you want to run with pygame
        env = SimEnv(visuals=True, **env_params)

        for ep in range(episodes):
            env.create_actors()
            env.generate_episode(model, replay_buffer, ep, action_map, eval=True)
            env.reset()
    finally:
        env.reset()
        env.quit()

if __name__ == "__main__":
    run()


