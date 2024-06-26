

from DQN_Control import model_binary, model_layer_binary
from DQN_Control.model import DQN
from DQN_Control.replay_buffer import ReplayBuffer

from config import action_map, env_params

from environment import SimEnv


def run_layer_binary():
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
        model = model_layer_binary.DQN(num_actions, state_dim, in_channels, device)

        # this only works if you have a model in your weights folder. Replace this by that file
        model.load('pesi/weights_layerbinari_town3_1400/model_ep_1400')

        # set to True if you want to run with pygame
        env = SimEnv(visuals=True, **env_params)

        for ep in range(episodes):
            env.create_actors()
            env.generate_episode(model, replay_buffer, ep, action_map, eval=True)
            env.reset()
    finally:
        env.reset()

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
        model.load('pesi/weights_5sensori_town3/model_ep_1000')

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
        model.load('pesi/weights_retetuttabianaria_town3/model_ep_1000')

        # set to True if you want to run with pygame
        env = SimEnv(visuals=True, **env_params)

        for ep in range(episodes):
            env.create_actors()
            env.generate_episode(model, replay_buffer, ep, action_map, eval=True)
            env.reset()
    finally:
        env.reset()


if __name__ == "__main__":
    run_layer_binary()


# Addestramento rete binaria: 14 ore di 1.093298 kWh  di cui: Energy consumed for RAM : 0.141213 kWh. RAM Power : 11.67989730834961 W
#                                                             Energy consumed for all GPUs : 0.438048 kWh. Total GPU Power : 35.15487607311413 W
#                                                             Energy consumed for all CPUs : 0.514038 kWh. Total CPU Power : 42.5 W
#
#
# Addestramento rete non binaria: 26 ore: 2.072642 kWh tot di cui : Energy consumed for RAM : 0.304728 kWh. RAM Power : 11.67989730834961 W
#                                                                   Energy consumed for all GPUs : 0.658512 kWh. Total GPU Power : 38.02060103683682 W
#                                                                   Energy consumed for all CPUs : 1.109402 kWh. Total CPU Power : 42.5 W

