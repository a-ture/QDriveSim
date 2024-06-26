import os

from DQN_Control.replay_buffer import ReplayBuffer
from DQN_Control.model import DQN
from config import action_map_steer, env_params, action_map_throttle, action_map_brake
from environment import SimEnv


def run():
    try:
        # Definizione dei parametri
        buffer_size = 1e4  # Dimensione del replay buffer
        batch_size = 32  # Dimensione del batch per il test
        state_dim = (5, 128, 128)  # Dimensione dello stato (immagine 128x128)
        device = "cuda"  # Dispositivo su cui eseguire il modello ()
        num_actions_steer = len(action_map_steer)  # Numero di azioni disponibili
        num_actions_throttle = len(action_map_throttle)  # Numero di azioni disponibili
        num_actions_brake = len(action_map_brake)  # Numero di azioni disponibili
        in_channels = 5  # Numero di canali dell'immagine (scala di grigi)
        episodes = 100  # Numero di episodi da eseguire per il test

        replay_buffer = ReplayBuffer(state_dim, batch_size, buffer_size, device)

        # Creazione del modello DQN
        model = DQN(num_actions_steer, num_actions_brake, num_actions_throttle, state_dim, in_channels, device)

        # Caricamento dei pesi del modello addestrato
        model.load('weights/model_ep_900')

        # Impostazione dell'ambiente di simulazione
        env = SimEnv(visuals=True, **env_params)

        for ep in range(episodes):
            env.create_actors()  # Creazione degli attori nell'ambiente
            env.generate_episode(model, replay_buffer, ep,
                                 evaluation=True)  # Esecuzione dell'episodio in modalità di valutazione
            env.reset()  # Reimpostazione dell'ambiente per il prossimo episodio
    finally:
        env.reset()
        env.quit()


if __name__ == "__main__":
    run()
