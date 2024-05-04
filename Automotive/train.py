import torch

from DQN_Control.replay_buffer import ReplayBuffer
from DQN_Control.model import DQN
from config import action_map_steer, env_params, action_map_throttle, action_map_brake
from environment import SimEnv


# Funzione principale per eseguire l'addestramento del modello
def run():
    try:
        # Definizione dei parametri
        buffer_size = 1e4  # Dimensione del replay buffer
        batch_size = 128  # Dimensione del batch per l'addestramento (quante azioni fa contemporaneamente)
        state_dim = (128, 128)  # Dimensione dello stato (immagine 128x128)
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  # Dispositivo su cui eseguire il modello (GPU se
        # disponibile, altrimenti CPU)
        num_actions_steer = len(action_map_steer)  # Numero di azioni disponibili
        num_actions_brake = len(action_map_brake)  # Numero di azioni disponibili
        num_actions_throttle = len(action_map_throttle)  # Numero di azioni disponibili

        in_channels = 3

        # Creazione del replay buffer
        replay_buffer = ReplayBuffer(state_dim, batch_size, buffer_size, device)

        # Creazione del modello DQN
        model = DQN(num_actions_steer, num_actions_brake, num_actions_throttle, state_dim, in_channels, device)

        # Creazione dell'ambiente di simulazione
        env = SimEnv(visuals=True, **env_params)

        # Ciclo di addestramento per un numero di episodi definito
        episodes = 1500
        for ep in range(episodes):
            # Creazione degli attori nell'ambiente
            env.create_actors()
            # Generazione dell'episodio e addestramento del modello
            env.generate_episode(model, replay_buffer, ep, action_map_steer, eval=False)
            # Reimpostazione dell'ambiente per il prossimo episodio
            env.reset()
    finally:
        # Chiusura dell'ambiente alla fine dell'esecuzione
        env.quit()


# Esecuzione della funzione run() se questo modulo è eseguito come script principale
if __name__ == "__main__":
    run()
