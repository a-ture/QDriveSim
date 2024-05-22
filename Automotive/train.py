import os
import torch

from DQN_Control.replay_buffer import ReplayBuffer
from DQN_Control.model import DQN
from config import action_map_steer, env_params, action_map_throttle, action_map_brake
from environment import SimEnv
from codecarbon import OfflineEmissionsTracker

from logger import setup_logger, close_loggers
from utils import create_folders


# Funzione principale per eseguire l'addestramento del modello
def run():
    try:
        # Definizione dei parametri
        buffer_size = 1e4  # Dimensione del replay buffer
        batch_size = 32  # Dimensione del batch per l'addestramento
        state_dim = (3, 128, 128)  # Dimensione dello stato Da cambiare in base al numero di sensori
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  # Dispositivo su cui eseguire il modello (GPU se disponibile, altrimenti CPU)
        num_actions_steer = len(action_map_steer)  # Numero di azioni disponibili
        num_actions_brake = len(action_map_brake)  # Numero di azioni disponibili
        num_actions_throttle = len(action_map_throttle)  # Numero di azioni disponibili

        in_channels = 3  # da cambiare in base al numero di sensori e al colore delle img

        # Creazione del replay buffer
        replay_buffer = ReplayBuffer(state_dim, batch_size, buffer_size, device)

        # Creazione del modello DQN
        model = DQN(num_actions_steer, num_actions_brake, num_actions_throttle, state_dim, in_channels, device)

        # Creazione dell'ambiente di simulazione
        env = SimEnv(visuals=True, **env_params)

        # Ciclo di addestramento per un numero di episodi definito
        episodes = 300
        for ep in range(episodes):
            # Creazione degli attori nell'ambiente
            env.create_actors()
            # Generazione dell'episodio e addestramento del modello
            # TODO qui facciamo restituire cosa le cose da salvare nel log per ogni eps
            env.generate_episode(model, replay_buffer, ep, eval=False)
            # Reimpostazione dell'ambiente per il prossimo episodio
            env.reset()
    finally:
        # Chiusura dell'ambiente alla fine dell'esecuzione
        env.quit()


# Esecuzione della funzione run() se questo modulo è eseguito come script principale
if __name__ == "__main__":
    create_folders(['log'])
    logger = setup_logger('logger', os.path.join('log', 'logger.log'))

    tracker = OfflineEmissionsTracker(country_iso_code="ITA")
    tracker.start()

    try:
        run()
    finally:
        emissions = tracker.stop()

        close_loggers([logger])
        del logger
