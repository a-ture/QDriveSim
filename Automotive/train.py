import os
import torch
import pandas as pd
import time
from DQN_Control.replay_buffer import ReplayBuffer
from DQN_Control.model import DQN
from config import action_map_steer, env_params, action_map_throttle, action_map_brake
from environment import SimEnv
from codecarbon import OfflineEmissionsTracker

from logger import setup_logger, close_loggers, log_params, log_codecarbon_metrics, write_separator
from utils import create_folders


# Funzione principale per eseguire l'addestramento del modello
def run(logger):
    try:
        start_time = time.time()  # Registra il tempo di inizio

        # Definizione dei parametri del modello
        buffer_size = 1e4  # Dimensione del replay buffer
        batch_size = 128  # Dimensione del batch per l'addestramento
        state_dim = (4, 128, 128)  # Dimensione dello stato Da cambiare in base al numero di sensori
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  # Dispositivo su cui eseguire il modello (GPU se disponibile, altrimenti CPU)
        num_actions_steer = len(action_map_steer)  # Numero di azioni disponibili
        num_actions_brake = len(action_map_brake)  # Numero di azioni disponibili
        num_actions_throttle = len(action_map_throttle)  # Numero di azioni disponibili

        in_channels = 4  # da cambiare in base al numero di sensori e al colore delle img

        model_params = {
            'num_actions_steer': num_actions_steer,
            'num_action_brake': num_actions_brake,
            'num_action_throttle': num_actions_throttle,
            'state_dim': state_dim,
            'in_channels': in_channels,
            'device': device,
            'discount': 0.9,
            'optimizer': "Adam",
            'optimizer_parameters': {'lr': 0.01},
            'target_update_frequency': 1e4,
            'initial_eps': 1,
            'end_eps': 0.05,
            'eps_decay_period': 25e4,
            'eval_eps': 0.001,
            'learning_rate': 0.1,
        }

        # Log dei parametri dell'ambiente e del modello
        log_params(logger, env_params, title="Ambiente - Parametri")
        write_separator(logger)
        log_params(logger, model_params, title="Modello - Parametri")

        # Creazione del replay buffer
        replay_buffer = ReplayBuffer(state_dim, batch_size, buffer_size, device)

        # Creazione del modello DQN
        model = DQN(num_actions_steer, num_actions_brake, num_actions_throttle, state_dim, in_channels, device)

        # Creazione dell'ambiente di simulazione
        env = SimEnv(visuals=True, **env_params)

        # Ciclo di addestramento per un numero di episodi definito
        episodes = 1300

        for ep in range(episodes):
            # Creazione degli attori nell'ambiente
            env.create_actors()
            # Generazione dell'episodio e addestramento del modello
            env.generate_episode(model, replay_buffer, ep, eval=False)
            # Reimpostazione dell'ambiente per il prossimo episodio
            env.reset()

        end_time = time.time()  # Registra il tempo di fine
        total_training_time = end_time - start_time  # Calcola il tempo totale di esecuzione
        logger.info(f"Tempo totale di addestramento: {total_training_time:.2f} secondi")

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
        run(logger)
    finally:
        tracker.stop()
        emissions_csv = pd.read_csv("emissions.csv")

        last_emissions = emissions_csv.tail(1)  # Ottenere l'ultima riga del dataframe
        emissions = last_emissions["emissions"].iloc[0] * 1000  # Estrai il valore numerico dall'ultima riga

        energy = last_emissions["energy_consumed"]
        cpu = last_emissions["cpu_energy"]
        gpu = last_emissions["gpu_energy"]
        ram = last_emissions["ram_energy"]

        write_separator(logger)

        # Log delle metriche
        logger.info(f"Emissioni: {emissions} g")
        logger.info(f"Energia consumata: {energy} kWh")
        logger.info(f"Energia CPU: {cpu} J")
        logger.info(f"Energia GPU: {gpu} J")
        logger.info(f"Energia RAM: {ram} J")

        # Log delle metriche di CodeCarbon
        log_codecarbon_metrics(logger, emissions)
        close_loggers([logger])
        del logger
