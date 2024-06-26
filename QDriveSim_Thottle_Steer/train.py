import time

import pandas as pd
import torch
from codecarbon import OfflineEmissionsTracker

from DQN_Control.replay_buffer import ReplayBuffer
from DQN_Control.model import DQN

from config import env_params, action_map_throttle, action_map_brake, action_map_steer
from logger import log_params, write_separator, setup_logger, close_loggers
from utils import *
from environment import SimEnv


def run(logger):
    try:
        buffer_size = 1e4
        batch_size = 32
        state_dim = (1, 128, 128)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_actions_steer = len(action_map_steer)  # Numero di azioni disponibili
        num_actions_throttle = len(action_map_throttle)  # Numero di azioni disponibili
        in_channels = 1
        episodes = 10000

        model_params = {
            'num_actions_steer': num_actions_steer,
            'num_action_throttle': num_actions_throttle,
            'state_dim': state_dim,
            'in_channels': in_channels,
        }

        # Log dei parametri dell'ambiente e del modello
        log_params(logger, env_params, title="Ambiente - Parametri")
        write_separator(logger)
        log_params(logger, model_params, title="Modello - Parametri")

        replay_buffer = ReplayBuffer(state_dim, batch_size, buffer_size, device)
        model = DQN(num_actions_steer, num_actions_throttle, state_dim, in_channels, device)

        env = SimEnv(visuals=True, **env_params)

        for ep in range(episodes):
            env.create_actors()
            env.generate_episode(model, replay_buffer, ep, eval=False)
            env.reset()
    finally:

        env.quit()


if __name__ == "__main__":
    start_time = time.time()  # Registra il tempo di inizio
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
        # Log delle metriche
        logger.info(f"Emissioni: {emissions} g")
        logger.info(f"Energia consumata: {energy} kWh")
        logger.info(f"Energia CPU: {cpu} J")
        logger.info(f"Energia GPU: {gpu} J")
        logger.info(f"Energia RAM: {ram} J")

        end_time = time.time()  # Registra il tempo di fine
        total_training_time = end_time - start_time  # Calcola il tempo totale di esecuzione
        logger.info(f"Tempo totale di addestramento: {total_training_time:.2f} secondi")

        end_time = time.time()  # Registra il tempo di fine
        total_training_time = end_time - start_time  # Calcola il tempo totale di esecuzione
        logger.info(f"Tempo totale di addestramento: {total_training_time:.2f} secondi")
        close_loggers([logger])
        del logger
