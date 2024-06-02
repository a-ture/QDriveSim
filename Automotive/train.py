import os

import numpy as np
import torch

import time

from DQN_Control.model import DQN
from DQN_Control.replay_buffer import ReplayBuffer
from config import action_map_steer, env_params, action_map_throttle, action_map_brake
from environment import SimEnv
from codecarbon import OfflineEmissionsTracker

from logger import setup_logger, close_loggers, log_params, write_separator
from utils import create_folders

# Funzione principale per eseguire l'addestramento del modello
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt


def plot_rewards(episode_rewards, avg_rewards):
    plt.figure(figsize=(12, 6))
    plt.plot(episode_rewards, label='Episode Rewards')
    plt.plot(avg_rewards, label='Average Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('Episode and Average Rewards Over Time')
    plt.legend()
    plt.show()


def run(logger):
    try:
        start_time = time.time()  # Registra il tempo di inizio

        # Definizione dei parametri del modello
        buffer_size = 1e4  # Dimensione del replay buffer
        batch_size = 32  # Dimensione del batch per l'addestramento
        state_dim = (1, 128, 128)  # Dimensione dello stato Da cambiare in base al numero di sensori
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  # Dispositivo su cui eseguire il modello (GPU se disponibile, altrimenti CPU)
        num_actions_steer = len(action_map_steer)  # Numero di azioni disponibili
        num_actions_brake = len(action_map_brake)  # Numero di azioni disponibili
        num_actions_throttle = len(action_map_throttle)  # Numero di azioni disponibili

        in_channels = 1  # da cambiare in base al numero di sensori e al colore delle img

        model_params = {
            'num_actions_steer': num_actions_steer,
            'num_action_brake': num_actions_brake,
            'num_action_throttle': num_actions_throttle,
            'state_dim': state_dim,
            'in_channels': in_channels,
            'device': device,
            'discount': 0.9,
            'optimizer': "Adam",
            'optimizer_parameters': {'lr': 0.005},
            'target_update_frequency': 5e3,
            'initial_eps': 1,
            'end_eps': 0.1,
            'eps_decay_period': 50e4,
            'eval_eps': 0.001,
        }

        # Log dei parametri dell'ambiente e del modello
        log_params(logger, env_params, title="Ambiente - Parametri")
        write_separator(logger)
        log_params(logger, model_params, title="Modello - Parametri")

        # Creazione del replay buffer
        replay_buffer = ReplayBuffer(state_dim, batch_size, buffer_size, device)

        # Creazione del modello DQN
        model = DQN(num_actions_steer, num_actions_brake, num_actions_throttle, state_dim, in_channels, device)
        # Caricamento dei pesi del modello addestrato
        # model.load('weights/model_ep_950')

        # Creazione dell'ambiente di simulazione
        env = SimEnv(visuals=True, **env_params)

        # Ciclo di addestramento per un numero di episodi definito
        episodes = 3000
        episode_rewards = []
        avg_rewards = []
        avg_window = 100  # Finestra per la media mobile

        for ep in range(episodes):
            # Creazione degli attori nell'ambiente
            env.create_actors()

            # Generazione dell'episodio e addestramento del modello
            episode_reward = env.generate_episode(model, replay_buffer, ep, evaluation=False)

            # Controlla se episode_reward è un valore numerico
            if episode_reward is not None:
                episode_rewards.append(episode_reward)

                # Calcola la ricompensa media
                if len(episode_rewards) >= avg_window:
                    avg_rewards.append(np.mean(episode_rewards[-avg_window:]))
                else:
                    avg_rewards.append(np.mean(episode_rewards))

            # Reimpostazione dell'ambiente per il prossimo episodio
            env.reset()

        end_time = time.time()  # Registra il tempo di fine
        total_training_time = end_time - start_time  # Calcola il tempo totale di esecuzione
        logger.info(f"Tempo totale di addestramento: {total_training_time:.2f} secondi")

        # Plot delle ricompense
        plot_rewards(episode_rewards, avg_rewards)

    finally:
        # Chiusura dell'ambiente alla fine dell'esecuzione
        env.quit()


if __name__ == "__main__":
    create_folders(['log'])

    logger = setup_logger('logger', os.path.join('log', 'logger.log'))

    try:
        run(logger)
    finally:
        close_loggers([logger])
        del logger
