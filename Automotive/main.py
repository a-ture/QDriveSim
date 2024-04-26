import os  # Importa il modulo os per operazioni di sistema

# Importa la classe ReplayBuffer dalla cartella DQN_Control
from DQN_Control.replay_buffer import ReplayBuffer
# Importa la classe DQN dalla cartella DQN_Control
from DQN_Control.model import DQN
# Importa action_map e env_params dalla configurazione
from config import action_map, env_params
# Importa funzioni di utilità dalla cartella utils
from utils import *
# Importa la classe SimEnv dall'ambiente
from environment import SimEnv


# Funzione principale per eseguire il test del modello addestrato
def run():
    try:
        # Definizione dei parametri
        buffer_size = 1e4  # Dimensione del replay buffer
        batch_size = 32  # Dimensione del batch per il test
        state_dim = (128, 128)  # Dimensione dello stato (immagine 128x128)
        device = "cpu"  # Dispositivo su cui eseguire il modello (CPU)
        num_actions = len(action_map)  # Numero di azioni disponibili
        in_channels = 1  # Numero di canali dell'immagine (scala di grigi)
        episodes = 10000  # Numero di episodi da eseguire per il test

        # Creazione del replay buffer
        replay_buffer = ReplayBuffer(state_dim, batch_size, buffer_size, device)

        # Creazione del modello DQN
        model = DQN(num_actions, state_dim, in_channels, device)

        # Caricamento dei pesi del modello addestrato
        model.load('weights/model_ep_4400')

        # Impostazione dell'ambiente di simulazione
        env = SimEnv(visuals=True, **env_params)

        # Ciclo per eseguire i test su più episodi
        for ep in range(episodes):
            env.create_actors()  # Creazione degli attori nell'ambiente
            env.generate_episode(model, replay_buffer, ep, action_map,
                                 eval=True)  # Esecuzione dell'episodio in modalità di valutazione
            env.reset()  # Reimpostazione dell'ambiente per il prossimo episodio
    finally:
        env.reset()  # Reimpostazione dell'ambiente alla fine del test
        env.quit()  # Chiusura dell'ambiente


# Esecuzione della funzione run() se questo modulo è eseguito come script principale
if __name__ == "__main__":
    run()
