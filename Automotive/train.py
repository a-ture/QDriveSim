import torch  # Importa PyTorch, una libreria per il machine learning

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


# Funzione principale per eseguire l'addestramento del modello
def run():
    try:
        # Definizione dei parametri
        buffer_size = 1e4  # Dimensione del replay buffer
        batch_size = 32  # Dimensione del batch per l'addestramento
        state_dim = (128, 128)  # Dimensione dello stato (immagine 128x128)
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  # Dispositivo su cui eseguire il modello (GPU se disponibile, altrimenti CPU)
        num_actions = len(action_map)  # Numero di azioni disponibili
        in_channels = 1  # Numero di canali dell'immagine (scala di grigi)

        # Creazione del replay buffer
        replay_buffer = ReplayBuffer(state_dim, batch_size, buffer_size, device)

        # Creazione del modello DQN
        model = DQN(num_actions, state_dim, in_channels, device)

        # Creazione dell'ambiente di simulazione
        env = SimEnv(visuals=False, **env_params)

        # Ciclo di addestramento per un numero di episodi definito
        episodes = 10000
        for ep in range(episodes):
            # Creazione degli attori nell'ambiente
            env.create_actors()
            # Generazione dell'episodio e addestramento del modello
            env.generate_episode(model, replay_buffer, ep, action_map, eval=False)
            # Reimpostazione dell'ambiente per il prossimo episodio
            env.reset()
    finally:
        # Chiusura dell'ambiente alla fine dell'esecuzione
        env.quit()


# Esecuzione della funzione run() se questo modulo è eseguito come script principale
if __name__ == "__main__":
    run()
