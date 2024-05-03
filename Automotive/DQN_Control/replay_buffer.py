import cv2  # Importa il modulo OpenCV per il lavoro con immagini
import torch  # Importa PyTorch, una libreria per il machine learning
import numpy as np  # Importa NumPy, una libreria per il calcolo numerico
from torchvision import transforms  # Importa il modulo torchvision per trasformazioni delle immagini


class ReplayBuffer(object):
    def __init__(self, state_dim, batch_size, buffer_size, device) -> None:
        # Inizializza il ReplayBuffer
        self.batch_size = batch_size  # Dimensione del batch
        self.max_size = int(buffer_size)  # Dimensione massima del buffer
        self.device = device  # Dispositivo su cui lavorare (CPU o GPU)

        self.ptr = 0  # Puntatore all'ultimo dato memorizzato
        self.crt_size = 0  # Dimensione corrente del buffer

        # Inizializza i buffer per memorizzare stati, azioni, prossimi stati, ricompense e flag di terminazione
        self.state = np.zeros((self.max_size,) + state_dim)
        self.steer = np.zeros((self.max_size, 1))  # Buffer per l'azione di sterzo
        self.brake = np.zeros((self.max_size, 1))  # Buffer per l'azione del freno
        self.throttle = np.zeros((self.max_size, 1))  # Buffer per l'azione dell'accelerazione
        self.next_state = np.array(self.state)
        self.reward = np.zeros((self.max_size, 1))
        self.done = np.zeros((self.max_size, 1))

    def add(self, state, steer, brake, throttle, next_state, reward, done):
        # Aggiunge un'esperienza al buffer
        self.state[self.ptr] = state
        self.steer[self.ptr] = steer
        self.brake[self.ptr] = brake
        self.throttle[self.ptr] = throttle
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done

        # Aggiorna il puntatore e la dimensione corrente del buffer
        self.ptr = (self.ptr + 1) % self.max_size
        self.crt_size = min(self.crt_size + 1, self.max_size)

    def sample(self):
        # Campiona un batch casuale dal buffer
        ind = np.random.randint(0, self.crt_size, size=self.batch_size)  # Indici casuali
        return (
            torch.FloatTensor(self.state[ind]).unsqueeze(1).to(self.device),
            torch.FloatTensor(self.steer[ind]).to(self.device),
            torch.FloatTensor(self.brake[ind]).to(self.device),
            torch.FloatTensor(self.throttle[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).unsqueeze(1).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.done[ind]).to(self.device)
        )
