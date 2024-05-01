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
        self.action = np.zeros((self.max_size, 1))
        self.next_state = np.array(self.state)
        self.reward = np.zeros((self.max_size, 1))
        self.done = np.zeros((self.max_size, 1))

    def add(self, state, action, next_state, reward, done):
        # Aggiunge un'esperienza al buffer
        self.state[self.ptr] = state
        self.action[self.ptr] = action
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
            # Converte i dati campionati in tensori PyTorch e li sposta sul dispositivo specificato
            torch.FloatTensor(self.state[ind]).unsqueeze(1).to(self.device),
            torch.LongTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).unsqueeze(1).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.done[ind]).to(self.device)
        )


def test_buffer():
    # Funzione di test per il ReplayBuffer
    img0 = np.zeros((5, 5))  # Creazione di un'immagine di zero
    img1 = img0 + 1  # Creazione di un'immagine di uno
    img2 = img0 + 2  # Creazione di un'immagine di due
    img3 = img0 + 3  # Creazione di un'immagine di tre

    action = 1  # Azione
    reward = 10  # Ricompensa
    done = 0  # Flag di terminazione

    device = "cpu"  # Dispositivo su cui lavorare (in questo caso, CPU)

    # Creazione del ReplayBuffer
    buffer = ReplayBuffer((5, 5), 2, 10, device)

    # Aggiunta di esperienze al buffer
    buffer.add(img0, action, img1, reward, done)
    buffer.add(img1, action, img2, reward, done)
    buffer.add(img2, action, img3, reward, done + 1)

    # Campionamento di un batch dal buffer e stampa della forma del batch
    sample = buffer.sample()[0]
    print(sample.shape)

    # Normalizzazione del batch e stampa della sua forma
    norm = transforms.Normalize((0.5, 0.5), (0.5, 0.5))
    print(norm(sample).shape)

# Test della funzione test_buffer
# test_buffer()

# Commentata la chiamata a test_buffer per evitare l'esecuzione durante l'importazione o l'esecuzione del codice
