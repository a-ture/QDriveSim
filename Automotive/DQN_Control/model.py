import copy  # Importa il modulo copy per creare copie profonde di oggetti
import numpy as np  # Importa NumPy, una libreria per il calcolo numerico
import torch  # Importa PyTorch, una libreria per il machine learning
import torch.nn as nn  # Importa il modulo nn di PyTorch per definire reti neurali
import torch.nn.functional as F  # Importa funzioni di attivazione e altre funzioni utili da nn.functional


# TODO: computer vision only non devono arrivare altri dati
# Definizione della rete neurale convoluzionale (CNN)
class ConvNet(nn.Module):
    def __init__(self, dim, in_channels, num_actions_steer, num_actions_brake, num_actions_throttle) -> None:
        super(ConvNet, self).__init__()
        # Definizione dei layer convoluzionali
        self.conv1 = nn.Conv2d(in_channels, 32, 8, 4)
        self.conv1_bn = nn.BatchNorm2d(32)  # Batch normalization per il primo layer convoluzionale
        self.conv2 = nn.Conv2d(32, 64, 4, 3)
        self.conv2_bn = nn.BatchNorm2d(64)  # Batch normalization per il secondo layer convoluzionale
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.conv3_bn = nn.BatchNorm2d(64)  # Batch normalization per il terzo layer convoluzionale

        # Definizione dei layer fully connected (FC)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc1_bn = nn.BatchNorm1d(256)  # Batch normalization per il primo layer fully connected
        self.fc2 = nn.Linear(256, 32)
        self.fc2_bn = nn.BatchNorm1d(32)  # Batch normalization per il secondo layer fully connected

        # Output layer per i valori di azione
        self.fc3_steer = nn.Linear(32, num_actions_steer)  # Output layer per gli angoli di sterzata
        self.fc3_brake = nn.Linear(32, num_actions_brake)  # Output layer per i valori di frenata
        self.fc3_throttle = nn.Linear(32, num_actions_throttle)  # Output layer per i valori di accelerazione

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = x.view(-1, 64 * 8 * 8)  # flatten
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.relu(self.fc2_bn(self.fc2(x)))

        # Output per tutte e tre le azioni
        steer_output = self.fc3_steer(x)
        brake_output = self.fc3_brake(x)
        throttle_output = self.fc3_throttle(x)

        return steer_output, brake_output, throttle_output


# Definizione dell'algoritmo Deep Q-Network (DQN)
class DQN(object):
    def __init__(
            self,
            num_actions_steer,
            num_action_brake,
            num_action_throttle,
            state_dim,
            in_channels,
            device,
            discount=0.9,
            optimizer="Adam",
            optimizer_parameters={'lr': 0.01},
            target_update_frequency=1e4,
            initial_eps=1,
            end_eps=0.05,
            eps_decay_period=25e4,
            eval_eps=0.001
    ) -> None:
        self.current_eps = None
        self.device = device  # Dispositivo su cui lavorare (CPU o GPU)

        # Rete neurale Q per approssimare la funzione Q
        self.Q = ConvNet(state_dim, in_channels, num_actions_steer, num_action_brake, num_action_throttle).to(
            self.device)
        # Rete neurale Q target (inizialmente uguale a Q)
        self.Q_target = copy.deepcopy(self.Q)  # Copia profonda della rete Q per inizializzare la rete target
        # Ottimizzatore per la rete Q
        self.Q_optimizer = getattr(torch.optim, optimizer)(self.Q.parameters(), **optimizer_parameters)

        self.discount = discount  # Fattore di sconto per le future ricompense

        self.target_update_frequency = target_update_frequency  # Frequenza di aggiornamento della rete target

        # Parametri per la decrescita dell'epsilon-greedy
        self.initial_eps = initial_eps  # Epsilon iniziale
        self.end_eps = end_eps  # Epsilon finale
        self.slope = (self.end_eps - self.initial_eps) / eps_decay_period  # Pendenza della decrescita

        self.state_shape = (-1,) + state_dim  # Forma dello stato
        self.eval_eps = eval_eps  # Epsilon per valutazioni
        self.num_actions_steer = num_actions_steer  # Numero di azioni disponibili
        self.num_actions_brake = num_action_brake  # Numero di azioni disponibili
        self.num_actions_throttle= num_action_throttle  # Numero di azioni disponibili

        self.iterations = 0  # Contatore delle iterazioni

    # Metodo per selezionare azioni secondo una politica epsilon-greedy
    def select_action(self, state, eval=False):
        eps = self.eval_eps if eval \
            else max(self.slope * self.iterations + self.initial_eps, self.end_eps)
        self.current_eps = eps

        # Selezione delle azioni per ciascuna delle azioni disponibili
        if np.random.uniform(0, 1) > eps:
            self.Q.eval()  # Imposta la rete Q in modalità di valutazione
            with torch.no_grad():
                # senza batch norm, rimuovere il unsqueeze
                state = torch.FloatTensor(state).reshape(self.state_shape).unsqueeze(0).to(self.device)
                steer, brake, throttle = self.Q(state)
                # Seleziona l'azione con il valore Q massimo per ciascuna azione
                steer_action = int(steer.argmax(1))
                brake_action = int(brake.argmax(1))
                throttle_action = int(throttle.argmax(1))
                return steer_action, brake_action, throttle_action
        else:
            # Seleziona azioni casuali per ciascuna delle azioni disponibili
            steer_action = np.random.randint(self.num_actions_steer)
            brake_action = np.random.randint(self.num_actions_brake)
            throttle_action = np.random.randint(self.num_actions_throttle)
            return steer_action, brake_action, throttle_action

    # Metodo per addestrare la rete Q utilizzando il replay buffer
    def train(self, replay_buffer):
        self.Q.train()  # Imposta la rete Q in modalità di addestramento
        # Campiona un minibatch dal replay buffer
        state, action, next_state, reward, done = replay_buffer.sample()

        # Calcola il target Q value
        with torch.no_grad():
            target_Q = reward + (1 - done) * self.discount * self.Q_target(next_state).max(1, keepdim=True)[0]

        # Calcola la stima Q attuale
        current_Q = self.Q(state).gather(1, action)

        # Calcola la perdita Q
        Q_loss = F.smooth_l1_loss(current_Q, target_Q)

        # Ottimizza la rete Q
        self.Q_optimizer.zero_grad()
        Q_loss.backward()
        self.Q_optimizer.step()

        # Aggiorna la rete target ogni tot iterazioni
        self.iterations += 1
        self.copy_target_update()

    # Metodo per aggiornare la rete target con i pesi della rete Q
    def copy_target_update(self):
        if self.iterations % self.target_update_frequency == 0:
            print('target network updated')
            print('current epsilon', self.current_eps)
            self.Q_target.load_state_dict(self.Q.state_dict())

    # Metodo per salvare i pesi della rete Q e dell'ottimizzatore
    def save(self, filename):
        torch.save(self.Q.state_dict(), filename + "_Q")
        torch.save(self.Q_optimizer.state_dict(), filename + "_optimizer")

    # Metodo per caricare i pesi della rete Q e dell'ottimizzatore
    def load(self, filename):
        self.Q.load_state_dict(torch.load(filename + "_Q"))
        self.Q_target = copy.deepcopy(self.Q)  # Aggiorna la rete target con i pesi della rete Q
        self.Q_optimizer.load_state_dict(torch.load(filename + "_optimizer"))  # Carica gli stati dell'ottimizzatore
