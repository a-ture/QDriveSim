import numpy as np

action_values = [-0.75, -0.5, -0.25, -0.15, -0.1, -0.05, 0,
                 0.05, 0.1, 0.15, 0.25, 0.5, 0.75]
action_map_steer = {i: x for i, x in enumerate(action_values)}

# Definizione dei valori di azione del freno
num_brake_levels = 15
max_brake = 1
action_values_brake = np.linspace(0, max_brake, num_brake_levels + 1)  # Crea intervalli regolari escludendo lo 0
action_map_brake = {i: x for i, x in enumerate(action_values_brake)}

# Definizione dei valori di azione dell'acceleratore
num_throttle_levels = 15
max_throttle = 1
action_values_throttle = np.linspace(0, max_throttle, num_throttle_levels + 1)  # Crea intervalli regolari
# escludendo lo 0
action_map_throttle = {i: x for i, x in enumerate(action_values_throttle)}

# Definizione dei parametri dell'ambiente
env_params = {
    'target_speed': 30,  # Velocità target dell'auto
    'max_iter': 4000,  # Numero massimo di iterazioni per episodio
    'start_buffer': 10,  # Numero di iterazioni iniziali senza addestramento
    'train_freq': 1,  # Frequenza di aggiornamento della rete neurale (addestramento)
    'save_freq': 50,  # Frequenza di salvataggio dei pesi del modello
    'start_ep': 401,  # Numero di episodi iniziali già eseguiti
    'max_dist_from_waypoint': 20  # Distanza massima dal percorso consentita
}
