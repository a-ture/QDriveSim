# Definizione dei valori di azione per l'algoritmo DQN
action_values = [-0.75, -0.5, -0.25, -0.15, -0.1, -0.05, 0,
                 0.05, 0.1, 0.15, 0.25, 0.5, 0.75]

# Creazione di un dizionario che mappa gli indici ai valori di azione
# Questo è utile per convertire gli indici in valori di azione facilmente
action_map = {i: x for i, x in enumerate(action_values)}

# Definizione dei parametri dell'ambiente
env_params = {
    'target_speed': 30,  # Velocità target dell'auto
    'max_iter': 4000,    # Numero massimo di iterazioni per episodio
    'start_buffer': 10,  # Numero di iterazioni iniziali senza addestramento
    'train_freq': 1,     # Frequenza di aggiornamento della rete neurale (addestramento)
    'save_freq': 50,    # Frequenza di salvataggio dei pesi del modello
    'start_ep': 401,       # Numero di episodi iniziali già eseguiti
    'max_dist_from_waypoint': 20  # Distanza massima dal percorso consentita
}
