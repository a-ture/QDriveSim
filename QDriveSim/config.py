# steer action values
import numpy as np

steer_values = [-0.45, -0.25, -0.20, -0.15, -0.10, -0.05, 0, 0.45,
                0.25, 0.15, 0.05, 0.15, 0.20]
action_map_steer = {i: x for i, x in enumerate(steer_values)}

# Definizione dei valori di azione del freno
num_brake_levels = 5
max_brake = 1
action_values_brake = np.linspace(0, max_brake, num_brake_levels + 1)  # Crea intervalli regolari escludendo lo 0
action_map_brake = {i: x for i, x in enumerate(action_values_brake)}

# Definizione dei valori di azione dell'acceleratore
num_throttle_levels = 12
max_throttle = 1
action_values_throttle = np.linspace(0, max_throttle, num_throttle_levels + 1)  # Crea intervalli regolari
# escludendo lo 0
action_map_throttle = {i: x for i, x in enumerate(action_values_throttle)}

env_params = {
    'target_speed': 20,
    'max_iter': 4000,
    'start_buffer': 10,
    'train_freq': 1,
    'save_freq': 50,
    'start_ep': 0,
    'max_dist_from_waypoint': 20
}

