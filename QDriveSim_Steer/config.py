# dqn action values
action_values = [-0.45, -0.25, -0.20, -0.15, -0.10, -0.05, 0,
                 0.05, 0.10, 0.15, 0.20, 0.25, 0.45]
action_map = {i: x for i, x in enumerate(action_values)}

env_params = {
    'target_speed': 20,
    'max_iter': 6000,
    'start_buffer': 10,
    'train_freq': 1,
    'save_freq': 100,
    'start_ep': 0,
    'max_dist_from_waypoint': 20
}
