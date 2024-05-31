import glob
import os
import sys
import time
import random
import pandas as pd

from config import action_map_steer, action_map_brake, action_map_throttle

# Aggiungi il percorso per importare i moduli di Carla
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

from synch_mode import CarlaSyncMode
from utils import *

random.seed(78)


def is_curve(waypoints, threshold=10.0):
    """
    Determine if a sequence of waypoints represents a curve based on the angle change.
    :param waypoints: A list of consecutive waypoints.
    :param threshold: The minimum change in angle to consider as a curve.
    :return: True if the waypoints represent a curve, False otherwise.
    """
    for i in range(len(waypoints) - 1):
        current_yaw = waypoints[i].transform.rotation.yaw
        next_yaw = waypoints[i + 1].transform.rotation.yaw
        angle_diff = abs(next_yaw - current_yaw)
        if angle_diff > threshold:
            return True
    return False


def find_curve_spawn_points(world, distance_before_curve=10.0, threshold=10.0):
    """
    Find spawn points a certain distance before curves.
    :param world: The Carla world object.
    :param distance_before_curve: Distance in meters before the curve to place the spawn point.
    :param threshold: The minimum change in angle to consider as a curve.
    :return: List of spawn points before curves.
    """
    map = world.get_map()
    all_spawn_points = map.get_spawn_points()
    curve_spawn_points = []

    waypoints = map.generate_waypoints(2.0)  # Generate waypoints with a distance of 2 meters between them

    for i in range(len(waypoints) - 5):  # Check groups of 5 waypoints to determine curves
        if is_curve(waypoints[i:i + 5], threshold):
            curve_point = waypoints[i]
            # Find the nearest spawn point before the curve
            for spawn_point in all_spawn_points:
                if spawn_point.location.distance(curve_point.transform.location) < distance_before_curve:
                    curve_spawn_points.append(spawn_point)
                    break

    return curve_spawn_points


class SimEnv(object):
    def __init__(self,
                 visuals=True,
                 target_speed=10,
                 max_iter=4000,
                 start_buffer=10,
                 train_freq=1,
                 save_freq=50,
                 start_ep=0,
                 max_dist_from_waypoint=20
                 ) -> None:
        # Inizializzazione degli attributi
        self.camera_rgb_left = None
        self.camera_rgb_right = None

        self.segmentation_sensor = None
        self.lidar_sensor = None
        self.camera_depth = None
        self.lane_invasion_sensor = None
        self.speed_controller = None
        self.camera_rgb_vis = None
        self.vehicle = None
        self.actor_list = None
        self.camera_rgb = None
        self.collision_sensor = None
        self.visuals = visuals
        if self.visuals:
            self._initiate_visuals()

        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)

        # Percorso al file .xodr
        xodr_path = "C:/Users/aless/Desktop/WindowsNoEditor/Import/Example_2.xodr"

        # Lettura del file .xodr
        with open(xodr_path, 'r') as f:
            xodr_data = f.read()

        # Generazione del mondo dalla mappa OpenDRIVE
        self.world = self.client.generate_opendrive_world(xodr_data)

        #self.world = self.client.load_world('Town02_Opt')
        self.world.unload_map_layer(carla.MapLayer.Decals)
        self.world.unload_map_layer(carla.MapLayer.Foliage)
        self.world.unload_map_layer(carla.MapLayer.ParkedVehicles)
        self.world.unload_map_layer(carla.MapLayer.Particles)
        self.world.unload_map_layer(carla.MapLayer.Props)
        self.world.unload_map_layer(carla.MapLayer.StreetLights)

        self.spawn_points = self.world.get_map().get_spawn_points()

        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicle_blueprint = self.blueprint_library.find('vehicle.tesla.model3')

        # input these later on as arguments
        self.global_t = 0  # global timestep
        self.target_speed = target_speed  # km/h
        self.max_iter = max_iter
        self.start_buffer = start_buffer
        self.train_freq = max(1, train_freq // 2)  # Dimezza la frequenza di addestramento
        self.save_freq = save_freq
        self.start_ep = start_ep

        self.max_dist_from_waypoint = max_dist_from_waypoint
        self.start_train = self.start_ep + self.start_buffer

        self.total_rewards = 0
        self.average_rewards_list = []

    # Inizializzazione delle finestre pygame per la visualizzazione
    def _initiate_visuals(self):
        pygame.init()

        self.display = pygame.display.set_mode(
            (800, 600),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.font = get_font()
        self.clock = pygame.time.Clock()

    # Creazione degli attori nell'ambiente
    def create_actors(self):

        self.actor_list = []
        # spawn vehicle at random location
        self.vehicle = self.world.spawn_actor(self.vehicle_blueprint, random.choice(self.spawn_points))
        self.actor_list.append(self.vehicle)

        # # Trova i punti di spawn vicini alle curve
        # curve_spawn_points = find_curve_spawn_points(self.world, threshold=10.0)
        # if not curve_spawn_points:
        #     raise ValueError("No curve spawn points found. Adjust the threshold or ensure the map has curves.")
        #
        # # Spawn vehicle at a random curve location
        # self.vehicle = self.world.spawn_actor(self.vehicle_blueprint, random.choice(curve_spawn_points))
        # self.actor_list.append(self.vehicle)

        # Sensori RGB
        self.camera_rgb = self.world.spawn_actor(
            self.blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-15)),
            attach_to=self.vehicle)
        self.actor_list.append(self.camera_rgb)

        self.camera_rgb_vis = self.world.spawn_actor(
            self.blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            attach_to=self.vehicle)
        self.actor_list.append(self.camera_rgb_vis)

        # Aggiungi telecamere laterali
        # self.camera_rgb_left = self.world.spawn_actor(
        #     self.blueprint_library.find('sensor.camera.rgb'),
        #     carla.Transform(carla.Location(x=1.5, y=-0.5, z=2.4), carla.Rotation(pitch=-15, yaw=-30)),
        #     attach_to=self.vehicle)
        # self.actor_list.append(self.camera_rgb_left)
        #
        # self.camera_rgb_right = self.world.spawn_actor(
        #     self.blueprint_library.find('sensor.camera.rgb'),
        #     carla.Transform(carla.Location(x=1.5, y=0.5, z=2.4), carla.Rotation(pitch=-15, yaw=30)),
        #     attach_to=self.vehicle)
        # self.actor_list.append(self.camera_rgb_right)

        # Sensore di profondità
        # self.camera_depth = self.world.spawn_actor(
        #     self.blueprint_library.find('sensor.camera.depth'),
        #     carla.Transform(carla.Location(x=0.5, z=2.4), carla.Rotation(pitch=-15)),
        #     attach_to=self.vehicle)
        # self.actor_list.append(self.camera_depth)

        # # Sensore Lidar per rilevare gli ostacoli
        # self.lidar_sensor = self.world.spawn_actor(
        #     self.blueprint_library.find('sensor.lidar.ray_cast'),
        #     carla.Transform(carla.Location(z=2.4)),  # Posizione in cui viene montato il Lidar
        #     attach_to=self.vehicle)
        # self.actor_list.append(self.lidar_sensor)

        # Sensori per l'invasione di corsia e la collisione
        self.lane_invasion_sensor = self.world.spawn_actor(
            self.blueprint_library.find('sensor.other.lane_invasion'),
            carla.Transform(),
            attach_to=self.vehicle)
        self.actor_list.append(self.lane_invasion_sensor)

        # Sensori per la segmentazione
        # self.segmentation_sensor = self.world.spawn_actor(
        #     self.blueprint_library.find('sensor.camera.semantic_segmentation'),
        #     carla.Transform(),
        #     attach_to=self.vehicle)
        # self.actor_list.append(self.segmentation_sensor)

        self.collision_sensor = self.world.spawn_actor(
            self.blueprint_library.find('sensor.other.collision'),
            carla.Transform(),
            attach_to=self.vehicle)
        self.actor_list.append(self.collision_sensor)

    # Reimposta lo stato dell'ambiente
    def reset(self):
        for actor in self.actor_list:
            if actor.is_alive:
                actor.destroy()

    def calculate_metrics(self, episode, total_reward, vehicle_location, waypoint, duration, timesteps,
                          collisions, lane_invasions, avg_speed, waypoint_distance, total_distance, max_speed,
                          hard_brakes):
        metrics_data = {
            'episode': [episode],
            'reward': [total_reward],
            'vehicle_location': [str(vehicle_location)],  # Convert to string to save in CSV
            'waypoint': [str(waypoint.transform.location)],  # Convert to string to save in CSV
            'duration': [duration],
            'timesteps': [timesteps],
            'collisions': [collisions],
            'lane_invasions': [lane_invasions],
            'avg_speed': [avg_speed],
            'waypoint_similarity': waypoint_distance,
            'total_distance': [total_distance],
            'max_speed': [max_speed],
            'hard_brakes': [hard_brakes]
        }

        metrics_df = pd.DataFrame(metrics_data)
        metrics_file = 'episode_metrics.csv'

        if not os.path.isfile(metrics_file):
            metrics_df.to_csv(metrics_file, index=False)
        else:
            metrics_df.to_csv(metrics_file, mode='a', header=False, index=False)

    def generate_episode(self, model, replay_buffer, ep, evaluation=True):
        start_time = time.time()
        with CarlaSyncMode(self.world, self.camera_rgb, self.camera_rgb_vis, self.lane_invasion_sensor, self.collision_sensor,
                           fps=30) as sync_mode:
            counter = 0
            total_collisions = 0
            total_lane_invasions = 0
            speed_sum = 0
            total_distance = 0
            max_speed = 0
            hard_brakes = 0
            previous_speed = 0
            no_collision_timesteps = 0  # Timestep senza collisioni

            waypoints = [self.world.get_map().get_waypoint(self.vehicle.get_location(), project_to_road=True,
                                                           lane_type=carla.LaneType.Driving)]

            previous_location = self.vehicle.get_location()  # Inizializza la posizione precedente

            snapshot, image_rgb, image_rgb_vis,  lane_invasion, collision = sync_mode.tick(
                timeout=1.0)

            if snapshot is None or image_rgb is None:
                print("No data, skipping episode")
                self.reset()
                return None

            image_rgb = process_img(image_rgb)
            image_rgb_vis = process_img(image_rgb_vis)
            # image_rgb_right = process_img(image_rgb_right)
            # image_rgb_left = process_img(image_rgb_left)
            # image_depth = process_img(camera_depth)
            # image_segmentation = process_img(segmentation_sensor)

            next_state = [image_rgb]
            while True:
                if self.visuals:
                    if should_quit():
                        return
                    self.clock.tick_busy_loop(30)

                vehicle_location = self.vehicle.get_location()

                # Calcola la distanza percorsa
                distance = vehicle_location.distance(previous_location)
                total_distance += distance
                previous_location = vehicle_location

                waypoint = self.world.get_map().get_waypoint(vehicle_location, project_to_road=True,
                                                             lane_type=carla.LaneType.Driving)

                speed = get_speed(self.vehicle)
                speed_sum += speed

                # Aggiorna la velocità massima raggiunta
                if speed > max_speed:
                    max_speed = speed

                # Conta le frenate brusche
                if previous_speed - speed > 5:  # Assumiamo che una frenata brusca sia una decelerazione > 5 m/s²
                    hard_brakes += 1

                previous_speed = speed

                state = next_state

                counter += 1
                self.global_t += 1

                action = model.select_action(state, eval=evaluation)
                steer, brake, throttle = action
                if action is not None:
                    steer = action_map_steer[steer]
                    brake = action_map_brake[brake]
                    throttle = action_map_throttle[throttle]

                throttle, brake = balance_throttle_brake(throttle, brake)
                control = self.vehicle.get_control()
                control.throttle = throttle
                control.brake = brake
                control.steer = steer
                self.vehicle.apply_control(control)

                fps = round(1.0 / snapshot.timestamp.delta_seconds)

                snapshot, image_rgb, image_rgb_vis, lane_invasion, collision = sync_mode.tick(
                    timeout=1.0)

                avg_speed = speed_sum / counter if counter > 0 else 0

                cos_yaw_diff, dist, collision, waypoint_similarity = get_reward_comp(self.vehicle, waypoint, collision,
                                                                                     waypoints)
                reward = reward_value(cos_yaw_diff, dist, collision, total_lane_invasions, avg_speed,
                                      waypoint_similarity, no_collision_timesteps)

                if collision:
                    total_collisions += 1
                    no_collision_timesteps = 0  # Resetta il contatore se c'è stata una collisione
                else:
                    no_collision_timesteps += 1  # Incrementa il contatore se non c'è stata collisione

                if lane_invasion:
                    total_lane_invasions += 1

                if snapshot is None or image_rgb is None or image_rgb_vis is None or collision is None is None:
                    print("Process ended here")
                    break

                done = 1 if collision else 0

                self.total_rewards += reward

                image_rgb = process_img(image_rgb)

                # image_rgb_right = process_img(image_rgb_right)
                # image_rgb_left = process_img(image_rgb_left)
                # image_depth = process_img(camera_depth)
                # image_segmentation = process_img(segmentation_sensor)

                next_state = [image_rgb]

                replay_buffer.add(state, steer, brake, throttle, next_state, reward, done)

                if not evaluation:
                    if ep > self.start_train and (self.global_t % self.train_freq) == 0:
                        model.train(replay_buffer)

                if self.visuals:
                    draw_image(self.display, image_rgb_vis)
                    self.display.blit(
                        self.font.render('% 5d FPS (real)' % self.clock.get_fps(), True, (255, 255, 255)),
                        (8, 10))
                    self.display.blit(
                        self.font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)),
                        (8, 28))
                    self.display.blit(
                        self.font.render('Speed: %.2f m/s' % speed, True, (255, 255, 255)),
                        (8, 46))
                    self.display.blit(
                        self.font.render('Steer: %.2f' % steer, True, (255, 255, 255)),
                        (8, 64))
                    self.display.blit(
                        self.font.render('Brake: %.2f' % brake, True, (255, 255, 255)),
                        (8, 82))
                    self.display.blit(
                        self.font.render('Throttle: %.2f' % throttle, True, (255, 255, 255)),
                        (8, 100))

                    pygame.display.flip()

                # Evidenzia i waypoint
                for wp in waypoints:
                    self.world.debug.draw_point(wp.transform.location, size=0.1, color=carla.Color(0, 255, 0),
                                                life_time=0.1)

                if collision == 1 or counter >= self.max_iter:
                    duration = time.time() - start_time
                    print("Episode {} processed".format(ep), counter, "total reward: ", reward, "duration:", duration)
                    self.calculate_metrics(ep, reward, vehicle_location, waypoint, duration, counter,
                                           total_collisions, total_lane_invasions, avg_speed, waypoint_similarity,
                                           total_distance, max_speed, hard_brakes)
                    break

            if ep % self.save_freq == 0 and ep > 0:
                self.save(model, ep)

    # Salva i pesi del modello
    def save(self, model, ep):
        if ep % self.save_freq == 0 and ep > self.start_ep:
            avg_reward = self.total_rewards / self.save_freq
            self.average_rewards_list.append(avg_reward)
            self.total_rewards = 0

            model.save('weights/model_ep_{}'.format(ep))

            print("Saved model with average reward =", avg_reward)

    # Termina la visualizzazione Pygame
    def quit(self):
        pygame.quit()


def calculate_waypoint_similarity(vehicle, waypoints):
    """
    Calcola la similarità con i waypoint misurando la distanza euclidea tra
    la posizione attuale del veicolo e i waypoint successivi.
    :param vehicle: Il veicolo attuale.
    :param waypoints: Lista di waypoint da seguire.
    :return: La somma delle distanze ai waypoint.
    """
    vehicle_location = vehicle.get_location()
    distances = []

    for waypoint in waypoints:
        wp_location = waypoint.transform.location
        distance = vehicle_location.distance(wp_location)
        distances.append(distance)

    # La similarità può essere definita come l'inverso della somma delle distanze
    similarity = 1.0 / (sum(distances) + 1e-6)  # Aggiungi un piccolo valore per evitare divisioni per zero
    return similarity


# Calcola i componenti della ricompensa
def get_reward_comp(vehicle, waypoint, collision, waypoints):
    vehicle_location = vehicle.get_location()
    x_wp = waypoint.transform.location.x
    y_wp = waypoint.transform.location.y

    x_vh = vehicle_location.x
    y_vh = vehicle_location.y

    wp_array = np.array([x_wp, y_wp])
    vh_array = np.array([x_vh, y_vh])

    dist = np.linalg.norm(wp_array - vh_array)

    vh_yaw = correct_yaw(vehicle.get_transform().rotation.yaw)
    wp_yaw = correct_yaw(waypoint.transform.rotation.yaw)
    cos_yaw_diff = np.cos((vh_yaw - wp_yaw) * np.pi / 180.)

    collision = 0 if collision is None else 1

    # Calcola la similarità con i waypoint
    waypoint_similarity = calculate_waypoint_similarity(vehicle, waypoints)

    return cos_yaw_diff, dist, collision, waypoint_similarity


def reward_value(cos_yaw_diff, dist, collision, lane_invasion, speed, waypoint_similarity, no_collision_timesteps,
                 target_speed=15, lambda_1=1, lambda_2=1, lambda_3=20, lambda_4=2, lambda_5=1, lambda_6=1,
                 yaw_penalty=2, lane_penalty=2, timestep_reward=0.1):
    """
    Calcola il valore della ricompensa.
    :param cos_yaw_diff: Differenza di orientamento tra il veicolo e il waypoint.
    :param dist: Distanza tra il veicolo e il waypoint.
    :param collision: Indicatore di collisione (1 se c'è stata una collisione, 0 altrimenti).
    :param lane_invasion: Indicatore di invasione di corsia (1 se c'è stata un'invasione di corsia, 0 altrimenti).
    :param speed: Velocità attuale del veicolo.
    :param waypoint_similarity: Similarità con i waypoint.
    :param target_speed: Velocità target che il veicolo dovrebbe mantenere.
    :param lambda_1: Peso del termine cos_yaw_diff.
    :param lambda_2: Peso del termine dist.
    :param lambda_3: Peso del termine collision.
    :param lambda_4: Peso del termine per la velocità.
    :param lambda_5: Peso del termine per l'invasione di corsia.
    :param lambda_6: Peso del termine per la similarità con i waypoint.
    :param yaw_penalty: Penalità per la differenza di orientamento.
    :param lane_penalty: Penalità per la distanza dalla corsia.
    :param timestep_reward: Ricompensa per ogni timestep senza collisioni.
    :param no_collision_timesteps: Numero di timestep consecutivi senza collisioni.
    :return: Ricompensa calcolata.
    """
    # Differenza di velocità dalla velocità target
    speed_diff = abs(speed - target_speed)

    reward = (
            (lambda_1 * cos_yaw_diff) -
            (lambda_2 * dist) -
            (lambda_3 * collision) -
            (lambda_4 * speed_diff) -
            (yaw_penalty * (1 - cos_yaw_diff)) -
            (lane_penalty * dist * 2) +
            (lambda_6 * waypoint_similarity)
    )

    # Aggiungi una ricompensa per mantenere la velocità target
    if target_speed + 2 >= speed >= target_speed - 2:
        reward += 2

    # Aggiungi ricompensa per timestep senza collisioni
    reward += timestep_reward * no_collision_timesteps

    return reward
