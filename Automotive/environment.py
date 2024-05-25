import glob
import os
import sys
import time

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
import random

from synch_mode import CarlaSyncMode
from controllers import PIDLongitudinalController
from utils import *

random.seed(78)


# Definizione della classe SimEnv
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

        self.world = self.client.load_world('Town02_Opt')
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
        self.train_freq = train_freq
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

        # Sensore di profondità
        self.camera_depth = self.world.spawn_actor(
            self.blueprint_library.find('sensor.camera.depth'),
            carla.Transform(carla.Location(x=0.5, z=2.4), carla.Rotation(pitch=-15)),
            attach_to=self.vehicle)
        self.actor_list.append(self.camera_depth)

        # Sensore Lidar per rilevare gli ostacoli
        self.lidar_sensor = self.world.spawn_actor(
            self.blueprint_library.find('sensor.lidar.ray_cast'),
            carla.Transform(carla.Location(z=2.4)),  # Posizione in cui viene montato il Lidar
            attach_to=self.vehicle)
        self.actor_list.append(self.lidar_sensor)

        # Sensori per l'invasione di corsia e la collisione
        self.lane_invasion_sensor = self.world.spawn_actor(
            self.blueprint_library.find('sensor.other.lane_invasion'),
            carla.Transform(),
            attach_to=self.vehicle)
        self.actor_list.append(self.lane_invasion_sensor)

        # Sensori per la segmentazione
        self.segmentation_sensor = self.world.spawn_actor(
            self.blueprint_library.find('sensor.camera.semantic_segmentation'),
            carla.Transform(),
            attach_to=self.vehicle)
        self.actor_list.append(self.segmentation_sensor)

        self.collision_sensor = self.world.spawn_actor(
            self.blueprint_library.find('sensor.other.collision'),
            carla.Transform(),
            attach_to=self.vehicle)
        self.actor_list.append(self.collision_sensor)

        # Controllore PID
        self.speed_controller = PIDLongitudinalController(self.vehicle)

    # Reimposta lo stato dell'ambiente
    def reset(self):
        for actor in self.actor_list:
            if actor.is_alive:
                actor.destroy()

    def calculate_metrics(self, episode, speed, total_reward, vehicle_location, waypoint, duration, timesteps,
                          collisions, lane_invasions, avg_speed):
        metrics_data = {
            'episode': [episode],
            'speed': [speed],
            'total_reward': [total_reward],
            'vehicle_location': [str(vehicle_location)],  # Convert to string to save in CSV
            'waypoint': [str(waypoint.transform.location)],  # Convert to string to save in CSV
            'duration': [duration],
            'timesteps': [timesteps],
            'collisions': [collisions],
            'lane_invasions': [lane_invasions],
            'avg_speed': [avg_speed],
        }

        metrics_df = pd.DataFrame(metrics_data)
        metrics_file = 'episode_metrics.csv'

        if not os.path.isfile(metrics_file):
            metrics_df.to_csv(metrics_file, index=False)
        else:
            metrics_df.to_csv(metrics_file, mode='a', header=False, index=False)

    def generate_episode(self, model, replay_buffer, ep, eval=True):
        start_time = time.time()
        with CarlaSyncMode(self.world, self.camera_rgb, self.camera_rgb_vis, self.camera_depth, self.lidar_sensor,
                           self.segmentation_sensor, self.lane_invasion_sensor, self.collision_sensor,
                           fps=30) as sync_mode:
            counter = 0
            total_collisions = 0
            total_lane_invasions = 0
            speed_sum = 0

            snapshot, image_rgb, image_rgb_vis, camera_depth, lidar_data, segmentation_sensor, lane_invasion, collision = sync_mode.tick(
                timeout=1.0)

            # destroy if there is no data
            if snapshot is None or image_rgb is None:
                print("No data, skipping episode")
                self.reset()
                return None

            image_rgb = process_img(image_rgb)
            image_rgb_vis = process_img(image_rgb_vis)
            image_depth = process_img(camera_depth)
            image_segmentation = process_img(segmentation_sensor)
            lidar_points = process_lidar(lidar_data)

            next_state = [image_rgb, image_depth, image_segmentation, lidar_points]

            while True:
                if self.visuals:
                    if should_quit():
                        return
                    self.clock.tick_busy_loop(30)

                vehicle_location = self.vehicle.get_location()

                waypoint = self.world.get_map().get_waypoint(vehicle_location, project_to_road=True,
                                                             lane_type=carla.LaneType.Driving)

                speed = get_speed(self.vehicle)
                speed_sum += speed

                # Advance the simulation and wait for the data.
                state = next_state

                counter += 1
                self.global_t += 1

                action = model.select_action(state, eval=eval)
                steer, brake, throttle = action  # Ottieni le azioni per sterzata, frenata e accelerazione
                # Applica il mapping agli indici delle azioni se action_map non è None
                if action is not None:
                    steer = action_map_steer[steer]
                    brake = action_map_brake[brake]
                    throttle = action_map_throttle[throttle]

                # Controllo della velocità della macchina
                # Bilancia throttle e brake
                throttle, brake = balance_throttle_brake(throttle, brake)
                # Applica le azioni alla macchina
                control = self.vehicle.get_control()
                control.throttle = throttle
                control.brake = brake
                control.steer = steer
                self.vehicle.apply_control(control)

                fps = round(1.0 / snapshot.timestamp.delta_seconds)

                snapshot, image_rgb, image_rgb_vis, camera_depth, lidar_data, segmentation_sensor, lane_invasion, collision = sync_mode.tick(
                    timeout=1.0)

                avg_speed = speed_sum / counter if counter > 0 else 0

                cos_yaw_diff, dist, collision = get_reward_comp(self.vehicle, waypoint, collision)
                reward = reward_value(cos_yaw_diff, dist, collision, total_lane_invasions,avg_speed)

                if collision:
                    total_collisions += 1

                if lane_invasion:
                    total_lane_invasions += 1

                if snapshot is None or image_rgb is None or image_rgb_vis is None or collision is None or lidar_data is None or segmentation_sensor is None or camera_depth is None:
                    print("Process ended here")
                    break

                image = process_img(image_rgb)
                image_depth = process_img(camera_depth)
                image_segmentation = process_img(segmentation_sensor)
                lidar_points = process_lidar(lidar_data)

                done = 1 if collision else 0

                self.total_rewards += reward

                next_state = [image, image_depth, image_segmentation, lidar_points]

                # Aggiungi le azioni al replay buffer
                replay_buffer.add(state, steer, brake, throttle, next_state, reward, done)

                if not eval:
                    if ep > self.start_train and (self.global_t % self.train_freq) == 0:
                        model.train(replay_buffer)

                # Draw the display.
                if self.visuals:
                    draw_image(self.display, image_rgb_vis)
                    self.display.blit(
                        self.font.render('% 5d FPS (real)' % self.clock.get_fps(), True, (255, 255, 255)),
                        (8, 10))
                    self.display.blit(
                        self.font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)),
                        (8, 28))
                    # Aggiungi la velocità attuale della macchina
                    self.display.blit(
                        self.font.render('Speed: %.2f m/s' % speed, True, (255, 255, 255)),
                        (8, 46))

                    # Draw sensor data
                    draw_image(self.display, image_rgb)
                    draw_depth_image(self.display, image_depth)
                    draw_segmentation_image(self.display, image_segmentation)
                    draw_lidar_image(self.display, lidar_points)

                    pygame.display.flip()

                if collision == 1 or counter >= self.max_iter or dist > self.max_dist_from_waypoint:
                    print("Episode {} processed".format(ep), counter, "total reward: ", reward)
                    duration = time.time() - start_time

                    self.calculate_metrics(ep, speed, self.total_rewards, vehicle_location, waypoint, duration, counter,
                                           total_collisions, total_lane_invasions, avg_speed)
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


# Calcola i componenti della ricompensa
def get_reward_comp(vehicle, waypoint, collision):
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

    return cos_yaw_diff, dist, collision


# Calcola il valore della ricompensa
def reward_value(cos_yaw_diff, dist, collision, lane_invasion, speed, target_speed=15, lambda_1=1, lambda_2=1,
                 lambda_3=5, lambda_4=2, lambda_5=0.5):
    """
    Calcola il valore della ricompensa.
    :param cos_yaw_diff: Differenza di orientamento tra il veicolo e il waypoint.
    :param dist: Distanza tra il veicolo e il waypoint.
    :param collision: Indicatore di collisione (1 se c'è stata una collisione, 0 altrimenti).
    :param lane_invasion: Indicatore di invasione di corsia (1 se c'è stata un'invasione di corsia, 0 altrimenti).
    :param speed: Velocità attuale del veicolo.
    :param target_speed: Velocità target che il veicolo dovrebbe mantenere.
    :param lambda_1: Peso del termine cos_yaw_diff.
    :param lambda_2: Peso del termine dist.
    :param lambda_3: Peso del termine collision.
    :param lambda_4: Peso del termine per la velocità.
    :param lambda_5: Peso del termine per l'invasione di corsia.
    :return: Ricompensa calcolata.
    """
    speed_diff = abs(speed - target_speed)
    reward = (lambda_1 * cos_yaw_diff) - (lambda_2 * dist) - (lambda_3 * collision) - (lambda_4 * speed_diff) - (
            lambda_5 * lane_invasion)
    return reward
