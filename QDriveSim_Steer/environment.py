import glob
import os
import sys
import time

import pandas as pd

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


def calculate_metrics(episode, total_reward, vehicle_location, waypoint, duration, timesteps,
                      collisions, lane_invasions, avg_speed, total_distance, max_speed):
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
        'total_distance': [total_distance],
        'max_speed': [max_speed],
    }

    metrics_df = pd.DataFrame(metrics_data)
    metrics_file = 'episode_metrics.csv'

    if not os.path.isfile(metrics_file):
        metrics_df.to_csv(metrics_file, index=False)
    else:
        metrics_df.to_csv(metrics_file, mode='a', header=False, index=False)


class SimEnv(object):
    def __init__(self,
                 visuals=True,
                 target_speed=30,
                 max_iter=4000,
                 start_buffer=10,
                 train_freq=1,
                 save_freq=200,
                 start_ep=0,
                 max_dist_from_waypoint=20
                 ) -> None:
        self.camera_rgb_left = None
        self.segmentation_sensor = None
        self.camera_rgb_right = None
        self.camera_depth = None
        self.camera_rgb_vis = None
        self.speed_controller = None
        self.collision_sensor = None
        self.actor_list = None
        self.vehicle = None
        self.camera_rgb = None
        self.lane_invasion_sensor = None
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
        self.world = self.client.load_world('Town03')
        self.world.unload_map_layer(carla.MapLayer.Decals)
        self.world.unload_map_layer(carla.MapLayer.Foliage)
        self.world.unload_map_layer(carla.MapLayer.ParkedVehicles)
        self.world.unload_map_layer(carla.MapLayer.Particles)
        self.world.unload_map_layer(carla.MapLayer.Props)
        self.world.unload_map_layer(carla.MapLayer.StreetLights)

        self.spawn_points = self.world.get_map().get_spawn_points()

        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicle_blueprint = self.blueprint_library.find('vehicle.nissan.patrol')

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

    def _initiate_visuals(self):
        pygame.init()

        self.display = pygame.display.set_mode(
            (800, 600),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.font = get_font()
        self.clock = pygame.time.Clock()

    def create_actors(self):
        self.actor_list = []
        # spawn vehicle at random location
        self.vehicle = self.world.spawn_actor(self.vehicle_blueprint, random.choice(self.spawn_points))
        # vehicle.set_autopilot(True)
        self.actor_list.append(self.vehicle)

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

        self.lane_invasion_sensor = self.world.spawn_actor(
            self.blueprint_library.find('sensor.other.lane_invasion'),
            carla.Transform(),
            attach_to=self.vehicle)
        self.actor_list.append(self.lane_invasion_sensor)

        self.collision_sensor = self.world.spawn_actor(
            self.blueprint_library.find('sensor.other.collision'),
            carla.Transform(),
            attach_to=self.vehicle
        )

        # Sensori per la segmentazione
        self.segmentation_sensor = self.world.spawn_actor(
            self.blueprint_library.find('sensor.camera.semantic_segmentation'),
            carla.Transform(),
            attach_to=self.vehicle)
        self.actor_list.append(self.segmentation_sensor)

        # Aggiungi telecamere laterali
        self.camera_rgb_left = self.world.spawn_actor(
            self.blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x=1.5, y=-0.5, z=2.4), carla.Rotation(pitch=-15, yaw=-30)),
            attach_to=self.vehicle)
        self.actor_list.append(self.camera_rgb_left)

        self.camera_rgb_right = self.world.spawn_actor(
            self.blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x=1.5, y=0.5, z=2.4), carla.Rotation(pitch=-15, yaw=30)),
            attach_to=self.vehicle)
        self.actor_list.append(self.camera_rgb_right)

        # Sensore di profondità
        self.camera_depth = self.world.spawn_actor(
            self.blueprint_library.find('sensor.camera.depth'),
            carla.Transform(carla.Location(x=0.5, z=2.4), carla.Rotation(pitch=-15)),
            attach_to=self.vehicle)
        self.actor_list.append(self.camera_depth)

        self.actor_list.append(self.collision_sensor)

        self.speed_controller = PIDLongitudinalController(self.vehicle)

    def reset(self):
        for actor in self.actor_list:
            if actor.is_alive:
                actor.destroy()
        self.actor_list = []

    def generate_episode(self, model, replay_buffer, ep, action_map=None, eval=True):
        start_time = time.time()
        with CarlaSyncMode(self.world, self.camera_rgb, self.camera_rgb_vis, self.segmentation_sensor,
                           self.camera_rgb_right, self.camera_rgb_left, self.camera_depth, self.lane_invasion_sensor,
                           self.collision_sensor,
                           fps=30) as sync_mode:

            counter = 0
            total_lane_invasions = 0
            speed_sum = 0
            total_distance = 0
            max_speed = 0

            snapshot, image_rgb, image_rgb_vis, image_segmentation, image_right, image_left, image_depth, lane_invasion, collision = sync_mode.tick(
                timeout=2.0)
            previous_location = self.vehicle.get_location()
            # destroy if there is no data
            if snapshot is None or image_rgb is None:
                print("No data, skipping episode")
                self.reset()
                return None

            image = process_img(image_rgb)
            image_right = process_img(image_right)
            image_left = process_img(image_left)
            image_depth = process_img(image_depth)
            image_segmentation = process_img(image_segmentation)
            next_state = [image, image_right, image_left, image_depth, image_segmentation]

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
                steer = action
                if action_map is not None:
                    steer = action_map[action]

                control = self.speed_controller.run_step(self.target_speed)
                control.steer = steer
                self.vehicle.apply_control(control)

                fps = round(1.0 / snapshot.timestamp.delta_seconds)

                snapshot, image_rgb, image_rgb_vis, image_segmentation, image_right, image_left, image_depth, lane_invasion, collision = sync_mode.tick(
                    timeout=2.0)

                cos_yaw_diff, dist, collision = get_reward_comp(self.vehicle, waypoint, collision)
                reward = reward_value(cos_yaw_diff, dist, collision)

                if snapshot is None or image_rgb is None:
                    print("Process ended here")
                    break

                image = process_img(image_rgb)
                image_right = process_img(image_right)
                image_left = process_img(image_left)
                image_depth = process_img(image_depth)
                image_segmentation = process_img(image_segmentation)

                done = 1 if collision else 0

                self.total_rewards += reward

                next_state = [image, image_right, image_left, image_depth, image_segmentation]

                replay_buffer.add(state, action, next_state, reward, done)

                distance = vehicle_location.distance(previous_location)
                total_distance += distance
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
                    self.display.blit(
                        self.font.render('Steer: %.2f' % steer, True, (255, 255, 255)),
                        (8, 64))
                    pygame.display.flip()

                if lane_invasion:
                    total_lane_invasions += 1

                # Aggiorna la velocità massima raggiunta
                if speed > max_speed:
                    max_speed = speed

                avg_speed = speed_sum / counter if counter > 0 else 0

                if collision == 1 or counter >= self.max_iter or dist > self.max_dist_from_waypoint:
                    duration = time.time() - start_time
                    print("Episode {} processed".format(ep), counter, "reward", reward, "duration", duration)
                    calculate_metrics(ep, reward, vehicle_location, waypoint, duration, counter,
                                      collision, total_lane_invasions, avg_speed,
                                      total_distance, max_speed)
                    break

            if ep % self.save_freq == 0 and ep > 0:
                self.save(model, ep)

    def save(self, model, ep):
        if ep % self.save_freq == 0 and ep > self.start_ep:
            avg_reward = self.total_rewards / self.save_freq
            self.average_rewards_list.append(avg_reward)
            self.total_rewards = 0

            model.save('weights/model_ep_{}'.format(ep))

            print("Saved model with average reward =", avg_reward)

    def quit(self):
        pygame.quit()


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


def reward_value(cos_yaw_diff, dist, collision, lambda_1=1, lambda_2=1, lambda_3=5):
    reward = (lambda_1 * cos_yaw_diff) - (lambda_2 * dist) - (lambda_3 * collision)
    return reward
