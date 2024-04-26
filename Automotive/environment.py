import glob
import os
import sys
import numpy as np

# Aggiunge al percorso di sistema il percorso del modulo di Carla
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# Importa il modulo carla
import carla
import random
import pickle

# Importa la classe CarlaSyncMode e il controller PIDLongitudinalController dalla cartella synch_mode e controllers rispettivamente
from synch_mode import CarlaSyncMode
from controllers import PIDLongitudinalController
# Importa funzioni di utilità dalla cartella utils
from utils import *

# Imposta il seme per la generazione casuale
random.seed(78)


# Classe per l'ambiente di simulazione
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
        # Impostazioni iniziali
        self.visuals = visuals  # Indica se abilitare la modalità visuale
        if self.visuals:
            self._initiate_visuals()  # Inizializza i componenti per la modalità visuale

        # Connessione al server di Carla
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)

        # Caricamento della mappa e rimozione di alcuni elementi della mappa per migliorare le prestazioni
        self.world = self.client.load_world('Town02_Opt')
        self._unload_map_elements()

        # Ottenimento dei punti di spawn dalla mappa
        self.spawn_points = self.world.get_map().get_spawn_points()

        # Ottenimento della libreria dei blueprint degli oggetti
        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicle_blueprint = self.blueprint_library.find('vehicle.nissan.patrol')

        # Impostazioni dell'episodio
        self.target_speed = target_speed  # Velocità obiettivo in km/h
        self.max_iter = max_iter  # Numero massimo di iterazioni per episodio
        self.start_buffer = start_buffer  # Numero di episodi iniziali da saltare prima di iniziare l'addestramento
        self.train_freq = train_freq  # Frequenza di addestramento del modello (numero di passi tra ogni addestramento)
        self.save_freq = save_freq  # Frequenza di salvataggio dei pesi del modello
        self.start_ep = start_ep  # Episodio iniziale
        self.max_dist_from_waypoint = max_dist_from_waypoint  # Distanza massima da un punto di waypoint prima di considerare l'episodio terminato
        self.start_train = self.start_ep + self.start_buffer  # Episodio iniziale per iniziare l'addestramento

        # Inizializzazione delle variabili per i punteggi e per il calcolo della media dei punteggi
        self.total_rewards = 0
        self.average_rewards_list = []

    # Metodo per inizializzare i componenti della modalità visuale
    def _initiate_visuals(self):
        pygame.init()
        self.display = pygame.display.set_mode((800, 600), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.font = get_font()
        self.clock = pygame.time.Clock()

    # Metodo per rimuovere elementi non necessari dalla mappa per migliorare le prestazioni
    def _unload_map_elements(self):
        self.world.unload_map_layer(carla.MapLayer.Decals)
        self.world.unload_map_layer(carla.MapLayer.Foliage)
        self.world.unload_map_layer(carla.MapLayer.ParkedVehicles)
        self.world.unload_map_layer(carla.MapLayer.Particles)
        self.world.unload_map_layer(carla.MapLayer.Props)
        self.world.unload_map_layer(carla.MapLayer.StreetLights)

    # Metodo per creare gli attori (veicolo, sensori) nell'ambiente
    def create_actors(self):
        self.actor_list = []

        # Spawn del veicolo in un punto casuale
        self.vehicle = self.world.spawn_actor(self.vehicle_blueprint, random.choice(self.spawn_points))
        self.actor_list.append(self.vehicle)

        # Aggiunta di sensori al veicolo (camera RGB, collision sensor)
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

        self.collision_sensor = self.world.spawn_actor(
            self.blueprint_library.find('sensor.other.collision'),
            carla.Transform(),
            attach_to=self.vehicle
        )
        self.actor_list.append(self.collision_sensor)

        self.speed_controller = PIDLongitudinalController(self.vehicle)

    # Metodo per reimpostare l'ambiente
    def reset(self):
        for actor in self.actor_list:
            actor.destroy()

    # Metodo per generare un episodio
    def generate_episode(self, model, replay_buffer, ep, action_map=None, eval=True):
        with CarlaSyncMode(self.world, self.camera_rgb, self.camera_rgb_vis, self.collision_sensor,
                           fps=30) as sync_mode:
            counter = 0

            snapshot, image_rgb, image_rgb_vis, collision = sync_mode.tick(timeout=2.0)

            # Interrompe se non ci sono dati
            if snapshot is None or image_rgb is None:
                print("No data, skipping episode")
                self.reset()
                return None

            image = process_img(image_rgb)
            next_state = image

            while True:
                if self.visuals:
                    if should_quit():
                        return
                    self.clock.tick_busy_loop(30)

                # Ottenimento della posizione del veicolo e del waypoint più vicino
                vehicle_location = self.vehicle.get_location()
                waypoint = self.world.get_map().get_waypoint(vehicle_location, project_to_road=True,
                                                             lane_type=carla.LaneType.Driving)

                # Calcolo della velocità del veicolo
                speed = get_speed(self.vehicle)

                # Avanzamento della simulazione e attesa dei dati
                state = next_state

                counter += 1
                self.global_t += 1

                # Selezione dell'azione
                action = model.select_action(state, eval=eval)
                steer = action
                if action_map is not None:
                    steer = action_map[action]

                # Applicazione del controllo (steering) al veicolo
                control = self.speed_controller.run_step(self.target_speed)
                control.steer = steer
                self.vehicle.apply_control(control)

                # Calcolo della ricompensa e verifiche di terminazione
                cos_yaw_diff, dist, collision = get_reward_comp(self.vehicle, waypoint, collision)
                reward = reward_value(cos_yaw_diff, dist, collision)

                # Controllo di fine episodio
                if snapshot is None or image_rgb is None:
                    print("Process ended here")
                    break

                image = process_img(image_rgb)
                done = 1 if collision else 0
                next_state = image

                # Aggiunta delle informazioni al replay buffer
                replay_buffer.add(state, action, next_state, reward, done)

                # Addestramento del modello se non in modalità di valutazione
                if not eval:
                    if ep > self.start_train and (self.global_t % self.train_freq) == 0:
                        model.train(replay_buffer)

                # Disegno del display
                if self.visuals:
                    draw_image(self.display, image_rgb_vis)
                    self.display.blit(self.font.render('% 5d FPS (real)' % self.clock.get_fps(), True, (255, 255, 255)),
                                      (8, 10))
                    self.display.blit(self.font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)), (8, 28))
                    pygame.display.flip()

                # Controllo di fine episodio
                if collision == 1 or counter >= self.max_iter or dist > self.max_dist_from_waypoint:
                    print("Episode {} processed".format(ep), counter)
                    break

            # Salvataggio dei pesi del modello e calcolo della media dei punteggi
            if ep % self.save_freq == 0 and ep > 0:
                self.save(model, ep)

    # Metodo per salvare i pesi del modello e calcolare la media dei punteggi
    def save(self, model, ep):
        if ep % self.save_freq == 0 and ep > self.start_ep:
            avg_reward = self.total_rewards / self.save_freq
            self.average_rewards_list.append(avg_reward)
            self.total_rewards = 0

            model.save('weights/model_ep_{}'.format(ep))

            print("Saved model with average reward =", avg_reward)

    # Metodo per chiudere la finestra visuale di Pygame
    def quit(self):
        pygame.quit()


# Metodo per calcolare la differenza di yaw corretta tra due angoli
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


# Metodo per calcolare la ricompensa
def reward_value(cos_yaw_diff, dist, collision, lambda_1=1, lambda_2=1, lambda_3=5):
    reward = (lambda_1 * cos_yaw_diff) - (lambda_2 * dist) - (lambda_3 * collision)
    return reward
