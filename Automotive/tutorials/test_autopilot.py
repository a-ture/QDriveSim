import carla
import random
import threading
import time

# Imposta il seed per la riproducibilità dei risultati casuali
random.seed(42)


# Funzione principale
def main():
    # Variabili per tenere traccia delle statistiche della simulazione
    collision_count = 0  # Conteggio delle collisioni
    total_velocity = 0  # Velocità totale dell'ego vehicle
    total_time = 0  # Tempo totale della simulazione
    lane_invasion_count = 0  # Conteggio delle invasioni di corsia
    obstacle_count = 0  # Conteggio degli ostacoli
    simulation_running = threading.Event()  # Evento per controllare lo stato della simulazione

    # Connessione al server CARLA
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    print("Connected to CARLA server.")

    # Caricamento della mappa
    world = client.load_world('Town03')

    # Verifica se la mappa è stata caricata correttamente
    if not world:
        print("CARLA world not available. Please check the CARLA server.")
        return

    # Imposta le condizioni meteorologiche iniziali
    world.set_weather(carla.WeatherParameters(
        cloudiness=10.0,
        precipitation=0.0,
        sun_altitude_angle=70.0
    ))

    # Ottiene la libreria dei 'blueprint' per creare gli attori
    blueprint_library = world.get_blueprint_library()

    # Configura il Traffic Manager
    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_global_distance_to_leading_vehicle(2.0)
    traffic_manager.set_synchronous_mode(True)

    # Spawn dei veicoli
    vehicle_blueprints = blueprint_library.filter('vehicle.*')
    spawn_points = world.get_map().get_spawn_points()
    vehicles = []
    for _ in range(20):
        bp = random.choice(vehicle_blueprints)
        spawn_point = random.choice(spawn_points)
        vehicle = world.try_spawn_actor(bp, spawn_point)
        if vehicle:
            vehicle.set_autopilot(True, traffic_manager.get_port())
            vehicles.append(vehicle)

    # Aggiunge un ego vehicle
    ego_bp = blueprint_library.find('vehicle.tesla.model3')
    #hero indica che la macchina che controlliamo noi
    ego_bp.set_attribute('role_name', 'hero')

    ego_vehicle = world.try_spawn_actor(ego_bp,random.choice(spawn_points))
    if not ego_vehicle:
        print("Failed to spawn ego vehicle.")
        return

    ego_vehicle.set_autopilot(True, traffic_manager.get_port())

    # Attacca un sensore di collisione all'ego vehicle
    collision_sensor_bp = blueprint_library.find('sensor.other.collision')
    collision_sensor_location = carla.Transform(carla.Location(x=0, z=1.0))  # Regola la posizione se necessario
    collision_sensor = world.spawn_actor(collision_sensor_bp, collision_sensor_location, attach_to=ego_vehicle)
    collision_sensor.listen(lambda event: collision_event(event, collision_count))

    # Attacca un sensore di invasione di corsia all'ego vehicle
    lane_invasion_sensor_bp = blueprint_library.find('sensor.other.lane_invasion')
    lane_invasion_sensor_location = carla.Transform(carla.Location(x=0, z=1.0))  # Regola la posizione se necessario
    lane_invasion_sensor = world.spawn_actor(lane_invasion_sensor_bp, lane_invasion_sensor_location,
                                             attach_to=ego_vehicle)
    lane_invasion_sensor.listen(lambda event: lane_invasion_event(event, lane_invasion_count))

    # Attacca un sensore di ostacoli all'ego vehicle
    obstacle_sensor_bp = blueprint_library.find('sensor.other.obstacle')
    obstacle_sensor_location = carla.Transform(carla.Location(x=0, z=1.0))  # Regola la posizione se necessario
    obstacle_sensor = world.spawn_actor(obstacle_sensor_bp, obstacle_sensor_location, attach_to=ego_vehicle)
    obstacle_sensor.listen(lambda event: obstacle_event(event, obstacle_count))

    # Attacca una telecamera
    cameras = []
    cam_bp = blueprint_library.find('sensor.camera.rgb')
    cam_bp.set_attribute("image_size_x", "800")
    cam_bp.set_attribute("image_size_y", "600")
    cam_location = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(cam_bp, cam_location, attach_to=ego_vehicle)
    cameras.append(camera)
    camera.listen(lambda image: image.save_to_disk('output/%06d.png' % image.frame))

    # Spawn dei pedoni
    walkers = []
    walker_bp = blueprint_library.filter('walker.pedestrian.*')
    for i in range(10):
        spawn_point = random.choice(spawn_points)
        walker = world.try_spawn_actor(random.choice(walker_bp), spawn_point)
        if walker:
            walkers.append(walker)

    # Logica per il cambio delle condizioni meteorologiche
    changes = [
        (carla.WeatherParameters(cloudiness=0.0, precipitation=0.0, sun_altitude_angle=90.0), 50),
        (carla.WeatherParameters(cloudiness=50.0, precipitation=10.0, sun_altitude_angle=60.0), 100),
        (carla.WeatherParameters(cloudiness=80.0, precipitation=30.0, sun_altitude_angle=30.0), 150),
        (carla.WeatherParameters(cloudiness=90.0, precipitation=70.0, sun_altitude_angle=10.0), 200),
        (carla.WeatherParameters(cloudiness=100.0, precipitation=100.0, sun_altitude_angle=0.0), 250),
        (carla.WeatherParameters(cloudiness=20.0, precipitation=0.0, sun_altitude_angle=80.0), 300)
    ]

    # Loop di simulazione
    spectator = world.get_spectator()
    start_time = time.time()
    change_index = 0

    try:
        # Avvia un thread per terminare la simulazione dopo un minuto
        simulation_thread = threading.Thread(target=run_simulation, args=(simulation_running,))
        simulation_thread.start()

        while not simulation_running.is_set():
            current_time = time.time() - start_time
            if current_time > changes[change_index][1]:
                change_index = (change_index + 1) % len(changes)
                world.set_weather(changes[change_index][0])
                print(f"Weather changed to: {changes[change_index][0]}")

            world.tick()

            # Calcola la velocità dell'ego vehicle
            velocity = ego_vehicle.get_velocity().length()
            total_velocity += velocity
            total_time += 1

            # Aggiorna la posizione dello spettatore
            vehicle_transform = ego_vehicle.get_transform()
            spectator_transform = carla.Transform(
                carla.Location(
                    x=vehicle_transform.location.x - 6.5 * vehicle_transform.get_forward_vector().x,
                    y=vehicle_transform.location.y - 6.5 * vehicle_transform.get_forward_vector().y,
                    z=vehicle_transform.location.z + 2.5
                ),
                carla.Rotation(
                    pitch=-15,
                    yaw=vehicle_transform.rotation.yaw,
                    roll=0
                )
            )
            spectator.set_transform(spectator_transform)

            time.sleep(0.1)  # Aggiungi una piccola pausa per rallentare il loop

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Interrompe il thread della simulazione e aspetta che termini
        simulation_running.set()
        simulation_thread.join()

        # Pulisce tutti gli attori
        print("Destroying actors.")
        for vehicle in vehicles:
            vehicle.destroy()
        for camera in cameras:
            camera.stop()
            camera.destroy()
        for walker in walkers:
            walker.destroy()
        if ego_vehicle:
            ego_vehicle.destroy()
        # Distrugge i sensori
        if collision_sensor:
            collision_sensor.destroy()
        if lane_invasion_sensor:
            lane_invasion_sensor.destroy()
        if obstacle_sensor:
            obstacle_sensor.destroy()
        print("Simulation ended.")

        # Calcola la velocità media
        average_velocity = total_velocity / total_time if total_time > 0 else 0

        # Stampa le statistiche
        print("\nSimulation statistics:")
        print(f"Total collisions: {collision_count}")
        print(f"Average velocity: {average_velocity} m/s")
        print(f"Total lane invasions: {lane_invasion_count}")
        print(f"Total obstacles detected: {obstacle_count}")


# Funzione per eseguire la simulazione per un certo periodo di tempo
def run_simulation(simulation_running_event):
    # Esegue la simulazione per 60 secondi
    time.sleep(60)
    # Imposta l'evento per indicare la fine della simulazione
    simulation_running_event.set()


# Funzione chiamata quando viene rilevata una collisione
def collision_event(event, collision_count):
    collision_count += 1
    print(f"Collision detected! Total collisions: {collision_count}")


# Funzione chiamata quando viene rilevata un'invasione di corsia
def lane_invasion_event(event, lane_invasion_count):
    lane_invasion_count += 1
    print(f"Lane invasion detected! Total invasions: {lane_invasion_count}")


# Funzione chiamata quando viene rilevato un ostacolo
def obstacle_event(event, obstacle_count):
    obstacle_count += 1
    print(f"Obstacle detected! Total obstacles: {obstacle_count}")


# Esegue la funzione main quando il file viene eseguito direttamente
if __name__ == '__main__':
    main()
