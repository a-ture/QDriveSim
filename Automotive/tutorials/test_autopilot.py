import carla
import random
import time

random.seed(42)


def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    print("Connected to CARLA server.")

    # Load a map
    world = client.load_world('Town03')

    if not world:
        print("CARLA world not available. Please check the CARLA server.")
        return

    # Set initial weather conditions
    world.set_weather(carla.WeatherParameters(
        cloudiness=10.0,
        precipitation=0.0,
        sun_altitude_angle=70.0
    ))

    blueprint_library = world.get_blueprint_library()

    # Set up the Traffic Manager
    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_global_distance_to_leading_vehicle(2.0)
    traffic_manager.set_synchronous_mode(True)

    # Spawn vehicles
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

    # Add an ego vehicle
    ego_bp = blueprint_library.find('vehicle.tesla.model3')
    ego_bp.set_attribute('role_name', 'hero')
    ego_vehicle = world.try_spawn_actor(ego_bp, random.choice(spawn_points))
    if not ego_vehicle:
        print("Failed to spawn ego vehicle.")
        return

    ego_vehicle.set_autopilot(True, traffic_manager.get_port())

    # Attach a camera
    cameras = []
    cam_bp = blueprint_library.find('sensor.camera.rgb')
    cam_bp.set_attribute("image_size_x", "800")
    cam_bp.set_attribute("image_size_y", "600")
    cam_location = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(cam_bp, cam_location, attach_to=ego_vehicle)
    cameras.append(camera)
    camera.listen(lambda image: image.save_to_disk('output/%06d.png' % image.frame))

    # Spawn walkers
    walkers = []
    walker_bp = blueprint_library.filter('walker.pedestrian.*')
    for i in range(10):
        spawn_point = random.choice(spawn_points)
        walker = world.try_spawn_actor(random.choice(walker_bp), spawn_point)
        if walker:
            walkers.append(walker)

    # Weather change logic
    changes = [
        (carla.WeatherParameters(cloudiness=0.0, precipitation=0.0, sun_altitude_angle=90.0), 50),
        (carla.WeatherParameters(cloudiness=50.0, precipitation=10.0, sun_altitude_angle=60.0), 100),
        (carla.WeatherParameters(cloudiness=80.0, precipitation=30.0, sun_altitude_angle=30.0), 150),
        (carla.WeatherParameters(cloudiness=90.0, precipitation=70.0, sun_altitude_angle=10.0), 200),
        (carla.WeatherParameters(cloudiness=100.0, precipitation=100.0, sun_altitude_angle=0.0), 250),
        (carla.WeatherParameters(cloudiness=20.0, precipitation=0.0, sun_altitude_angle=80.0), 300)
    ]

    # Simulation loop
    spectator = world.get_spectator()
    start_time = time.time()
    change_index = 0

    try:
        while time.time() - start_time < 300:
            current_time = time.time() - start_time
            if current_time > changes[change_index][1]:
                change_index = (change_index + 1) % len(changes)
                world.set_weather(changes[change_index][0])
                print(f"Weather changed to: {changes[change_index][0]}")

            vehicle_transform = ego_vehicle.get_transform()
            world.tick()
            # Aggiorna la posizione dello spectator
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


    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Clean up all actors
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
        print("Simulation ended.")


if __name__ == '__main__':
    main()
