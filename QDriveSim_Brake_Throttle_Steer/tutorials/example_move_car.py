import carla
import random


def main():
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()

        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.audi.a2')
        spawn_point = random.choice(world.get_map().get_spawn_points())
        ego_vehicle = world.spawn_actor(vehicle_bp, spawn_point)

        spectator = world.get_spectator()

        # Attacca una telecamera
        cameras = []
        cam_bp = blueprint_library.find('sensor.camera.rgb')
        cam_bp.set_attribute("image_size_x", "800")
        cam_bp.set_attribute("image_size_y", "600")
        cam_location = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(cam_bp, cam_location, attach_to=ego_vehicle)
        cameras.append(camera)
        camera.listen(lambda image: image.save_to_disk('output/%06d.png' % image.frame))

        # Inizializza il veicolo con un comando di accelerazione
        ego_vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0.0, brake=0.0))

        for _ in range(800):  # circa 40 secondi
            world.tick()
            vehicle_transform = ego_vehicle.get_transform()

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

            # Decide di girare a sinistra dopo 10 secondi e a destra dopo 20 secondi
            if _ == 200:  # Dopo 10 secondi
                ego_vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=-0.5, brake=0.0))
            elif _ == 400:  # Dopo 20 secondi
                ego_vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0.5, brake=0.0))
            elif _ == 600:  # Dopo 30 secondi, torna dritto
                ego_vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0.0, brake=0.0))

        ego_vehicle.apply_control(carla.VehicleControl(brake=1.0))
        world.tick()

    finally:
        settings.synchronous_mode = False
        world.apply_settings(settings)
        if ego_vehicle is not None:
            ego_vehicle.destroy()


if __name__ == '__main__':
    main()
