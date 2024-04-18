import carla
import time
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
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)

        spectator = world.get_spectator()

        # Inizializza il veicolo con un comando di accelerazione
        vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0.0, brake=0.0))

        for _ in range(800):  # circa 40 secondi
            world.tick()
            vehicle_transform = vehicle.get_transform()

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
                vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=-0.5, brake=0.0))
            elif _ == 400:  # Dopo 20 secondi
                vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0.5, brake=0.0))
            elif _ == 600:  # Dopo 30 secondi, torna dritto
                vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0.0, brake=0.0))

        vehicle.apply_control(carla.VehicleControl(brake=1.0))
        world.tick()

    finally:
        settings.synchronous_mode = False
        world.apply_settings(settings)
        if vehicle is not None:
            vehicle.destroy()

if __name__ == '__main__':
    main()
