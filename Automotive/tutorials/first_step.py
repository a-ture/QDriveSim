"""
Esecuzione delle istruzioni della sezione "Fist Step" della documentazione di CARLA
"""
import carla
import random


"""
Per muoversi all'interno del client usiamo i seguenti tasti della tastiera: 
Q - move upwards (towards the top edge of the window)
E - move downwards (towards the lower edge of the window)

W - move forwards

S - move backwards
A - move left
D - move right
"""
import carla
import random

def main():
    # Connect to the CARLA server
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)  # seconds
    print("CARLA client connected.")

    # Load a specific map
    world = client.load_world('Town05')

    # Retrieve the world and blueprint library
    blueprint_library = world.get_blueprint_library()

    # Spawn NPCs
    vehicle_blueprints = blueprint_library.filter('vehicle.*')
    spawn_points = world.get_map().get_spawn_points()
    for i in range(50):  # Spawn 50 random vehicles
        bp = random.choice(vehicle_blueprints)
        spawn_point = random.choice(spawn_points)
        npc_vehicle = world.try_spawn_actor(bp, spawn_point)
        if npc_vehicle:
            npc_vehicle.set_autopilot(True)

    # Add an ego vehicle
    ego_bp = blueprint_library.find('vehicle.lincoln.mkz_2020')
    ego_bp.set_attribute('role_name', 'hero')
    ego_vehicle = world.try_spawn_actor(ego_bp, random.choice(spawn_points))

    # Attach a camera sensor to the ego vehicle
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_transform = carla.Transform(carla.Location(z=1.5))  # Adjust the height as needed
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)
    camera.listen(lambda image: image.save_to_disk('out/%06d.png' % image.frame))

    # Run the simulation for some time
    import time
    time.sleep(30)  # Run simulation for 30 seconds

    # Cleanup
    camera.destroy()
    ego_vehicle.destroy()
    print("Simulation ended and cleaned up.")

if __name__ == '__main__':
    main()
