import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import queue


def _detect_lane_invasion(sensor):
    # Rileva una collisione
    # Questa funzione non è completamente allineata con gli altri sensori, da correggere in futuro
    try:
        data = sensor.get(block=False)
        return data
    except queue.Empty:
        return None


def _detect_collision(sensor):
    # This collision is not fully aligned with other sensors, fix later
    try:
        data = sensor.get(block=False)
        return data
    except queue.Empty:
        return None


class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None
        self.collisions = []

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        try:
            # Esegue un tick del mondo
            self.frame = self.world.tick()

            # Recupera i dati da ciascuna coda dei sensori
            data = [self._retrieve_data(q, timeout) for q in self._queues[:-2]]

            # Verifica che tutti i dati abbiano lo stesso frame
            assert all(x.frame == self.frame for x in data)

            # Rileva le invasioni di corsia
            lane_invasion = _detect_lane_invasion(self._queues[-2])

            # Rileva le collisioni
            collision = _detect_collision(self._queues[-1])

            return data + [lane_invasion, collision]
        except queue.Empty:
            print("Empty queue")
            return None, None, None, None, None, None, None, None, None

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data

