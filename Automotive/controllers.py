import os
import sys
import glob

# Aggiunge il percorso della libreria Carla al path di sistema
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import numpy as np
from collections import deque
from utils import get_speed

class PIDLongitudinalController():
    """
    PIDLongitudinalController implementa il controllo longitudinale utilizzando un PID.
    """

    def __init__(self, vehicle, max_throttle=0.75, max_brake=0.3, K_P=1.0, K_I=0.0, K_D=0.0, dt=0.03):
        """
        Metodo costruttore.
            :param vehicle: veicolo su cui applicare la logica del local planner
            :param K_P: termine proporzionale
            :param K_D: termine differenziale
            :param K_I: termine integrale
            :param dt: differenziale di tempo in secondi
        """
        self._vehicle = vehicle
        self.max_throttle = max_throttle  # Massima accelerazione
        self.max_brake = max_brake  # Massima decelerazione
        self._k_p = K_P  # Coefficiente proporzionale
        self._k_i = K_I  # Coefficiente integrale
        self._k_d = K_D  # Coefficiente differenziale
        self._dt = dt  # Differenziale di tempo
        self._error_buffer = deque(maxlen=10)  # Buffer per memorizzare gli errori

    def run_step(self, target_speed, debug=False):
        """
        Esegue un passaggio del controllo longitudinale per raggiungere una velocità target.
            :param target_speed: velocità target in Km/h
            :param debug: booleano per il debug
            :return: controllo dell'acceleratore
        """
        current_speed = get_speed(self._vehicle)  # Ottiene la velocità attuale del veicolo

        if debug:
            print('Current speed = {}'.format(current_speed))

        acceleration = self._pid_control(target_speed, current_speed)  # Calcola l'accelerazione necessaria
        control = carla.VehicleControl()
        if acceleration >= 0.0:
            control.throttle = min(acceleration, self.max_throttle)  # Applica l'acceleratore
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(abs(acceleration), self.max_brake)  # Applica il freno
        return control

    def _pid_control(self, target_speed, current_speed):
        """
        Calcola l'accelerazione/freno del veicolo basandosi sulle equazioni PID
            :param target_speed: velocità target in Km/h
            :param current_speed: velocità attuale del veicolo in Km/h
            :return: controllo dell'acceleratore/freno
        """

        error = target_speed - current_speed  # Calcola l'errore
        self._error_buffer.append(error)  # Aggiunge l'errore al buffer

        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt  # Calcola la derivata dell'errore
            _ie = sum(self._error_buffer) * self._dt  # Calcola l'integrale dell'errore
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._k_p * error) + (self._k_d * _de) + (self._k_i * _ie), -1.0, 1.0)  # Restituisce l'accelerazione/freno

    def change_parameters(self, K_P, K_I, K_D, dt):
        """Cambia i parametri del PID"""
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt
