import os
import cv2
import pygame
import math
import numpy as np


def process_img(image, dim_x=128, dim_y=128):
    # Processa l'immagine ricevuta da Carla
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))  # Converte i dati dell'immagine in un array numpy
    array = np.reshape(array,
                       (image.height, image.width, 4))  # Ridimensiona l'array secondo le dimensioni dell'immagine
    array = array[:, :, :3]  # Seleziona solo i canali RGB, scartando l'alpha
    array = array[:, :, ::-1]  # Inverte l'ordine dei canali per adattarsi alla convenzione BGR di OpenCV

    dim = (dim_x, dim_y)  # Dimensioni desiderate per l'immagine
    resized_img = cv2.resize(array, dim, interpolation=cv2.INTER_AREA)  # Ridimensiona l'immagine
    img_gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)  # Converte l'immagine in scala di grigi
    scaled_img = img_gray / 255.  # Scala i valori dei pixel nell'intervallo [0, 1]

    # Normalizza l'immagine
    mean, std = 0.5, 0.5
    normalized_img = (scaled_img - mean) / std

    return normalized_img


def draw_image(surface, image, blend=False):
    # Disegna un'immagine su una superficie di Pygame
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))  # Converte i dati dell'immagine in un array numpy
    array = np.reshape(array,
                       (image.height, image.width, 4))  # Ridimensiona l'array secondo le dimensioni dell'immagine
    array = array[:, :, :3]  # Seleziona solo i canali RGB, scartando l'alpha
    array = array[:, :, ::-1]  # Inverte l'ordine dei canali per adattarsi alla convenzione BGR di Pygame
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))  # Crea una superficie di Pygame dall'array
    if blend:
        image_surface.set_alpha(100)  # Imposta la trasparenza dell'immagine se richiesto
    surface.blit(image_surface, (0, 0))  # Disegna l'immagine sulla superficie di Pygame


def get_font():
    # Restituisce un font di Pygame
    fonts = [x for x in pygame.font.get_fonts()]  # Ottiene la lista dei font disponibili
    default_font = 'ubuntumono'  # Font predefinito
    font = default_font if default_font in fonts else fonts[
        0]  # Se il font predefinito non è disponibile, usa il primo della lista
    font = pygame.font.match_font(font)  # Trova il percorso del font
    return pygame.font.Font(font, 14)  # Crea il font e imposta la dimensione a 14


def should_quit():
    # Controlla se l'utente vuole chiudere il programma
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False


def get_speed(vehicle):
    """
    Calcola la velocità di un veicolo in Km/h.
        :param vehicle: il veicolo di cui calcolare la velocità
        :return: la velocità come float in Km/h
    """
    vel = vehicle.get_velocity()  # Ottiene la velocità del veicolo

    return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)  # Calcola la velocità totale in Km/h


def correct_yaw(x):
    # Corregge l'angolo di yaw (orientamento) portandolo nell'intervallo [0, 360)
    return ((x % 360) + 360) % 360


def create_folders(folder_names):
    # Crea le cartelle specificate se non esistono già
    for directory in folder_names:
        if not os.path.exists(directory):
            # Se non esiste, crea la cartella
            os.makedirs(directory)


#  una logica per bilanciare le azioni di throttle e brake
def balance_throttle_brake(throttle, brake):
    if throttle > 0 and brake > 0:
        brake = 0
    return throttle, brake
