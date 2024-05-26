import os
import cv2
import pygame
import math
import numpy as np
import numpy as np
import cv2


def process_img(image, dim_x=128, dim_y=128):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]

    resized_img = cv2.resize(array, (dim_x, dim_y), interpolation=cv2.INTER_AREA)
    img_gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

    # Equalizzazione dell'istogramma con CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_gray = clahe.apply(img_gray)

    # Filtro di Gauss per denoising
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # Rilevazione dei bordi con il filtro di Sobel
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel = np.hypot(sobelx, sobely)
    sobel = np.uint8(sobel / np.max(sobel) * 255)

    # Unisci l'immagine originale con i bordi rilevati
    combined = cv2.addWeighted(img_gray, 0.7, sobel, 0.3, 0)

    # Normalizza l'immagine con Z-score
    mean, std = combined.mean(), combined.std()
    normalized_img = (combined - mean) / std

    return normalized_img


def process_lidar(lidar_data, dim_x=128, dim_y=128):
    points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
    points = np.reshape(points, (int(points.shape[0] / 4), 4))

    lidar_image = np.zeros((dim_x, dim_y), dtype=np.float32)

    for point in points:
        x, y, z, _ = point
        if -10 < x < 10 and -10 < y < 10:
            i = int((x + 10) / 20 * dim_x)
            j = int((y + 10) / 20 * dim_y)
            lidar_image[i, j] = min(z, 1)

    # Filtro mediana per ridurre il rumore
    lidar_image = cv2.medianBlur(lidar_image, 5)

    # Interpolazione bicubica per una mappa più liscia
    lidar_image = cv2.resize(lidar_image, (dim_x, dim_y), interpolation=cv2.INTER_CUBIC)

    # Normalizza l'immagine
    lidar_image = cv2.normalize(lidar_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    lidar_image = np.uint8(lidar_image)

    return lidar_image


def draw_image(surface, image, blend=False):
    # Disegna un'immagine su una superficie di Pygame
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))  # Converte i dati dell'immagine in un array numpy
    array = np.reshape(array,
                       (image.height, image.width, 4))  # Ridimensiona array secondo le dimensioni dell'immagine
    array = array[:, :, :3]  # Seleziona solo i canali RGB, scartando alpha
    array = array[:, :, ::-1]  # Inverte l'ordine dei canali per adattarsi alla convenzione BGR di Pygame
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))  # Crea una superficie di Pygame

    if blend:
        image_surface.set_alpha(100)  # Imposta la trasparenza dell'immagine se richiesto
    surface.blit(image_surface, (0, 0))  # Disegna l'immagine sulla superficie di Pygame


def draw_depth_image(display, image):
    array = np.array(image * 255, dtype=np.uint8)
    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    display.blit(surface, (0, 128))


def draw_segmentation_image(display, image):
    # Normalizza l'immagine di segmentazione nell'intervallo [0, 255] e la converte in tipo CV_8U
    normalized_image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    normalized_image = np.uint8(normalized_image)

    # Converte l'immagine di segmentazione in un'immagine a tre canali
    colored_segmentation = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2BGR)

    # Applica una mappa dei colori alla segmentazione
    colored_segmentation = cv2.applyColorMap(colored_segmentation, cv2.COLORMAP_JET)

    # Converte l'immagine nel formato corretto per Pygame
    array = np.array(colored_segmentation, dtype=np.uint8)
    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

    # Disegna l'immagine sulla superficie di Pygame
    display.blit(surface, (0, 256))


def draw_lidar_image(display, image):
    array = np.array(image * 255, dtype=np.uint8)
    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    display.blit(surface, (0, 384))


def draw_text(surface, text, position):
    font = pygame.font.SysFont(None, 24)
    text_surface = font.render(text, True, (255, 255, 255))
    surface.blit(text_surface, position)


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
    if throttle > 0:
        brake = 0
    return throttle, brake
