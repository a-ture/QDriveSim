import cv2

# Carica un'immagine
img = cv2.imread('../output/000970.png')
print(img.shape)  # Stampa le dimensioni dell'immagine

# Ridimensiona l'immagine
scale_percent = 25
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (128, 128)  # Dimensioni desiderate
resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)  # Ridimensiona l'immagine
img_gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)  # Converte l'immagine in scala di grigi
print(img_gray.shape)  # Stampa le dimensioni dell'immagine in scala di grigi

# Visualizza l'immagine in scala di grigi
cv2.imshow('', img_gray)
cv2.waitKey(5000)  # Attende per 5 secondi
cv2.destroyAllWindows()  # Chiude le finestre
