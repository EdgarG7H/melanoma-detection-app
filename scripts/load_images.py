import cv2
import matplotlib.pyplot as plt
import os

# Carpeta con las imágenes
DATA_DIR = r"C:\Users\Edgar G\Downloads\melanoma_project\data\all_images"

# Función para cargar y preprocesar imagen
def preprocess_image(path, target_size=(224,224)):
    img = cv2.imread(path)                 # Leer imagen
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir BGR a RGB
    img = cv2.resize(img, target_size)     # Redimensionar
    img = img / 255.0                      # Normalizar
    return img

# Tomar las primeras 5 imágenes de la carpeta
sample_files = os.listdir(DATA_DIR)[:5]

# Mostrar imágenes
for f in sample_files:
    path = os.path.join(DATA_DIR, f)
    img = preprocess_image(path)
    plt.imshow(img)
    plt.title(f)
    plt.axis('off')
    plt.show()
