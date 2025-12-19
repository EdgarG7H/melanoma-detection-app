import os
import shutil
import pandas as pd
from tqdm import tqdm

# Rutas principales
BASE_DIR = "data"
ALL_IMAGES_DIR = os.path.join(BASE_DIR, "all_images")
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")

LABELS_FILE = os.path.join(BASE_DIR, "labels.csv")

# Crear carpetas si no existen
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)
os.makedirs(os.path.join(TRAIN_DIR, "benign"), exist_ok=True)
os.makedirs(os.path.join(TRAIN_DIR, "malignant"), exist_ok=True)
os.makedirs(os.path.join(TEST_DIR, "benign"), exist_ok=True)
os.makedirs(os.path.join(TEST_DIR, "malignant"), exist_ok=True)

print("📁 Carpetas creadas correctamente.")

# Leer CSV
df = pd.read_csv(LABELS_FILE)
print(f"🔍 Total imágenes en labels.csv: {len(df)}")

# Renombrar columnas si es necesario
df = df.rename(columns={"image_id": "image", "label": "label"})

# Agregar extensión .jpg
df["image"] = df["image"].astype(str) + ".jpg"

# Dividir dataset
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)

print("🚚 Moviendo imágenes de entrenamiento...")

def mover_imagenes(df, destino_base):
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_name = row["image"]
        label = "benign" if row["label"] == 0 else "malignant"
        
        src = os.path.join(ALL_IMAGES_DIR, img_name)
        dst = os.path.join(destino_base, label, img_name)

        if os.path.exists(src):
            shutil.copy(src, dst)
        else:
            print(f"⚠️ Imagen no encontrada: {img_name}")

mover_imagenes(train_df, TRAIN_DIR)

print("🚚 Moviendo imágenes de prueba...")
mover_imagenes(test_df, TEST_DIR)

print("✅ Dataset organizado correctamente.")
