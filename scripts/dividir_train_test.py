import os
import shutil
import random
from tqdm import tqdm

# Ruta base del dataset
BASE_DIR = "D:/melanoma_project/data"

SORTED_DIR = os.path.join(BASE_DIR, "sorted")
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")

# Crear carpetas si no existen
for subdir in ["train/benign", "train/melanoma", "test/benign", "test/melanoma"]:
    os.makedirs(os.path.join(BASE_DIR, subdir), exist_ok=True)

# Proporción de test (20%)
test_ratio = 0.2

for class_name in ["benign", "melanoma"]:
    source_dir = os.path.join(SORTED_DIR, class_name)
    images = os.listdir(source_dir)
    random.shuffle(images)

    test_count = int(len(images) * test_ratio)
    test_images = images[:test_count]
    train_images = images[test_count:]

    print(f"\n📂 Clase: {class_name}")
    print(f"➡️ Total: {len(images)}  | Train: {len(train_images)} | Test: {len(test_images)}")

    # Mover a train
    print("   Moviendo a TRAIN...")
    for img in tqdm(train_images):
        shutil.copy(
            os.path.join(source_dir, img),
            os.path.join(TRAIN_DIR, class_name, img)
        )

    # Mover a test
    print("   Moviendo a TEST...")
    for img in tqdm(test_images):
        shutil.copy(
            os.path.join(source_dir, img),
            os.path.join(TEST_DIR, class_name, img)
        )

print("\n✅ División completada correctamente.")
