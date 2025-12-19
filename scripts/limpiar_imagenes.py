import os
from PIL import Image

base_path = r"D:\melanoma_project\data"

def limpiar_carpeta(path):
    print(f"Revisando: {path}")
    for folder in ["benign", "melanoma"]:
        folder_path = os.path.join(path, folder)
        if not os.path.exists(folder_path):
            continue

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)

            try:
                with Image.open(img_path) as img:
                    img.verify()  # valida imagen
            except Exception:
                print(f"❌ Imagen dañada eliminada: {img_path}")
                os.remove(img_path)

# Limpia train y test
limpiar_carpeta(os.path.join(base_path, "train"))
limpiar_carpeta(os.path.join(base_path, "test"))

print("✅ Limpieza completada.")