import matplotlib.pyplot as plt
import json
import os

# Ruta del historial
HISTORY_PATH = "results/history.json"

# Crear carpeta results si no existe
os.makedirs("results", exist_ok=True)

# Cargar historial
with open(HISTORY_PATH, "r") as f:
    history = json.load(f)

# ===============================
# GRÁFICA ACCURACY
# ===============================
plt.figure()
plt.plot(history["accuracy"], label="Entrenamiento")
plt.plot(history["val_accuracy"], label="Validación")
plt.title("Accuracy del modelo")
plt.xlabel("Épocas")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("results/accuracy.png")
plt.close()

# ===============================
# GRÁFICA LOSS
# ===============================
plt.figure()
plt.plot(history["loss"], label="Entrenamiento")
plt.plot(history["val_loss"], label="Validación")
plt.title("Loss del modelo")
plt.xlabel("Épocas")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("results/loss.png")
plt.close()

print("✅ Gráficas guardadas en la carpeta results/")