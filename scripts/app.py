import os
import gdown
import tensorflow as tf
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image
import numpy as np

# =========================
# CONFIGURACIÓN
# =========================
MODEL_PATH = "models/mobilenetv2.keras"
MODEL_URL = "https://drive.google.com/uc?export=download&id=10pZ8XKU0HywGVDKXrsBRdRX8H_NGHIvj"
IMG_SIZE = (224, 224)

# =========================
# DESCARGAR MODELO SI NO EXISTE
# =========================
os.makedirs("models", exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print("📌 Descargando modelo desde Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

print("✅ Cargando modelo...")
model = tf.keras.models.load_model(MODEL_PATH)

# =========================
# FLASK APP
# =========================
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        img_file = request.files["image"]

        img_path = os.path.join("static", img_file.filename)
        img_file.save(img_path)

        img = image.load_img(img_path, target_size=IMG_SIZE)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        result = model.predict(img_array)[0][0]

        prediction = "Melanoma" if result > 0.5 else "No Melanoma"

    return render_template("index.html", prediction=prediction)

# =========================
# RUN
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)