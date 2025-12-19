from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image

# ===============================
# CONFIGURACIÓN DE RUTAS
# ===============================
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_ROOT)

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "mobilenetv2.keras")
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, "static", "uploads")

IMAGE_SIZE = (224, 224)

# ===============================
# INICIALIZAR FLASK
# ===============================
app = Flask(
    __name__,
    template_folder=os.path.join(PROJECT_ROOT, "templates"),
    static_folder=os.path.join(PROJECT_ROOT, "static")
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ===============================
# CARGAR MODELO
# ===============================
print("📌 Cargando modelo...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Modelo cargado correctamente")

# ===============================
# FUNCIONES
# ===============================
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(img_path):
    img_array = preprocess_image(img_path)
    prob = model.predict(img_array)[0][0]

    if prob < 0.3:
        clase = "BENIGNO"
    elif prob < 0.6:
        clase = "RIESGO MODERADO"
    else:
        clase = "MELANOMA (ALTO RIESGO)"

    return clase, float(prob)

# ===============================
# RUTA PRINCIPAL
# ===============================
@app.route("/", methods=["GET", "POST"])
def index():
    resultado = None
    probabilidad = None
    imagen = None
    error = None

    if request.method == "POST":
        if "image" not in request.files:
            error = "No se subió ninguna imagen"
        else:
            file = request.files["image"]

            if file.filename == "":
                error = "Archivo inválido"
            else:
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
                file.save(filepath)

                resultado, probabilidad = predict_image(filepath)
                imagen = file.filename

    return render_template(
        "index.html",
        resultado=resultado,
        probabilidad=probabilidad,
        imagen=imagen,
        error=error
    )

# ===============================
# EJECUTAR APP
# ===============================
if __name__ == "__main__":
    app.run(debug=True)