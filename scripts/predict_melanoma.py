import argparse
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

def load_selected_model(model_name):
    model_path = f"models/{model_name}.keras"
    print(f"\n📌 Cargando modelo: {model_name}...")
    return tf.keras.models.load_model(model_path)

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(model, image_array):
    pred = model.predict(image_array)[0][0]

    # 🔥 NUEVA CLASIFICACIÓN POR RIESGO (UMBRAL MÉDICO)
    if pred >= 0.30:
        clase = "ALTO RIESGO DE MELANOMA"
    elif pred >= 0.15:
        clase = "RIESGO MODERADO"
    else:
        clase = "BENIGNO"

    return clase, float(pred)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Ruta de la imagen")
    parser.add_argument(
        "--model",
        required=True,
        choices=["cnn_simple", "mobilenetv2", "resnet50"]
    )
    args = parser.parse_args()

    model = load_selected_model(args.model)
    img = preprocess_image(args.image)
    clase, prob = predict(model, img)

    print("\n===========================")
    print(f" Modelo: {args.model}")
    print(f" Predicción: {clase}")
    print(f" Probabilidad: {prob:.4f}")
    print("===========================\n")