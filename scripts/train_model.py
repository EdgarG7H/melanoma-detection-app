import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ===============================
# CONFIGURACIÓN GENERAL
# ===============================
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")

MODEL_DIR = "models"
RESULTS_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ===============================
# GENERADORES DE IMÁGENES
# ===============================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

validation_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

# ===============================
# MODELO: MobileNetV2
# ===============================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False  # Transfer Learning

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ===============================
# CALLBACKS
# ===============================
checkpoint = ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, "mobilenetv2.keras"),
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

callbacks = [checkpoint, early_stop]

# ===============================
# ENTRENAMIENTO
# ===============================
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ===============================
# GUARDAR HISTORIAL (GRÁFICAS)
# ===============================
history_data = {
    "accuracy": history.history["accuracy"],
    "val_accuracy": history.history["val_accuracy"],
    "loss": history.history["loss"],
    "val_loss": history.history["val_loss"]
}

with open(os.path.join(RESULTS_DIR, "history.json"), "w") as f:
    json.dump(history_data, f)

print("📊 Historial guardado en results/history.json")

# ===============================
# GUARDAR MODELO FINAL
# ===============================
model.save(os.path.join(MODEL_DIR, "mobilenetv2_final.keras"))
print("✅ Modelo entrenado y guardado correctamente")