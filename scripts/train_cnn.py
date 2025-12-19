import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from prepare_dataset import train_generator, val_generator

# Definir modelo CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Salida binaria: melanoma o benigno
])

# Compilar modelo
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Mostrar resumen del modelo
model.summary()

# Entrenar modelo
EPOCHS = 10  # Puedes aumentar después
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# Guardar modelo entrenado
model.save(r"C:\Users\Edgar G\Downloads\melanoma_project\melanoma_cnn_model.h5")
print("Modelo entrenado y guardado correctamente.")
