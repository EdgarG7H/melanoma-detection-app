import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Carpeta con las imágenes
DATA_DIR = r"C:\Users\Edgar G\Downloads\melanoma_project\data\all_images"

# CSV con etiquetas
LABELS_CSV = r"C:\Users\Edgar G\Downloads\melanoma_project\data\labels.csv"

# Leer CSV de etiquetas
df = pd.read_csv(LABELS_CSV)

# Asegurarse de que los nombres tengan .jpg
df['image_id'] = df['image_id'].apply(lambda x: x if x.lower().endswith('.jpg') else x + '.jpg')

# Convertir etiquetas a string
df['label'] = df['label'].astype(str)

# Dividir dataset
train_df = df.sample(frac=0.8, random_state=42)
val_df = df.drop(train_df.index)

# Generadores
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, vertical_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=DATA_DIR,
    x_col='image_id',
    y_col='label',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=DATA_DIR,
    x_col='image_id',
    y_col='label',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)
