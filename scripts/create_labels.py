import pandas as pd
import os

# Definir ruta al CSV original
csv_path = r"C:\Users\Edgar G\Downloads\melanoma_project\data\HAM10000_metadata.csv"

# Verificar que el archivo exista
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"No se encontró el archivo CSV en: {csv_path}")

# Leer CSV original
metadata = pd.read_csv(csv_path)

# Mapear etiquetas a binario: melanoma = 1, otras lesiones = 0
metadata['label'] = metadata['dx'].apply(lambda x: 1 if x.lower() == 'mel' else 0)

# Guardar CSV simplificado
output_csv = r"C:\Users\Edgar G\Downloads\melanoma_project\data\labels.csv"
metadata[['image_id','label']].to_csv(output_csv, index=False)

print("Archivo labels.csv creado correctamente en:")
print(output_csv)
print(metadata[['image_id','label']].head())
