import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Carregar dados
df = pd.read_csv("luxdata.csv")

X = df[["lux"]].values

# Normalizar igual ao treino
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Carregar modelo
model = tf.keras.models.load_model("lux_model.keras")

def representative_dataset():
    for i in range(100):
        yield [X_scaled[i:i+1].astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_quant_model = converter.convert()

with open("lux_model_int8.tflite", "wb") as f:
    f.write(tflite_quant_model)

print("Modelo INT8 convertido com sucesso!")
