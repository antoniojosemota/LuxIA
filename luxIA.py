import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('luxdata.csv')
X = df[["lux"]].values
y = df[["classe"]].values

## Converter as classes para one-hot encoding onde transforma as classes em vetores binarios
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y)

## Normalizando dados

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

## Treinando o modelo
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)
model.fit(X_train, y_train, epochs=300, callbacks=[early_stop], verbose=1)
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')

# Converter para TinyML
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("luxia.tflite", "wb") as f:
    f.write(tflite_model)

print("Modelo salvo!")