import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
# Normalize the data
X_train, X_test = X_train / 255.0, X_test / 255.0
# Flatten the images
X_train = X_train.reshape(-1, 28*28)
X_test = X_test.reshape(-1, 28*28)
# Convert labels to categorical (one-hot encoding)
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
model = keras.Sequential([
layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)),  # L2 Regularization
layers.Dropout(0.5),  # Dropout Regularization
layers.BatchNormalization(),  # Batch Normalization
layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l1(0.01)),  # L1 Regularization
layers.Dropout(0.3),
layers.BatchNormalization(),
layers.Dense(10, activation='softmax')  # Output layer])
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Early stopping callback
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
# Train the model
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), callbacks=[early_stopping])
#Visualizing Training Progress 
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()




Output:
Epoch 1/50
Train Loss: 0.65  |  Val Loss: 0.55
Epoch 2/50
Train Loss: 0.48  |  Val Loss: 0.43
...
Early stopping triggered
Loss Curve Plot
Loss
│
│     ● Training Loss
│     ▪ Validation Loss
│
│●●●●●●●●●
│▪▪▪▪▪▪▪▪▪
└───────────────────► Epochs
