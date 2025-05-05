import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras import Input
from keras.layers import Dense
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Print dataset shapes
print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_test:", y_test.shape)

# Display images of the first 10 digits in the training set
fig, axs = plt.subplots(2, 5, figsize=(12, 6), facecolor='white')
n = 0
for i in range(2):
    for j in range(5):
        axs[i, j].matshow(X_train[n], cmap='gray')
        axs[i, j].set_title(f"Label: {y_train[n]}")
        axs[i, j].axis('off')
        n += 1
plt.show()

# Reshape and normalize input data
X_train = X_train.reshape(60000, 784).astype("float32") / 255.0
X_test = X_test.reshape(10000, 784).astype("float32") / 255.0

# Print new shapes
print("New shape of X_train:", X_train.shape)
print("New shape of X_test:", X_test.shape)

# Design the Deep Feedforward Neural Network
model = Sequential(name="DFF-Model")

# Input Layer
model.add(Input(shape=(784,), name='Input-Layer'))

# Hidden Layers with He Normal Initialization
model.add(Dense(128, activation='relu', name='Hidden-Layer-1', kernel_initializer='he_normal'))
model.add(Dense(64, activation='relu', name='Hidden-Layer-2', kernel_initializer='he_normal'))
model.add(Dense(32, activation='relu', name='Hidden-Layer-3', kernel_initializer='he_normal'))

# Output Layer with Softmax Activation
model.add(Dense(10, activation='softmax', name='Output-Layer'))

# Compile the model
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, 
                    batch_size=32, epochs=5, 
                    validation_split=0.2, shuffle=True)

# Predict class labels on training data
pred_labels_tr = np.argmax(model.predict(X_train), axis=1)

# Predict class labels on test data
pred_labels_te = np.argmax(model.predict(X_test), axis=1)

# Print model summary
print("\nModel Summary")
model.summary()

# Evaluate model performance
print("\n---------- Evaluation on Training Data ----------")
print(classification_report(y_train, pred_labels_tr))

print("\n---------- Evaluation on Test Data ----------")
print(classification_report(y_test, pred_labels_te))
