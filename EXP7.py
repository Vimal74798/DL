import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt 

# Load MNIST dataset
(x_train, _), (x_test, _) = mnist.load_data()
 
# Normalize and reshape the data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = np.reshape(x_train, (len(x_train), 28, 28))
x_test = np.reshape(x_test, (len(x_test), 28, 28))
# Define the model
latent_dim = 32
inputs = Input(shape=(28, 28))
encoded = LSTM(latent_dim)(inputs)
decoded = RepeatVector(28)(encoded)
decoded = LSTM(28, return_sequences=True)(decoded)
sequence_autoencoder = Model(inputs, decoded)
# Compile the model
sequence_autoencoder.compile(optimizer='adam', loss='mean_squared_error')
# Print the model summary
sequence_autoencoder.summary()
 
# Train the model
sequence_autoencoder.fit(x_train, x_train, epochs=10, batch_size=128,
shuffle=True, validation_data=(x_test, x_test))  
 
 
# Generate reconstructed images
decoded_images = sequence_autoencoder.predict(x_test)   
# Plot original and reconstructed images
n = 10 # Number of images to display
plt.figure(figsize=(20, 4))
for i in range(n):
 # Original images
 ax = plt.subplot(2, n, i + 1)
 plt.imshow(x_test[i].reshape(28, 28))
 plt.gray()
 ax.get_xaxis().set_visible(True)
 ax.get_yaxis().set_visible(True) 


 
# Reconstructed images
 ax = plt.subplot(2, n, i + 1 + n)
 plt.imshow(decoded_images[i].reshape(28, 28))
 plt.gray()
 ax.get_xaxis().set_visible(False)
 ax.get_yaxis().set_visible(False)
plt.show()
