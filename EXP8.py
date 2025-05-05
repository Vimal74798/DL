import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Reshape, Flatten
from tensorflow.keras.layers import BatchNormalization, LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
# Load MNIST data
(x_train, _), (_, _) = mnist.load_data()
# Normalize and reshape data
x_train = x_train / 127.5 - 1.0
x_train = np.expand_dims(x_train, axis=3)
# Define the generator model
generator = Sequential()
generator.add(Dense(128 * 7 * 7, input_dim=100))
generator.add(LeakyReLU(0.2))
generator.add(Reshape((7, 7, 128)))
generator.add(BatchNormalization())
generator.add(Flatten())
generator.add(Dense(28 * 28 * 1, activation='tanh'))
generator.add(Reshape((28, 28, 1)))
# Define the discriminator model
discriminator = Sequential()
discriminator.add(Flatten(input_shape=(28, 28, 1)))
discriminator.add(Dense(128))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dense(1, activation='sigmoid'))
# Compile the discriminator
discriminator.compile(loss='binary_crossentropy',
optimizer=Adam(learning_rate=0.0002, beta_1=0.5), metrics=['accuracy'])
# Freeze the discriminator during GAN training
discriminator.trainable = False
# Combine generator and discriminator into a GAN model
gan = Sequential()
gan.add(generator)
gan.add(discriminator)
# Compile the GAN
gan.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002,
beta_1=0.5))
# Function to train the GAN
def train_gan(epochs=1, batch_size=128):
 batch_count = x_train.shape[0] // batch_size
 for e in range(epochs):
 for _ in range(batch_count):
 noise = np.random.normal(0, 1, size=[batch_size, 100])
 generated_images = generator.predict(noise)
 image_batch = x_train[np.random.randint(0, x_train.shape[0],
size=batch_size)]
 X = np.concatenate([image_batch, generated_images])
 y_dis = np.zeros(2 * batch_size)
 y_dis[:batch_size] = 0.9 # Label smoothing
 discriminator.trainable = True
 d_loss = discriminator.train_on_batch(X, y_dis)
 noise = np.random.normal(0, 1, size=[batch_size, 100])
 y_gen = np.ones(batch_size)
 discriminator.trainable = False
 g_loss = gan.train_on_batch(noise, y_gen)
 print(f"Epoch {e+1}/{epochs}, Discriminator Loss: {d_loss[0]},
Generator Loss: {g_loss}")
# Train the GAN
train_gan(epochs=200, batch_size=128)
# Generate and plot some images
def plot_generated_images(epoch, examples=10, dim=(1, 10), figsize=(10, 1)):
 noise = np.random.normal(0, 1, size=[examples, 100])
 generated_images = generator.predict(noise)
 generated_images = generated_images.reshape(examples, 28, 28)
 plt.figure(figsize=figsize)
 for i in range(generated_images.shape[0]):
 plt.subplot(dim[0], dim[1], i+1)
 plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
 plt.axis('off')
 plt.tight_layout()
 plt.savefig(f'gan_generated_image_epoch_{epoch}.png')
# Plot generated images for a few epochs
for epoch in range(1, 10):
 plot_generated_images(epoch)
