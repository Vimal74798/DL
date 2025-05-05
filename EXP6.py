import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Set your custom dataset path
train_dir = "D:/SJIT/DL/LAB/at/train"
test_dir = "D:/SJIT/DL/LAB/at/test"
# Define hyperparameters
img_width, img_height = 224, 224
batch_size = 32
num_classes = 2 # The number of classes in your dataset
epochs = 10
# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
 rescale=1./255,
 rotation_range=20,
 width_shift_range=0.2,
 height_shift_range=0.2,
 shear_range=0.2,
 zoom_range=0.2,
 horizontal_flip=True,
 fill_mode='nearest'
)
train_generator = train_datagen.flow_from_directory(
 train_data_dir,
 target_size=(img_width, img_height),
 batch_size=batch_size,
 class_mode='categorical')
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
 validation_data_dir,
 target_size=(img_width, img_height),
 batch_size=batch_size,
 class_mode='categorical')
# Load the pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False,
input_shape=(img_width, img_height, 3))
# Create a custom classification model on top of VGG16
model = Sequential()
model.add(base_model) # Add the pre-trained VGG16 model
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax')
# Freeze the pre-trained layers
for layer in base_model.layers:
 layer.trainable = False
# Compile the model
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy',
metrics=['accuracy'])
# Train the model
model.fit(train_generator, epochs=epochs, validation_data=validation_generator)
# Optionally, you can unfreeze and fine-tune some layers
for layer in base_model.layers[-4:]:
 layer.trainable = True
model.compile(optimizer=Adam(lr=0.00001), loss='categorical_crossentropy',
metrics=['accuracy'])
# Continue training for additional epochs
model.fit(train_generator, epochs=epochs, validation_data=validation_generator)
img_path = "D:\\SJIT\\DL\\LAB\\lp.jpg" # Replace with the path to your image
img = image.load_img(img_path, target_size=(224, 224)) # Adjust target_size if
needed
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = img / 255.0
predictions = model.predict(img)
1/1 [==============================] - 0s 140ms/step
predicted_class = np.argmax(predictions)
class_labels = {0: 'apples', 1: 'tomatoes'}
predicted_label = class_labels[predicted_class]
print(f"Predicted class: {predicted_class} (Label: {predicted_label})")


OUTPUT:
Predicted Class: apple
