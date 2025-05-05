import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os
from tensorflow.keras.preprocessing import image
import numpy as np
train_dir = "D:/SJIT/DL/LAB/at/train"
test_dir = "D:/SJIT/DL/LAB/at/test"
img_height, img_width = 224, 224
num_classes = len(os.listdir(train_dir))
datagen = ImageDataGenerator( rescale=1./255, validation_split=0.2)
train_generator = datagen.flow_from_directory(train_dir,
 target_size=(224,224), batch_size=20,
class_mode='categorical',subset='training',shuffle=True)
Found 236 images belonging to 2 classes.
validation_generator = datagen.flow_from_directory(train_dir,
target_size=(224,224), batch_size=20, class_mode='categorical',subset='validation',
shuffle=False)
Found 58 images belonging to 2 classes.
model = Sequential([
 Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
 MaxPooling2D((2, 2)),
 Conv2D(64, (3, 3), activation='relu'),
 MaxPooling2D((2, 2)),
 Conv2D(64, (3, 3), activation='relu'),
 MaxPooling2D((2, 2)),
 Conv2D(64, (3, 3), activation='relu'),
 MaxPooling2D((2, 2)),
 Conv2D(64, (3, 3), activation='relu'),
 Flatten(),
 Dense(64, activation='relu'),
 Dense(num_classes, activation='softmax')])
model.compile(optimizer='adam',loss='categorical_crossentropy',
metrics=['accuracy'])
model.fit(train_generator, epochs=10, validation_data=validation_generator)
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
