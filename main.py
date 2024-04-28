import tensorflow as tf
import os
import random
import numpy as np
import imageio
import moviepy.editor as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, Conv2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)

def preprocess_image(file):
    img = image.load_img(file, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

train_generator = train_datagen.flow_from_directory(
        "train",
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical')


model = Sequential([
  Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
  MaxPooling2D((2, 2)),
  Flatten(),
  Dense(250, activation="relu"),
  Dense(2, activation="softmax"),
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])

model.fit(train_generator, epochs=50)

model.save("model.keras")

user = ""
x = ["old", "young"]
while user != "exit":
  user = input("entrez le chemi  vers le fichier : ")

  predicted = model.predict(preprocess_image(x))
  print(x[predicted.argmex()], x)