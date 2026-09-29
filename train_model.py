import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

data_path = "dataset"
categories = sorted(os.listdir(data_path))
data = []
labels = []

for category in categories:
    folder_path = os.path.join(data_path, category)
    label = categories.index(category)

    for img in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img)
        image = cv2.imread(img_path)
        image = cv2.resize(image, (64, 64))
        data.append(image)
        labels.append(label)

data = np.array(data) / 255.0
labels = to_categorical(labels)

# CNN Model
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(categories), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(data, labels, epochs=10)

model.save("model/sign_model.h5")

print("Model Trained Successfully!")