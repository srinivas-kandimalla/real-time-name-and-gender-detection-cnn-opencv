import os
import cv2
import json
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.utils import to_categorical

DATASET_PATH = "dataset"

images = []
labels = []

label_map = {}
current_label = 0

for person in os.listdir(DATASET_PATH):

    person_path = os.path.join(DATASET_PATH, person)

    if not os.path.isdir(person_path):
        continue

    label_map[current_label] = person

    for image_name in os.listdir(person_path):

        img_path = os.path.join(person_path, image_name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (100, 100))

        images.append(img)

        labels.append(current_label)

    current_label += 1

X = np.array(images)
X = X.reshape(-1, 100, 100, 1)
X = X / 255.0

y = to_categorical(labels)

model = Sequential([
    Conv2D(32, (3,3), activation='relu',
           input_shape=(100,100,1)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),

    Dense(128, activation='relu'),

    Dense(y.shape[1], activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X,
    y,
    epochs=10,
    batch_size=16
)

os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

model.save("models/face_model.h5")

with open("data/labels.json", "w") as f:
    json.dump(label_map, f)

print("Model saved successfully!")