import os
import json
import cv2
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D

labels_dir: str = os.path.abspath("./labels.json")

with open(labels_dir, 'r') as f:
    labels: dict = json.load(f)

dat_dir: str = os.path.abspath("./dat/blnw-images-224")
categories: list = os.listdir(dat_dir)

_train: float = 0.75
_total: float= 1.00
train_imgs: dict = {}
test_imgs: dict = {}

for cat in categories:
    imgs = os.listdir(f"{dat_dir}/{cat}") 
    part = int(_train*len(imgs))
    train_imgs[cat] = imgs[:part]
    test_imgs[cat] = imgs[part:]

print(train_imgs, test_imgs)

#model:Sequential = Sequential()

#model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
#model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#model.add(layers.Flatten())
#model.add(layers.Dense(64, activation='relu'))
#model.add(layers.Dense(10))
