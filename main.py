import os
import json
import cv2
#import tensorflow as tf
import numpy as np

#from tensorflow.keras.models import Sequential
#from tensorflow.keras.utils import to_categorical
#from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D

labels_dir: str = os.path.abspath("./labels.json")

class Img(object):
    def __init__(self, path: str, label: str, data: np.array):
        self.path = path
        self.label = label
        self.data = data
    
    def __repr__(self):
        return f"at {self.path}, {self.label}"

with open(labels_dir, 'r') as f:
    labels: dict = json.load(f)

dat_dir: str = os.path.abspath("./dat/blnw-images-224")
categories: list = os.listdir(dat_dir)

_train: float = 0.75
_total: float= 1.00
train_imgs: dict = {}
test_imgs: dict = {}

for cat in categories:
    train_imgs[cat] = []
    test_imgs[cat] = []
    imgs = os.listdir(f"{dat_dir}/{cat}") 
    part = int(_train*len(imgs))
    for img in imgs[:part]:
        img_path = os.path.join(dat_dir, cat, img)
        img_to_dat = cv2.imread(img_path)
        print(img_to_dat)
        img_obj = Img(os.path.abspath(img_path), cat, cv2.imread(img_to_dat))
        train_imgs[cat].append(img_obj)
    for img in imgs[part:]:
        img_obj = Img(os.path.abspath(img), cat, cv2.imread(os.path.abspath(img)))
        test_imgs[cat].append(img_obj)

#model:Sequential = Sequential()

#model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
#model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#model.add(layers.Flatten())
#model.add(layers.Dense(64, activation='relu'))
#model.add(layers.Dense(10))
