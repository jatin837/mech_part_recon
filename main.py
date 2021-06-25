import os
import json
import cv2
import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D

labels_dir: str = os.path.abspath("./labels.json")

## model parameters
in_shape: tuple = (224, 224, 1)
num_of_filters: int = 64
kernel_size: tuple = (3, 3)

#training hyper parameters
epoch: int = 15
batch_size: int = 64

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
        print(f'loading {img_path}')
        img_to_dat = cv2.imread(img_path, 0)
        img_obj = Img(os.path.abspath(img_path), cat, img_to_dat)
        train_imgs[cat].append(img_obj)
    for img in imgs[part:]:
        img_path = os.path.join(dat_dir, cat, img)
        print(f'loading {img_path}')
        img_to_dat = cv2.imread(img_path, 0)
        img_obj = Img(img_path, cat, img_to_dat)
        test_imgs[cat].append(img_obj)

#Creation of a CNN . Sequential Model
def make_model(in_shape: tuple, kernel_size: tuple, num_of_filters:int) -> tf.keras.model.Sequential:
    model = Sequential()
    model.add(Conv2D(64, (3,3), input_shape=(224, 224, 1))) #input_shape matches our input image
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Dense(4)) #data of four types
    model.add(Activation('softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
    return model

def fit_model(X_train: np.array, Y_train: np.array, X_test: np.array, Y_test: np.array, epoch: int, batch_size: int, model: tf.keras.model.Sequential) -> ():
    history = model.fit(X_train, Y_train, batch_size,
                        epochs,
                        validation_data=(X_test, Y_test)
                    ) #Actual Training of model
