import os
import json
import cv2
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D

class Img(object):
    def __init__(self, path: str, label: str, data: np.array):
        self.path = path
        self.label = label
        self.data = data
    
    def __repr__(self):
        return f"at {self.path}, {self.label}"

class Imgs(object):
    def __init__(self):
        self.imgs = []

    def __add__(self, img):
        self.imgs.append(img)

    def __len__(self):
        return len(self.imgs)

    def __repr__(self):
        return f"total images so far = {len(self)}"

def load_labels(labels_dir) -> dict:
    with open(labels_dir, 'r') as f:
        labels: dict = json.load(f)
    return labels

#Creation of a CNN . Sequential Model
def make_model(in_shape: tuple, kernel_size: tuple, num_of_filters:int) -> Sequential:
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
    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adam(),metrics=['accuracy']
    )
    return model

def fit_model(X_train: np.array, Y_train: np.array, X_test: np.array, Y_test: np.array, epoch: int, batch_size: int, model: Sequential) -> ():
    history = model.fit(X_train, Y_train, batch_size,
                        epochs,
                        validation_data=(X_test, Y_test)
                    ) #Actual Training of model


def main() -> ():
    labels_dir: str = os.path.abspath("./labels.json")
    labels = load_labels(labels_dir)

    dat_dir: str = os.path.abspath("./dat/blnw-images-224")
    categories: list = os.listdir(dat_dir)

    _train: float = 0.75

    ## model parameters
    in_shape: tuple = (224, 224, 1)
    num_of_filters: int = 64
    kernel_size: tuple = (3, 3)

    #training hyper parameters
    epoch: int = 15
    batch_size: int = 64

    imgs: Imgs = Imgs()
    type(imgs)
    for cat in categories:
        imgs_list = os.listdir(f"{dat_dir}/{cat}")
        for img in imgs_list:
            img_path = os.path.join(dat_dir, cat, img)
            print(f'{len(imgs)} - loading {img_path}')
            img_to_dat = cv2.imread(img_path, 0)
            img_obj = Img(img_path, cat, img_to_dat)
            imgs + img_obj
            print(imgs)

if __name__ == "__main__":
    main()
