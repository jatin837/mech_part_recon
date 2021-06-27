import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # To avoid tensorflow warning
from tensorflow.keras.utils import to_categorical
import cv2
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

from helpers import *
from this_model import *

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

    model = make_model(in_shape, kernel_size, num_of_filters)

    imgs: Imgs = Imgs()
    type(imgs)
    for cat in categories:
        imgs_list = os.listdir(f"{dat_dir}/{cat}")
        for img in imgs_list:
            img_path = os.path.join(dat_dir, cat, img)
            print(f'{len(imgs)} - loading {img_path}')
            img_to_dat = cv2.imread(img_path, 0)/255.0
            img_obj = Img(img_path, labels[cat], img_to_dat)
            imgs + img_obj
            print(imgs)
    breakpoint()

    data, labels = imgs.to_array()
    X_train, X_test, Y_train, Y_test = train_test_split(
            data, 
            labels, 
            train_size = 0.75, 
            test_size = 0.25, 
            random_state = 42
        )


    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)
    history = fit_model(X_train, Y_train,
                        X_test, Y_test,
                        epoch, batch_size,
                        model
                )

if __name__ == "__main__":
    main()
