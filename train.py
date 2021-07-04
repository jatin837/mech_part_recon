import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # To avoid tensorflow warning
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from this_model import *


def main() -> ():
    in_shape: tuple = (64, 64, 1)
    num_of_filters: int = 64
    kernel_size: tuple = (3, 3)

    #training hyper parameters
    epoch: int = 15
    batch_size: int = 64

    model = make_model(in_shape, kernel_size, num_of_filters)

    data = np.load("X.npy")
    labels = np.load("Y.npy")

    X_train, X_test, Y_train, Y_test = train_test_split(
            data, 
            labels, 
            test_size = 0.20, 
            random_state = 1,
            shuffle = True
        )


    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)
    history = fit_model(X_train, Y_train,
                        X_test, Y_test,
                        epoch, batch_size,
                        model
                )
    history_file = os.path.abspath("./history.npy")

    np.save(history_file ,history.history)
    

if __name__ == "__main__":
    main()
    print("EVERYTHING DONE")
