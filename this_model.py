import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # To avoid tensorflow warning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D

def make_model(in_shape: tuple, kernel_size: tuple, num_of_filters:int) -> Sequential:
    model = Sequential()
    model.add(Conv2D(num_of_filters, kernel_size, input_shape=in_shape)) #input_shape matches our input image
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(num_of_filters, kernel_size))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Dense(4)) #data of four types
    model.add(Activation('softmax'))
    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy']
    )
    return model


def fit_model(X_train: np.array, Y_train: np.array,
              X_test: np.array, Y_test: np.array,
              epoch: int, batch_size: int,
              model: Sequential) -> ():

    history = model.fit(
        X_train,
        Y_train,
        batch_size,
        epoch,
        validation_data=(X_test, Y_test)
    )
    return history

