import os
import json

import tensorflow as tf

from tf.keras.models import Sequential
from tf.keras.utils import to_categorical
from tf.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D

labels_dir: str = os.path.abspath("./labels.json")

with open(labels_dir, 'r') as f:
    labels: dict = json.load(f)

dat_dir: str = os.path.abspath("./dat/blnw-images-224")
categories: list = os.listdir(dat_dir)

## TODO :
## Creation of CNN. Sequential Model

model = Sequential()


