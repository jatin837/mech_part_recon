import os
import json
import cv2
import tensorflow as tf

from tf.keras.models import Sequential
from tf.keras.utils import to_categorical
from tf.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D

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
   pass ## TODO -> for each category, store dictionary of each image along with it's label
