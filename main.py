#!/home/dj4t9n/dev/mech_part_recon/env/bin/python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # To avoid tensorflow warning
from helpers import *

import tensorflow as tf

from this_model import *
import cv2
import argparse
import json

labels = load_labels("./labels.json")
categories = list(labels.keys())

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="Path to file")

args = vars(ap.parse_args())

path = os.path.abspath(args["path"])

img = cv2.imread(path, 0)
img = img/255
img = img.reshape(1, 64, 64, 1)

model = tf.keras.models.load_model("model.h5")

prediction = list(model.predict(img))

indx = prediction.index(max(prediction))
print(categories[indx])
