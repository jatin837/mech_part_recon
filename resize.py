import os
import numpy as np
import cv2
from helpers import load_labels

labels_dir = os.path.abspath("./labels.json")
labels = load_labels(labels_dir)


data_dir = "/home/dj4t9n/dev/mech_part_recon/dat/resized_data"
