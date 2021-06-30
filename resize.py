import os
import numpy as np
import cv2
from helpers import load_labels

labels_dir = os.path.abspath("./labels.json")
labels = load_labels(labels_dir)


resized_dir = "/home/dj4t9n/dev/mech_part_recon/dat/resized_data"
data_dir = "/home/dj4t9n/dev/mech_part_recon/dat/blnw-images-224"

categories = labels.keys()

count = 0
for cat in categories:
    for img in os.listdir(f"{data_dir}/{cat}"):
        img_abs_path = os.path.join(data_dir, cat, img)
        count = count + 1
        print(count)
        temp = cv2.imread(img_abs_path, 0)
        resized_temp = cv2.resize(temp, (64, 64))
        cv2.imwrite(f"{resized_dir}/{cat}/{str(count)}.png", resized_temp)

print("DONE")
