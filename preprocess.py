import numpy as np
import cv2
import os

data_file = "./X.npy"
labels_file = "./Y.npy"

dat_dir = os.path.abspath("./dat/resized_data/")

labels = {
	"locatingpin": 0,
	"washer": 1,
	"bolt": 2, 
	"nut": 3
    } 

X = []
Y = []
categories = list(labels.keys())

for cat in categories:
    imgs_list = os.listdir(f"{dat_dir}/{cat}")
    for img in imgs_list:
        Y.append(labels[cat])
        img_path = os.path.join(dat_dir, cat, img)
        img_to_dat = cv2.imread(img_path, 0)/255.0
        X.append(img_to_dat.reshape(64, 64, 1))
        print(len(X))

np.save(data_file, X)
np.save(labels_file, Y)

