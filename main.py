import os
import cv2
import numpy as np
from helpers import load_labels
def main() -> ():
    labels_dir: str = os.path.abspath("./labels.json")
    labels = load_labels(labels_dir)

    dat_dir: str = os.path.abspath("./dat/blnw-images-224")
    categories: list = os.listdir(dat_dir)
    img_array = []
    labels_array = []
    img_array_file = os.path.abspath("./imgs.npy")
    img_labels_file = os.path.abspath("./labels.npy")
    for cat in categories:
        imgs_list = os.listdir(f"{dat_dir}/{cat}")
        for img in imgs_list:
            img_path = os.path.join(dat_dir, cat, img)
            print(f'loading {img_path}')
            img_to_dat = cv2.imread(img_path, 0)/255.0
            img_array.append(img_to_dat)
            labels_array.append(labels[cat])
    
    img_array = np.array(img_array)
    labels_array = np.array(labels_array)



if __name__ == "__main__":
    main()
