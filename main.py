import os
import cv2
import numpy as np
from helpers import load_labels
def main() -> ():
    labels_dir: str = os.path.abspath("./labels.json")
    labels = load_labels(labels_dir)

    dat_dir: str = os.path.abspath("./dat/blnw-images-224")
    categories: list = os.listdir(dat_dir)

    for cat in categories:
        imgs_list = os.listdir(f"{dat_dir}/{cat}")
        for img in imgs_list:
            img_path = os.path.join(dat_dir, cat, img)
            print(f'loading {img_path}')
            img_to_dat = cv2.imread(img_path, 0)/255.0


if __name__ == "__main__":
    main()
