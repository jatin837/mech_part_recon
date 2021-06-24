import os
import json

dat_dir: str = os.path.abspath("./dat/blnw-images-224")
categories: list = os.listdir(dat_dir)

print(categories)
