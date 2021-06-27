import json

def load_labels(labels_dir) -> dict:
    with open(labels_dir, 'r') as f:
        labels: dict = json.load(f)
    return labels

class Img(object):
    def __init__(self, path: str, label: str, data: np.array):
        self.path = path
        self.label = label
        self.data = data
    
    def __repr__(self):
        return f"at {self.path}, {self.label}"

class Imgs(object):
    def __init__(self):
        self.imgs = []

    def __add__(self, img):
        self.imgs.append(img)

    def __len__(self):
        return len(self.imgs)

    def __repr__(self):
        return f"total images so far = {len(self)}"

    def to_array(self):
        img_mat = np.array([img.data for img in self.imgs])
        labels = np.array([img.label for img in self.imgs])
        return img_mat, labels


