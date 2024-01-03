import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class BreastDataset(Dataset):
    def __init__(self, paths, labels, trans):
        super().__init__()
        self.paths = np.array(paths)
        self.labels = np.array(labels)
        self.transforms = trans

    def __len__(self):
        return self.paths.size

    def __getitem__(self, index):
        img_path = self.paths[index]
        img_label = self.labels[index]

        with Image.open(img_path) as image:
            img = self.transforms(image)

        return img, img_label
