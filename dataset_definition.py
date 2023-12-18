import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class BreastDataset(Dataset):
    def __init__(self, paths, labels, trans=None) -> None:
        super().__init__()
        self.paths = np.array(paths)
        self.labels = np.array(labels)
        self.transforms = trans

    def __len__(self):
        return self.paths.size

    def __getitem__(self, index):
        img_path = self.paths[index]
        img_label = self.labels[index]
        img = Image.open(img_path)
        # img = Image.open(img_path).convert('RGB')
        # img = Image.open(img_path).convert('L')
        # img_filter = ImageEnhance.Color(img)
        # img_filter.enhance(0)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, img_label

