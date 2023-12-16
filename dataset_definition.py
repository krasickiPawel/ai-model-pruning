from PIL import Image, ImageEnhance
from torch.utils.data import Dataset


class FingerTrainTestDataset(Dataset):
    def __init__(self, paths, labels, trans=None) -> None:
        super().__init__()
        self.paths = paths
        self.labels = labels
        self.transforms = trans

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img_path = self.paths[index]
        img_label = self.labels[index]
        img = Image.open(img_path).convert('RGB')
        # img = Image.open(img_path).convert('L')
        img_filter = ImageEnhance.Color(img)
        img_filter.enhance(0)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, img_label


class FingerValDataset(Dataset):
    def __init__(self, paths, trans=None) -> None:
        super().__init__()
        self.paths = paths
        self.transforms = trans

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img_path = self.paths[index]
        img = Image.open(img_path).convert('RGB')
        # img = Image.open(img_path).convert('L')
        img_filter = ImageEnhance.Color(img)
        img_filter.enhance(0)

        if self.transforms is not None:
            img = self.transforms(img)
        return img, img_path
