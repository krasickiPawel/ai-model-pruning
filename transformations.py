import torchvision.transforms as transforms
from PIL import Image


def get_transform_rgb():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_transform_grayscale():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ]
    )


def get_valid_transformer(img_sample_path):
    with Image.open(img_sample_path) as img:
        if img.mode == 'RGB':
            return get_transform_rgb()
        elif img.mode == 'L':
            return get_transform_grayscale()
        else:
            raise ValueError(f"Unsupported image mode: {img.mode}")
