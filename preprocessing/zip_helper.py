import os
import settings
import numpy as np
from zipfile import ZipFile

from PIL import Image


def extract_images_from_zip(zip_file_path, dataset_info, resized_images_dir_path):
    with ZipFile(zip_file_path, "r") as zip_file:
        for file_path in zip_file.namelist():
            img_destination_dir = os.path.join(resized_images_dir_path, dataset_info.name, str(int(dataset_info.malignant_name in file_path)))
            new_file_path = os.path.join(img_destination_dir, f"resized_{os.path.split(file_path)[1]}")
            with zip_file.open(file_path) as opened_zip_file_path:
                with Image.open(opened_zip_file_path) as img:
                    sqrWidth = np.ceil(np.sqrt(img.size[0] * img.size[1])).astype(int)
                    img_resized = img.resize((sqrWidth, sqrWidth))
                    img_resized.thumbnail(settings.IMG_RESIZED_SIZE, Image.LANCZOS)
                    img_resized.save(new_file_path, "PNG")
