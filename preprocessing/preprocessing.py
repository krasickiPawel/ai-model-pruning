import os
import sys

import settings
import os_helper
from dataset_info import DATASETS_INFOS
import zip_helper


if len(sys.argv) <= 1:
    print("Please provide datasets source dir as first sys argument!")
    exit()

settings.DATASETS_SOURCE_DIR = sys.argv[1]
settings.DATASETS_RESIZED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets'))

os_helper.create_dir_if_not_exist(settings.DATASETS_RESIZED_DIR)

for di in DATASETS_INFOS:
    os_helper.create_dir_if_not_exist(os.path.join(settings.DATASETS_RESIZED_DIR, di.name, "0"))
    os_helper.create_dir_if_not_exist(os.path.join(settings.DATASETS_RESIZED_DIR, di.name, "1"))
    zip_file_path = os.path.join(settings.DATASETS_SOURCE_DIR, di.name, di.zip_file_name)
    zip_helper.extract_images_from_zip(zip_file_path, di, settings.DATASETS_RESIZED_DIR)


