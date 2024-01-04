import os
import sys
import settings
import os_helper
import zip_helper
from dataset_info import DatasetInfo


if len(sys.argv) <= 1:
    print("Please provide datasets source dir as first sys argument!")
    exit()

settings.DATASETS_SOURCE_DIR = sys.argv[1]
settings.DATASETS_RESIZED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets'))

os_helper.create_dir_if_not_exist(settings.DATASETS_RESIZED_DIR)


DATASETS_INFOS = [
    DatasetInfo(
        name="BreaKHis400X",
        zip_file_name=settings.DEFAULT_ZIP_FILE_NAME,
        benign_name="benign",
        malignant_name="malignant"
    ),
    DatasetInfo(
        name="BreastUltrasoundImagesDataset(BUSI)",
        zip_file_name=settings.DEFAULT_ZIP_FILE_NAME,
        benign_name="benign",
        malignant_name="malignant",
        normal_name="normal"
    ),
    DatasetInfo(
        name="UltrasoundBreastImagesforBreastCancer",
        zip_file_name=settings.DEFAULT_ZIP_FILE_NAME,
        benign_name="benign",
        malignant_name="malignant"
    ),
    DatasetInfo(
        name="BreastHistopathologyImages",
        zip_file_name=settings.DEFAULT_ZIP_FILE_NAME,
        malignant_name="class1",
        normal_name="class0"
    )
]
DATASETS_INFOS.pop(-1)


for di in DATASETS_INFOS:
    os_helper.create_dir_if_not_exist(os.path.join(settings.DATASETS_RESIZED_DIR, di.name, "0"))
    os_helper.create_dir_if_not_exist(os.path.join(settings.DATASETS_RESIZED_DIR, di.name, "1"))
    zip_file_path = os.path.join(settings.DATASETS_SOURCE_DIR, di.name, di.zip_file_name)
    zip_helper.extract_images_from_zip(zip_file_path, di, settings.DATASETS_RESIZED_DIR)

os_helper.delete_mask_files()
