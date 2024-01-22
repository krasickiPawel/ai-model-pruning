import glob
import os


def create_dir_if_not_exist(dir_path_to_create):
    if not os.path.exists(dir_path_to_create):
        os.makedirs(dir_path_to_create)


def delete_mask_files():
    files = glob.glob(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets')), "BreastUltrasoundImagesDataset(BUSI)", "**", "*mask*.png"))
    print("Deleted files:", files)
    for file in files:
        os.remove(file)
