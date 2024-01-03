import glob
import os


def delete_redundant_files():
    files = glob.glob(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets')), "BreastUltrasoundImagesDataset(BUSI)", "**", "*mask*.png"))
    print("Deleted files:", files)
    for file in files:
        os.remove(file)
